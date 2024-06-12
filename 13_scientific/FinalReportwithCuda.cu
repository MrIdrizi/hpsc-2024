//
//  main.cpp
//  HPSC Final with CUDA
//
//  Created by Florian Idrizi on 28.05.24.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <chrono>

using namespace std;


__global__ void initialize(double* mat, int nx, int ny)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > 0 && j < nx && i > 0 && i < ny) {
        mat[j*nx+i] = 0.0;
    }
}

__global__ void loopbji(double* b, double* u, double* v, double dx, double dy, double dt, int rho, int nx, int ny)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j > 0 && j < ny - 1 && i > 0 && i < nx - 1) {
        b[j*nx+i] = rho * (1.0 / dt * ((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx) + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy))
                            - pow((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx), 2)
                            - 2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2 * dy) * (v[j*nx+i+1] - v[j*nx+i-1]) / (2 * dx))
                            - pow((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy), 2));
    }
}

__global__ void looppji(double* p, double* pn, double* b, double dx, double dy, int nx, int ny) 
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j > 0 && j < ny - 1 && i > 0 && i < nx - 1) {
        p[j*nx+i] = (dy * dy * (pn[j*nx+i+1] + pn[j*nx+i-1]) +
                         dx * dx * (pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) -
                         b[j*nx+i] * dx * dx * dy * dy) /
                         (2 * (dx * dx + dy * dy));
    }
}

__global__ void boundaryp1(double* p, int nx, int ny)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j < ny) {
        p[j*nx+nx-1] = p[j*nx+nx-2];
        p[j*nx] = p[j*nx+1];
    }
}

__global__ void boundaryp2(double* p, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nx) {
        p[i] = p[nx+i];
        p[(ny-1)*nx+i] = 0.0;
    }
}

__global__ void loopujivji(double* u, double* v, double* un, double* vn, double* p, double dx, double dy, double dt, int rho, double nu, int nx, int ny)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j > 0 && j < ny - 1 && i > 0 && i < nx - 1) {
        u[j*nx+i] = un[j*nx+i] - un[j*nx+i] * dt / dx * (un[j*nx+i] - un[j*nx+i-1]) -
                        un[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j-1)*nx+i]) -
                        dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1]) +
                        nu * dt / (dx * dx) * (un[j*nx+i+1] - 2 * un[j*nx+i] + un[j*nx+i-1]) +
                        nu * dt / (dy * dy) * (un[(j+1)*nx+i] - 2 * un[j*nx+i] + un[(j-1)*nx+i]);

        v[j*nx+i] = vn[j*nx+i] - vn[j*nx+i] * dt / dx * (vn[j*nx+i] - vn[j*nx+i-1]) -
                        vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j-1)*nx+i]) -
                        dt / (2 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i]) +
                        nu * dt / (dx * dx) * (vn[j*nx+i+1] - 2 * vn[j*nx+i] + vn[j*nx+i-1]) +
                        nu * dt / (dy * dy) * (vn[(j+1)*nx+i] - 2 * vn[j*nx+i] + vn[(j-1)*nx+i]);
    }
}

__global__ void boundaryuv1(double* u, double* v, int nx, int ny)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j < ny) {
        u[j*nx] = 0;
        u[j*nx+nx-1] = 0;
        v[j*nx] = 0;
        v[j*nx+nx-1] = 0;
    }
}

__global__ void boundaryuv2(double* u, double* v, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < nx) {
        u[i] = 0;
        u[(ny-1)*nx+i] = 1;
        v[i] = 0;
        v[(ny-1)*nx+i] = 0;
    }
}

int main()
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2.0 / double(nx - 1);
    double dy = 2.0 / double(ny - 1);
    double dt = 0.01;
    int rho = 1;
    double nu = 0.02;
    
    double *u;
    double *v;
    double *p;
    double *b;
    double *un;
    double *vn;
    double *pn;

    cudaMallocManaged(&u, nx * ny * sizeof(double));
    cudaMallocManaged(&v, nx * ny * sizeof(double));
    cudaMallocManaged(&p, nx * ny * sizeof(double));
    cudaMallocManaged(&b, nx * ny * sizeof(double));
    cudaMallocManaged(&un, nx * ny * sizeof(double));
    cudaMallocManaged(&vn, nx * ny * sizeof(double));
    cudaMallocManaged(&pn, nx * ny * sizeof(double));
    
    
    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    dim3 DimBlock(32, 32, 1);
    dim3 numBlocks((nx + DimBlock.x - 1)/DimBlock.x, (ny + DimBlock.y - 1)/DimBlock.y, 1);
    
    initialize<<<numBlocks, DimBlock>>>(u, nx, ny);
    initialize<<<numBlocks, DimBlock>>>(v, nx, ny);
    initialize<<<numBlocks, DimBlock>>>(p, nx, ny);
    initialize<<<numBlocks, DimBlock>>>(b, nx, ny);
    cudaDeviceSynchronize();

    for(int n = 0; n < nt; n++)
    {
        loopbji<<<numBlocks, DimBlock>>>(b, u, v, dx, dy, dt, rho, nx, ny);
        cudaDeviceSynchronize();

        for (int it = 0; it < nit; it++)
        {
            cudaMemcpy(pn, p, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);

            looppji<<<numBlocks, DimBlock>>>(p, pn, b, dx, dy, nx, ny);
            cudaDeviceSynchronize();
            
            boundaryp1<<<dim3(1, (ny+31)/32, 1), dim3(1, 32, 1)>>>(p, nx, ny);
            cudaDeviceSynchronize();
            
            boundaryp2<<<dim3((nx+31)/32, 1, 1), dim3(32, 1, 1)>>>(p, nx, ny);
            cudaDeviceSynchronize();
        }

        cudaMemcpy(un, u, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(vn, v, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);

        loopujivji<<<numBlocks, DimBlock>>>(u, v, un, vn, p, dx, dy, dt, rho, nu, nx, ny);
        cudaDeviceSynchronize();
        
        boundaryuv1<<<dim3(1, (ny+31)/32, 1), dim3(1, 32, 1)>>>(u, v, nx, ny);
        cudaDeviceSynchronize();
                     
        boundaryuv2<<<dim3((nx+31)/32, 1, 1), dim3(32, 1, 1)>>>(u, v, nx, ny);
        cudaDeviceSynchronize();

        if (n % 10 == 0) {
             for (int j=0; j<ny; j++)
               for (int i=0; i<nx; i++)
                 ufile << u[j*nx+i] << " ";
             ufile << "\n";
             for (int j=0; j<ny; j++)
               for (int i=0; i<nx; i++)
                 vfile << v[j*nx+i] << " ";
             vfile << "\n";
             for (int j=0; j<ny; j++)
               for (int i=0; i<nx; i++)
                 pfile << p[j*nx+i] << " ";
             pfile << "\n";
        }
    }

    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);
    
    std::chrono::duration<double> total = std::chrono::high_resolution_clock::now() - startTime;
    printf("\ntime = %f\n", total.count());

    ufile.close();
    vfile.close();
    pfile.close();
    return 0;
}
