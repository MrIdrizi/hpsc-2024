//
//  main.cpp
//  Final Report HPSC
//
//  Created by Florian Idrizi on 27.05.24.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

using namespace std;

void printMatrix(const vector<vector<double>>& matrix) {
    cout << "beginning\n";
    cout << "[\n";
    for (const auto& row : matrix) {
        cout << " [";
        for (const auto& elem : row) {
            cout << setw(4) << elem << " ";
        }
        cout << "]\n";
    }
    cout << "]\n";
    cout << "end\n";
}

int main()
{
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2.0 / double(nx - 1);
    double dy = 2.0 / double(ny - 1);
    double dt = 0.01;
    int rho = 1;
    double nu = 0.02;
    
    vector<vector<double>> u(ny, vector<double>(nx, 0.0));
    vector<vector<double>> v(ny, vector<double>(nx, 0.0));
    vector<vector<double>> p(ny, vector<double>(nx, 0.0));
    vector<vector<double>> b(ny, vector<double>(nx, 0.0));
    
    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    for(int n = 0; n < nt; n++)
    {
        for(int j = 1; j < (ny - 1); j++)
        {
            for(int i = 1; i < (nx - 1); i++)
            {
                b[j][i] = rho * (1.0 / dt * ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) 
                                 - pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2)
                                 - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx))
                                 - pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2));
            }
        }
        for (int it = 0; it < nit; it++)
        {
            vector<vector<double>> pn = p;
            
            for (int j = 1; j < ny - 1; j++)
            {
                for (int i = 1; i < nx - 1; i++)
                {
                    p[j][i] = (dy * dy * (pn[j][i+1] + pn[j][i-1]) +
                               dx * dx * (pn[j+1][i] + pn[j-1][i]) -
                               b[j][i] * dx * dx * dy * dy) /
                               (2 * (dx * dx + dy * dy));
                }
            }

            for (int j = 0; j < ny; j++)
            {
                p[j][nx - 1] = p[j][nx - 2];
                p[j][0] = p[j][1];
            }
            for (int i = 0; i < nx; i++)
            {
                p[0][i] = p[1][i];
                p[ny - 1][i] = 0.0;
            }
        }
        std::vector<std::vector<double>> un = u;
        std::vector<std::vector<double>> vn = v;
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = 1; i < nx - 1; i++) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
                          un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
                          dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) +
                          nu * dt / (dx * dx) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) +
                          nu * dt / (dy * dy) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);

                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
                          vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
                          dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i]) +
                          nu * dt / (dx * dx) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) +
                          nu * dt / (dy * dy) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
            }
        }

        for (int j = 0; j < ny; j++)
        {
            u[j][0] = 0;
            u[j][nx - 1] = 0;
            v[j][0] = 0;
            v[j][nx - 1] = 0;
        }
        
        for (int i = 0; i < nx; i++)
        {
            u[0][i] = 0;
            u[ny - 1][i] = 1;
            v[0][i] = 0;
            v[ny - 1][i] = 0;
        }
        /*if(n == 499){
            cout << "Matrix p: " << n << endl;
            printMatrix(b);
        }*/
        if (n % 10 == 0) {
             for (int j=0; j<ny; j++)
               for (int i=0; i<nx; i++)
                 ufile << u[j][i] << " ";
             ufile << "\n";
             for (int j=0; j<ny; j++)
               for (int i=0; i<nx; i++)
                 vfile << v[j][i] << " ";
             vfile << "\n";
             for (int j=0; j<ny; j++)
               for (int i=0; i<nx; i++)
                 pfile << p[j][i] << " ";
             pfile << "\n";
        }
    }
    ufile.close();
    vfile.close();
    pfile.close();
    return 0;
}
