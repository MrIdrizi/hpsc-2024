#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

//function to print the _m512 Vectors
void printvec(__m512 vec) {
    float output[16];
    _mm512_storeu_ps(output, vec);
    printf("Vektor");
    for (int i = 0; i < 16; i++) {
      printf("%d: %f\n", i, output[i]);
    }
}

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N]; 
  float mask[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    mask[i] = 0;
  }

  for(int i=0; i<N; i++) {
    //create mask for  condition
    mask[i] = 1;
    __m512 maskvec = _mm512_load_ps(mask);
    //x[N] vector
    __m512 xvec = _mm512_load_ps(x);
    //y[N] vector
    __m512 yvec = _mm512_load_ps(y);
    //m[N] vector 
    __m512 mvec = _mm512_load_ps(m);
    //Vector in which all Elements are x[i]
    __m512 xivec = _mm512_set1_ps(x[i]);
    //rx=  x[i]-x[j]
    __m512 rxvec = _mm512_sub_ps(xivec,xvec);
    //Vector in which all Elements are y[i]
    __m512 yivec = _mm512_set1_ps(y[i]);
    //ry=  y[i]-y[j]
    __m512 ryvec = _mm512_sub_ps(yivec,yvec);
    //r2 = rx * rx + ry * ry
    __m512 rxvecquad = _mm512_mul_ps(rxvec, rxvec);
    __m512 ryvecquad = _mm512_mul_ps(ryvec, ryvec);
    __m512 rvec2 = _mm512_add_ps(rxvecquad,ryvecquad);
    //add mask to r2 so that we wont devide by zero(wont change the result since the dominator will always be 0 for i = j)
    rvec2 = _mm512_add_ps(rvec2,maskvec); 
    __m512 rinvvec = _mm512_rsqrt14_ps(rvec2);
    //1/r^3
    __m512 rinv2vec = _mm512_mul_ps(rinvvec, rinvvec);
    __m512 rinv3vec = _mm512_mul_ps(rinv2vec, rinvvec);
    //rx*mj*1/r3
    __m512 rxmvec = _mm512_mul_ps(rxvec, mvec);
    __m512 fxreduce = _mm512_mul_ps(rxmvec, rinv3vec);
    //ry*mj*1/r3
    __m512 rymvec= _mm512_mul_ps(ryvec, mvec);
    __m512 fyreduce = _mm512_mul_ps(rymvec, rinv3vec);

    fx[i] -= _mm512_reduce_add_ps(fxreduce);
    fy[i] -= _mm512_reduce_add_ps(fyreduce);
    printf("%d %g %g\n",i,fx[i],fy[i]);
    mask[i] = 0;
  }
}
