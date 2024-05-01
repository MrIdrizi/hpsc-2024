#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
    // Example array of float values
    float a[16] = {0.1f, 0.5f, 0.01f, 0.0f, 0.001f, 0.7f, 0.008f, 0.0f, 0.02f, 0.0f, 0.03f, 0.0f, 0.4f, 0.0f, 0.001f, 0.1f};

    // Load the array into an AVX-512 vector
    __m512 rinvvec2 = _mm512_load_ps(a);

    // Set epsilon value
    float epsilon = 0;
    __m512 epsvec = _mm512_set1_ps(epsilon);
    // Create mask to check which elements are less than epsilon
    __mmask16 mask = _mm512_cmp_ps_mask(rinvvec2, epsvec, _MM_CMPINT_LE);
	
    // Output the mask
    printf("Mask:\n");
    for (int i = 15; i >= 0; --i) {
        printf("%d", (mask >> i) & 1);
    }

    printf("\n");
    // Create a vector with all elements set to 1.0
    __m512 one_vec = _mm512_set1_ps(1.0f);

    // Blend the vectors based on the mask
    __m512 updated_rinvvec2 = _mm512_mask_blend_ps(mask, one_vec, rinvvec2);

    // Output the original and updated vectors
    printf("Original vector:\n");
    for (int i = 0; i < 16; ++i) {
        printf("%f ", a[i]);
    }
    printf("\n");

    printf("Updated vector:\n");
    float updated_a[16];
    _mm512_store_ps(updated_a, updated_rinvvec2);
    for (int i = 0; i < 16; ++i) {
        printf("%f ", updated_a[i]);
    }
    printf("\n");    
    return 0;
}
