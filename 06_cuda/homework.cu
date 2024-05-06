#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void printArray(int *a){
  int i = threadIdx.x;
  printf("%d: %d\n", i, a[i]);
  __syncthreads();
}

__global__ void initialize(int *bucket){
  int i = threadIdx.x;
  bucket[i] = 0;
}

__global__ void  bucketAdd(int *bucket, int *key){
  int i = threadIdx.x;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void sort(int *key, int *bucket, int *offset, int range, int n){
  int i = threadIdx.x;
  for (int j = 1; j<range; j<<=1) {
    offset[i] = bucket[i];
    __syncthreads();
    if(i>=j) bucket[i] += offset[i-j];
    __syncthreads();
  }

  int start = 0;
  if(i > 0){
    start = bucket[i-1];
  }
  for (int j = start; j<bucket[i]; j++){
    key[j] = i;
  }
}

int main() {
  const int n = 50;
  const int range = 5;

  int *key;
  cudaMallocManaged(&key, range*sizeof(int));
 
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  initialize<<<1,range>>>(bucket);
  cudaDeviceSynchronize(); 
  bucketAdd<<<1,n>>>(bucket, key);
  cudaDeviceSynchronize();
  int *offset;
  cudaMallocManaged(&offset, range*sizeof(int));  
  sort<<<1,range>>>(key, bucket, offset, range, n);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
  cudaFree(offset);
}

