#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <ctime>
#include <chrono>
using namespace std;


static const int64_t _two31 = INT64_C(1) << 31; // 2^31
static const int64_t _two32 = INT64_C(1) << 32; // 2^32
typedef uint32_t Torus32; 
const int n = 630;
const int Msize = 8;
const int BLOCK_SIZE = 128;
const int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
const uint32_t inter = uint32_t(_two31 / Msize * 2);

int sign(float d)
{
    if (d > 0)
    {
        return 1;
    }
    else if(d < 0)
    {
        return -1;
    }
    else{
        return 0;
    }
}

uint32_t mutoT(int mu,int Msize)
{
    return uint32_t(_two31 / Msize * 2 * mu);
}

Torus32 dtot32(float d)
{
    int dsign = sign(d);
    return Torus32(round(fmod(d * dsign, 1) * _two32) * dsign);
}

uint32_t gaussian32(uint32_t mu, float alpha, int size = 1)
{
    std::default_random_engine generator;
    std::normal_distribution<> distribution {0, alpha};
    return uint32_t(dtot32(distribution(generator)) + mu);
}

uint32_t Ttomu(uint32_t phase, uint32_t inter)
{
    uint32_t half = uint32_t(inter / 2);
    return uint32_t(uint32_t(phase + half) / inter);
}

void tlweKeyGen(int32_t *h_tlwekey)
{
    srand((int)time(NULL));
    for (int i = 0; i < n; i++)
    {
        h_tlwekey[i] = rand() % 2;
    }
}

__global__ void dot(int32_t *d_tlwekey, uint32_t *d_a, uint64_t *d_product)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = blockDim.x * bid + tid;
    __shared__ uint64_t s_product[128];   
    s_product[tid] = (idx < n) ? d_tlwekey[idx] * d_a[idx] : 0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(tid < offset){
            s_product[tid] += s_product[tid + offset];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        d_product[bid] = s_product[0];
    }
}

void tlweSymEnc(uint32_t mu, float alpha, int32_t *d_tlwekey, uint32_t *h_a, uint32_t *d_a, uint32_t &cipher)
{  
    const int ymem = sizeof(uint64_t) * GRID_SIZE;
    uint64_t *d_product, *h_product;
    cudaMalloc(&d_product, ymem);
    h_product = (uint64_t *) malloc(ymem);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    for (int i = 0; i < n; i++)
    {
        h_a[i] = g();
    }
    cudaMemcpy(d_a, h_a, sizeof(uint32_t) * n, cudaMemcpyHostToDevice);
    

    dot<<<GRID_SIZE, BLOCK_SIZE>>>(d_tlwekey, d_a, d_product);
    cudaMemcpy(h_product, d_product, ymem, cudaMemcpyDeviceToHost);
    uint64_t innerproduct = 0;
    for(int i = 0; i < GRID_SIZE; i++)
    {
        innerproduct += h_product[i];
    }
    uint32_t ga = gaussian32(mu, alpha);
    cipher = uint32_t(ga - innerproduct);

    cudaFree(d_product);
    free(h_product);
}

uint32_t tlweSymDec(uint32_t cipher, uint32_t *d_a, int32_t *d_tlwekey)
{
    const int ymem = sizeof(uint64_t) * GRID_SIZE;
    uint64_t *d_product, *h_product;
    cudaMalloc(&d_product, ymem);
    h_product = (uint64_t *) malloc(ymem);
    dot<<<GRID_SIZE, BLOCK_SIZE>>>(d_tlwekey, d_a, d_product);
    cudaMemcpy(h_product, d_product, ymem, cudaMemcpyDeviceToHost);
    uint64_t innerproduct = 0;
    for(int i = 0; i < GRID_SIZE; i++)
    {
        innerproduct += h_product[i];
    }
    uint32_t phase = cipher + innerproduct;

    cudaFree(d_product);
    free(h_product);

    return Ttomu(phase, inter);
}

void Test(int msg)
{
    uint32_t mu = mutoT(msg, Msize);
    int32_t *h_tlwekey, *d_tlwekey;
    uint32_t *h_a, *d_a;
    float alpha = pow(2.0, -15.4);
    uint32_t cipher;
    h_tlwekey =(int32_t*)malloc(n * sizeof(int32_t));
    h_a =(uint32_t*)malloc(n * sizeof(uint32_t));
    cudaMalloc(&d_tlwekey, n * sizeof(int32_t));
    cudaMalloc(&d_a, n * sizeof(uint32_t));


    tlweKeyGen(h_tlwekey);
    
    cudaMemcpy(d_tlwekey, h_tlwekey, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    

    tlweSymEnc(mu, alpha, d_tlwekey, h_a, d_a, cipher);
    cout << "msg:" << msg << endl;
    cout << "decmsg:" << tlweSymDec(cipher, d_a, d_tlwekey) << endl;


    cudaFree(d_tlwekey);
    cudaFree(d_a);
    free(h_tlwekey);

}

int main()
{
    for(int i = 0; i < 8; i++)
    {
        Test(i);
    }
    return 0;
}