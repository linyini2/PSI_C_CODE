#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <ctime>
#include <chrono>
#include<typeinfo>
#include "error.cuh"
#include "cufft.h"
#include <cufftXt.h>
using namespace std;

static const int64_t _two31 = INT64_C(1) << 31; // 2^31
static const int64_t _two32 = INT64_C(1) << 32; // 2^32
typedef uint32_t Torus32;
const int N = 4;
const int M = N / 2;

const int Msize = 2;
const int BLOCK_SIZE = 1;
const int GRID_SIZE = 1;
const uint32_t inter = uint32_t(_two31 / Msize * 2);
float alpha = pow(2.0, -15.4);


int sign(float d)
{
    if (d > 0)
    {
        return 1;
    }
    else if (d < 0)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

uint32_t mutoT(int mu, int Msize)
{
    return uint32_t(_two31 / Msize * 2 * mu);
}

Torus32 dtot32(float d)
{
    int dsign = sign(d);
    return Torus32(round(fmod(d * dsign, 1) * _two32) * dsign);
}

void gaussian32(uint32_t *vecmu, float alpha, uint32_t *h_ga, int size = 1)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, alpha);

    for (size_t i = 0; i < size; i++)
    {
        h_ga[i] = dtot32(dis(gen)) + vecmu[i];
        // h_ga[i].x = dtot32(dis(gen)) + vecmu[i];
        // h_ga[i].y = 0;
    }
}

uint32_t Ttomu(uint32_t phase, uint32_t inter)
{
    uint32_t half = uint32_t(inter / 2);
    return uint32_t(uint32_t(phase + half) / inter);
}

void trlweKeyGen(int32_t *h_trlwekey)
{
    cout << "htrlwekey: " << endl;
    srand((int)time(NULL));
    for (int i = 0; i < N; i++)
    {
        h_trlwekey[i] = rand() % 2;
        cout << h_trlwekey[i] << endl;
    }
}

__global__ void product(cuDoubleComplex* d_Comp_a, cuDoubleComplex* d_Comp_trlwekey, cuDoubleComplex* d_Comp_product)
{
	for (size_t i = 0; i < M; i++)
    {
        /* code */
        d_Comp_product[i] = cuCmul(d_Comp_a[i], d_Comp_trlwekey[i]);
    }
}

void PolyMul(uint32_t *h_a, int32_t *h_trlwekey, uint32_t *h_product)
{
    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
    
    // process h_a
    cuDoubleComplex *h_Comp_a = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
    cuDoubleComplex *d_Comp_a;
    cudaMalloc((void **)&d_Comp_a, M * sizeof(cuDoubleComplex));

    for (int i = 0; i < M; i++)
    {
        h_Comp_a[i].x = h_a[i];
        h_Comp_a[i].y = h_a[i + M];
    }
    cudaMemcpy(d_Comp_a, h_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_a, (cuDoubleComplex *)d_Comp_a, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Comp_a, d_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cout  << "h_Comp_a: polynomial Point value representation" << endl;
    for (int i = 0; i < M; i++)
    {
        cout << i << ": " << h_Comp_a[i].x << ", " << h_Comp_a[i].y << endl;
    }
    cout << "------------------" << endl;
    
    // process h_trlwekey
    cuDoubleComplex *h_Comp_trlwekey = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
    cuDoubleComplex *d_Comp_trlwekey;
    cudaMalloc((void **)&d_Comp_trlwekey, M * sizeof(cuDoubleComplex));

    for (int i = 0; i < M; i++)
    {
        h_Comp_trlwekey[i].x = h_trlwekey[i];
        h_Comp_trlwekey[i].y = h_trlwekey[i + M];
    }
    cudaMemcpy(d_Comp_trlwekey, h_Comp_trlwekey, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_trlwekey, (cuDoubleComplex *)d_Comp_trlwekey, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Comp_trlwekey, d_Comp_trlwekey, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cout  << "h_Comp_trlwekey: polynomial Point value representation" << endl;
    for (int i = 0; i < M; i++)
    {
        cout << i << ": " << h_Comp_trlwekey[i].x << ", " << h_Comp_trlwekey[i].y << endl;
    }
    cout << "------------------" << endl;

    // process mul
    cuDoubleComplex *h_Comp_product = (cuDoubleComplex *)malloc(N * sizeof(cuDoubleComplex));
    cuDoubleComplex *d_Comp_product;
    cudaMalloc((void **)&d_Comp_product, M * sizeof(cuDoubleComplex));

    product<<<GRID_SIZE,BLOCK_SIZE>>>(d_Comp_a, d_Comp_trlwekey, d_Comp_product);
    cudaMemcpy(h_Comp_product, d_Comp_product, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cout  << "polynomial mul :Point value representation" << endl;
    for (int i = 0; i < M; i++)
    {
        cout << i << ": " << h_Comp_product[i].x << ", " << h_Comp_product[i].y << endl;
    }
    cout << "------------------" << endl;

    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_product, (cuDoubleComplex *)d_Comp_product, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Comp_product, d_Comp_product, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cout  << "h_product:" << endl;
    for (int i = 0; i < M; i++)
    {
        h_product[i] = h_Comp_product[i].x;
        h_product[i + M] = h_Comp_product[i].y;
    }
    
    for (int i = 0; i < M; i++)
    {
        cout << i << ": " << h_product[i] << endl;
    }

    cufftDestroy(plan);
    free(h_Comp_a);
    free(h_Comp_trlwekey);
    free(h_Comp_product);
    cudaFree(d_Comp_a);
    cudaFree(d_Comp_trlwekey);
    cudaFree(d_Comp_product);
}

void TwistFFT(uint32_t *a, cuDoubleComplex *CompData)
{
    for (int i = 0; i < N; i++)
    {
        CompData[i].x = a[i];
        CompData[i].y = 0;
    }
    
    cuDoubleComplex *d_CompData;
    cudaMalloc((void **)&d_CompData, N * sizeof(cuDoubleComplex));
    cudaMemcpy(d_CompData, CompData, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_CompData, (cuDoubleComplex *)d_CompData, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    cudaMemcpy(CompData, d_CompData, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_CompData);
}

void TwistIFFT(cuDoubleComplex *CompData)
{
    cuDoubleComplex *d_CompData;
    cudaMalloc((void **)&d_CompData, N * sizeof(cuDoubleComplex));
    cudaMemcpy(d_CompData, CompData, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_CompData, (cuDoubleComplex *)d_CompData, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    cudaMemcpy(CompData, d_CompData, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_CompData);
}


void trlweSymEnc(uint32_t *vecmu, float alpha, int32_t *h_trlwekey, uint32_t *cipher, uint32_t *h_a)
{
    // uint32_t *h_a = (uint32_t *)malloc(N * sizeof(uint32_t));

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    cout << "h_a:" << endl;
    for (int i = 0; i < N; i++)
    {
        h_a[i] = g();
        cout << h_a[i] << endl;
    }

    uint32_t *h_product = (uint32_t *)malloc(N * sizeof(uint32_t));
    PolyMul(h_a, h_trlwekey, h_product);
    
    uint32_t *h_ga = (uint32_t *)malloc(N * sizeof(uint32_t));
    gaussian32(vecmu, alpha, h_ga, N);
    cout << "h_ga:" << endl;
    for (int i = 0; i < N; i++)
    {
        cout << h_ga[i] << endl;
    }

    cout << "cipher:" << endl;
    for (int i = 0; i < N; i++)
    {
        // cipher[i] = round(h_ga[i] - h_product[i]) % (_two32);
        cipher[i] = (h_ga[i] - h_product[i]) % (_two32);
        cout << cipher[i] << endl;
    }

    free(h_product);
    free(h_ga);
}

void trlweSymDec(uint32_t *cipher, uint32_t *h_a, int32_t *h_trlwekey)
{
    uint32_t *h_product = (uint32_t *)malloc(N * sizeof(uint32_t));
    PolyMul(h_a, h_trlwekey, h_product);

    cout << "dec phase = mu + e" << endl;
    uint32_t *h_phase = (uint32_t *)malloc(N * sizeof(uint32_t));
    for (int i = 0; i < N; i++)
    {
        h_phase[i] = cipher[i] + h_product[i];
    }
    cout << "decryption result: " << endl;
    for (int i = 0; i < N; i++)
    {
        cout << i << ": " << Ttomu(h_phase[i], inter) << endl;
    }

}

void Test()
{
    // generate message
    cout << "message to Torus: " << endl;
    uint32_t *vecmu = (uint32_t *)malloc(N * sizeof(uint32_t));
    for (int i = 0; i < N; i++)
    {
        if(i % 2 == 0)
        {
            vecmu[i] = mutoT(0, Msize);
        }
        else
        {
            vecmu[i] = mutoT(1, Msize);
        }
        cout << vecmu[i] << endl;
    }

    cout << "---------------------------------" << endl;
    // trlwekey generation
    int32_t *h_trlwekey = (int32_t *)malloc(N * sizeof(int32_t));
    trlweKeyGen(h_trlwekey);

    // encryption
    uint32_t *cipher = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *a = (uint32_t *)malloc(N * sizeof(uint32_t));
    trlweSymEnc(vecmu, alpha, h_trlwekey, cipher, a);

    // decryption
    trlweSymDec(cipher, a, h_trlwekey);

    free(vecmu);
    free(h_trlwekey);
    free(cipher);
    free(a);
}


int main()
{
    Test();
    return 0;
}