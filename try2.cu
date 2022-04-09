#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <ctime>
#include <chrono>
#include <typeinfo>
#include "error.cuh"
#include "cufft.h"
#include <cufftXt.h>
using namespace std;

static const int64_t _two31 = INT64_C(1) << 31; // 2^31
static const int64_t _two32 = INT64_C(1) << 32; // 2^32
typedef uint32_t Torus32;

const int BLOCK_SIZE = 1;
const int GRID_SIZE = 1;

class Params128
{
public:
    int N;
    int n;
    int bk_l;
    int bk_Bgbit;
    float ks_stdev;
    float bk_stdev;
    int Bg;
    int Msize;
    double *H;
    uint32_t inter;

public:
    Params128(int N, int n, int bk_l, int bk_Bgbit, float ks_stdev, float bk_stdev, int Msize) : N(N),
                                                                                                 n(n),
                                                                                                 bk_l(bk_l),
                                                                                                 bk_Bgbit(bk_Bgbit),
                                                                                                 ks_stdev(ks_stdev),
                                                                                                 bk_stdev(bk_stdev),
                                                                                                 Bg(pow(2, bk_Bgbit)),
                                                                                                 Msize(Msize),
                                                                                                 inter(_two31 / Msize * 2)
    {
        this->H = (double *)malloc(sizeof(double) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->H[i] = pow(this->Bg, (-(i + 1)));
            // cout << this->H[i] << endl;
        }
    }
    ~Params128() {
        free(H);
    }
};

// Params128& getp128()
// {
//     // Params128 *p128 = new Params128(4, 630, 2, 10, pow(2.0, -15.4), pow(2.0, -28), 2);
//     return Params128(4, 630, 2, 10, pow(2.0, -15.4), pow(2.0, -28), 2);
// }

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

    for (int i = 0; i < size; i++)
    {
        h_ga[i] = dtot32(dis(gen)) + vecmu[i];
    }
}

uint32_t Ttomu(uint32_t phase, uint32_t inter)
{
    uint32_t half = uint32_t(inter / 2);
    return uint32_t(uint32_t(phase + half) / inter);
}

int32_t* trlweKeyGen(Params128 p128)
{
    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));
    srand((int)time(NULL));
    for (int i = 0; i < p128.N; i++)
    {
        trlwekey[i] = rand() % 2;
    }
    return trlwekey;
}

__global__ void product(cuDoubleComplex *d_Comp_a, cuDoubleComplex *d_Comp_trlwekey, cuDoubleComplex *d_Comp_product, Params128 p128)
{
    int M = p128.N / 2;
    for (int i = 0; i < M; i++)
    {
        d_Comp_product[i] = cuCmul(d_Comp_a[i], d_Comp_trlwekey[i]);
    }
}

uint32_t *PolyMul(uint32_t *h_a, int32_t *h_trlwekey, Params128 p128)
{
    int M = p128.N / 2;
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

    // process mul
    cuDoubleComplex *h_Comp_product = (cuDoubleComplex *)malloc(p128.N * sizeof(cuDoubleComplex));
    cuDoubleComplex *d_Comp_product;
    cudaMalloc((void **)&d_Comp_product, M * sizeof(cuDoubleComplex));

    product<<<GRID_SIZE, BLOCK_SIZE>>>(d_Comp_a, d_Comp_trlwekey, d_Comp_product, p128);

    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_product, (cuDoubleComplex *)d_Comp_product, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Comp_product, d_Comp_product, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    uint32_t *h_product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < M; i++)
    {
        h_product[i] = h_Comp_product[i].x;
        h_product[i + M] = h_Comp_product[i].y;
    }

    cufftDestroy(plan);
    free(h_Comp_a);
    free(h_Comp_trlwekey);
    free(h_Comp_product);
    cudaFree(d_Comp_a);
    cudaFree(d_Comp_trlwekey);
    cudaFree(d_Comp_product);

    return h_product;
}

uint32_t **trlweSymEnc(uint32_t *vecmu, int32_t *trlwekey, Params128 p128)
{
    uint32_t *b = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    uint32_t *a = (uint32_t *)malloc(p128.N * sizeof(uint32_t));

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    for (int i = 0; i < p128.N; i++)
    {
        a[i] = g();
    }

    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    product = PolyMul(a, trlwekey, p128);

    uint32_t *ga = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    gaussian32(vecmu, p128.ks_stdev, ga, p128.N);

    for (int i = 0; i < p128.N; i++)
    {
        b[i] = (ga[i] - product[i]) % (_two32);
    }

    uint32_t **c = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    for (int i = 0; i < p128.N; i++)
    {
        c[0][i] = b[i];
    }
    for (int i = 0; i < p128.N; i++)
    {
        c[1][i] = a[i];
    }

    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < p128.N; j++)
    //     {
    //         cout << c[i][j] << " ";
    //     }
    //     cout <<  endl;
    // }

    free(b);
    free(a);
    free(product);
    free(ga);

    return c;
}

uint32_t *trlweSymDec(uint32_t **c, int32_t *trlwekey, Params128 p128)
{
    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    product = PolyMul(c[1], trlwekey, p128);

    uint32_t *phase = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        phase[i] = c[0][i] + product[i];
    }

    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        mu[i] = Ttomu(phase[i], p128.inter);
    }

    free(product);
    free(phase);

    return mu;
}

uint32_t*** trgswSymEnc(uint32_t *vecmu, int32_t *trlwekey, Params128 p128)
{
    // uint32_t **muh = (uint32_t **)malloc(p128.bk_l * sizeof(uint32_t *));
    // for (int i = 0; i < p128.bk_l; i++)
    // {
    //     muh[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    // }
    // cout << p128.Bg << endl;
    // cout << p128.bk_Bgbit << endl;
    // for (int j = 0; j < p128.bk_l; j++)
    // {
    //     cout << p128.H[j] << "  ";
    // }
    // external product : H * plain
    // for (int i = 0; i < p128.bk_l; i++)
    // {
    //     cout << p128.H[i] << "----" << endl;
        // for (int j = 0; j < p128.N; j++)
        // {
        //     cout << vecmu[j] << "----";
        //     muh[i][j] = p128.H[i] * vecmu[j];
        //     cout << muh[i][j] << "~~~~";
            
        // }
        // cout << endl;
    // }

    int lines = 2 * p128.bk_l;
    uint32_t ***c;
    c = (uint32_t ***)malloc(lines * sizeof(uint32_t **));

    for (int i = 0; i < lines; i++)
    {
        c[i] = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            c[i][j] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
        }
    }

    // uint32_t *vec_zero = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    // for (int i = 0; i < p128.N; i++)
    // {
    //     vec_zero[i] = 0;
    // }

    // for (int i = 0; i < lines; i++)
    // {
    //     c[i] = trlweSymEnc(vec_zero, trlwekey, p128);
    // }
    // for (int i = 0; i < p128.bk_l; i++)
    // {
    //     for (int j = 0; j < p128.N; j++)
    //     {
    //         c[i][0][j] += muh[i][j];
    //     }
    // }
    // for (int i = p128.bk_l; i < lines; i++)
    // {
    //     for (int j = 0; j < p128.N; j++)
    //     {
    //         c[i][1][j] += muh[i][j];
    //     }
    // }

    // free operations
    // for (int i = 0; i < p128.bk_l; i++)
    // {
    //     free(muh[i]);
    // }
    // free(muh);

    return c;
}

uint32_t* trgswSymDec(uint32_t ***c, int32_t *trlwekey, Params128 p128)
{
    uint32_t *phase = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    product = PolyMul(c[0][1], trlwekey, p128);
    for (int i = 0; i < p128.N; i++)
    {
        phase[i] = (c[0][0][i] + product[i]) * p128.Bg;
        vecmu[i] = Ttomu(phase[i], p128.inter);
        cout << vecmu[i] << endl;
    }
    free(phase);
    free(product);
    return vecmu;
}

uint32_t*** fun(uint32_t *vecmu, int32_t *trlwekey, Params128 p128)
{
    uint32_t **muh = (uint32_t **)malloc(p128.bk_l * sizeof(uint32_t *));
    for (int i = 0; i < p128.bk_l; i++)
    {
        muh[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }

    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            muh[i][j] = p128.H[i] * vecmu[j];
            cout << muh[i][j] << "~~~~";
        }
        cout << endl;
    }

    int lines = 2 * p128.bk_l;
    uint32_t ***c;
    c = (uint32_t ***)malloc(lines * sizeof(uint32_t **));

    for (int i = 0; i < lines; i++)
    {
        c[i] = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            c[i][j] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
        }
    }

    uint32_t *vec_zero = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    for (int i = 0; i < p128.N; i++)
    {
        vec_zero[i] = mutoT(1, p128.Msize);
        cout << vec_zero[i] << endl;
    }
    uint32_t **cipher = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        cipher[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    cipher = trlweSymEnc(vec_zero, trlwekey, p128);
    uint32_t *mu = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(cipher, trlwekey, p128);
    for (int i = 0; i < p128.N; i++)
    {
        cout << mu[i] << endl;
    }
    
    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < p128.N; j++)
    //     {
    //         cout << c[i][j] << " ";
            
    //     }
    //     cout << "------------------" << endl;
    // }

    // for (int i = 0; i < lines; i++)
    // {
    //     c[i] = trlweSymEnc(vec_zero, trlwekey, p128);
    // }
    // for (int i = 0; i < lines; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         for (int k = 0; k < p128.N; k++)
    //         {
    //             cout << c[i][j][k] << " ";
    //         }   
    //         cout << endl;
    //     }
    //     cout << "------------------" << endl;
    // }


    for (int i = 0; i < p128.bk_l; i++)
    {
        free(muh[i]);
    }
    free(muh);
    free(mu);

    return c;
}

void Test()
{
    Params128 p128 = Params128(4, 630, 2, 10, pow(2.0, -15.4), pow(2.0, -28), 2);

    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N / 2; i++)
    {
        if (i % 2 == 0)
        {
            vecmu[i] = mutoT(1, p128.Msize);
        }
        else
        {
            vecmu[i] = mutoT(1, p128.Msize);
        }
    }
    for (int i = p128.N / 2; i < p128.N; i++)
    {
        vecmu[i] = mutoT(0, p128.Msize);
    }

    // trlwekey generation
    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));
    trlwekey = trlweKeyGen(p128);

    int lines = 2 * p128.bk_l;
    uint32_t ***c;
    c = (uint32_t ***)malloc(lines * sizeof(uint32_t **));

    for (int i = 0; i < lines; i++)
    {
        c[i] = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            c[i][j] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
        }
    }
    c = fun(vecmu, trlwekey, p128);

    free(vecmu);
    free(trlwekey);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free((c[i][j]));
        }
    }
    for (int i = 0; i < p128.N; i++)
    {
        free(c[i]);
    }
}

void Test_TRLWE()
{
    Params128 p128 = Params128(4, 630, 2, 10, pow(2.0, -15.4), pow(2.0, -28), 2);

    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 0)
        {
            vecmu[i] = mutoT(1, p128.Msize);
        }
        else
        {
            vecmu[i] = mutoT(0, p128.Msize);
        }
    }
    for (int i = 0; i < p128.N; i++)
    {
        cout << vecmu[i] << endl;
    }
    cout << "-----------------" << endl;
    // trlwekey generation
    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));
    trlwekey = trlweKeyGen(p128);

    uint32_t **cipher = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        cipher[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    cipher = trlweSymEnc(vecmu, trlwekey, p128);
    uint32_t *mu = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    mu = trlweSymDec(cipher, trlwekey, p128);
    for (int i = 0; i < p128.N; i++)
    {
        cout << mu[i] << endl;
    }

    free(vecmu);
    free(trlwekey);
    for (int i = 0; i < p128.N; i++)
    {
        free(cipher[i]);
    }
    free(cipher);
    free(mu);
}

int main()
{
    Test_TRLWE();
    return 0;
}