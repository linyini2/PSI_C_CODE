#include <iostream>
#include <math.h>
#include <cuComplex.h>
#include <complex>
#include <string>
#include <random>
#include <ctime>
#include "error.cuh"
#include <chrono>
#include "cufft.h"
#include <cufftXt.h>
#include <iomanip>
#include "cudaArray_4dim.cuh"
using namespace std;

typedef std::complex<double> Complex;
#define PI acos(-1)
static const int64_t _two31 = INT64_C(1) << 31; // 2^31
static const int64_t _two32 = INT64_C(1) << 32; // 2^32
typedef uint32_t Torus32;
#define EPSILON 1e-15




__host__ cuDoubleComplex cexp(const cuDoubleComplex &z)
{
    Complex stl_complex(cuCreal(z), cuCimag(z));
    stl_complex = exp(stl_complex);
    return make_cuDoubleComplex(real(stl_complex), imag(stl_complex));
}



class Params128
{
public:
    int N;
    int n;
    int bk_l;
    int bk_Bgbit;
    int bk_Bgbitbar;
    double ks_stdev;
    double bk_stdev;
    int Bg;
    int Bgbar;
    int Msize;
    double *H;
    uint32_t offset;
    int *decbit;
    uint32_t inter;
    cuDoubleComplex *twist;

public:
    Params128(int N, int n, int bk_l, int bk_Bgbit, int bk_Bgbitbar, double ks_stdev, double bk_stdev, int Msize) : N(N),
                                                                                                 n(n),
                                                                                                 bk_l(bk_l),
                                                                                                 bk_Bgbit(bk_Bgbit),
                                                                                                 bk_Bgbitbar(bk_Bgbitbar),
                                                                                                 ks_stdev(ks_stdev),
                                                                                                 bk_stdev(bk_stdev),
                                                                                                 Bg(pow(2, bk_Bgbit)),
                                                                                                 Bgbar(pow(2, bk_Bgbitbar)),
                                                                                                 Msize(Msize),
                                                                                                 inter(_two31 / Msize * 2)
    {
        this->H = (double *)malloc(sizeof(double) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->H[i] = pow(this->Bg, (-(i + 1)));
        }

        this->offset = 0;
        for (int i = 0; i < this->bk_l; i++)
        {
            this->offset += _two32 * this->H[i];
        }
        this->offset = this->Bg / 2 * this->offset;

        this->decbit = (int *)malloc(sizeof(int) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->decbit[i] = 32 - (i + 1) * this->bk_Bgbit;
        }
        
        

        this->twist = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * this->N / 2);
        for (int k = 0; k < this->N / 2; k++)
        {
            // attention!
            // twist[k] = cexp(make_cuDoubleComplex(0, 2 * double(k) * PI / this->N));
            twist[k] = cexp(make_cuDoubleComplex(0, double(k) * PI / this->N));
        }
    }
    Params128(const Params128 &p128)
    {
        this->N = p128.N;
        this->n = p128.n;
        this->bk_l = p128.bk_l;
        this->bk_Bgbit = p128.bk_Bgbit;
        this->bk_Bgbitbar = p128.bk_Bgbitbar;
        this->ks_stdev = p128.ks_stdev;
        this->bk_stdev = p128.bk_stdev;
        this->Bg = p128.Bg;
        this->Msize = p128.Msize;
        this->inter = p128.inter;

        this->H = (double *)malloc(sizeof(double) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->H[i] = pow(this->Bg, (-(i + 1)));
        }

        this->offset = 0;
        for (int i = 0; i < this->bk_l; i++)
        {
            this->offset += _two32 * this->H[i];
        }
        this->offset = this->Bg / 2 * this->offset;

        this->decbit = (int *)malloc(sizeof(int) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->decbit[i] = 32 - (i + 1) * this->bk_Bgbit;
        }

        this->twist = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * this->N / 2);
        for (int k = 0; k < this->N / 2; k++)
        {
            twist[k] = cexp(make_cuDoubleComplex(0, 2 * double(k) * PI / this->N));
        }
    }
    Params128 &operator=(const Params128 &p128)
    {
        if (this == &p128)
        {
            return *this;
        }
        this->N = p128.N;
        this->n = p128.n;
        this->bk_l = p128.bk_l;
        this->bk_Bgbit = p128.bk_Bgbit;
        this->bk_Bgbitbar = p128.bk_Bgbitbar;
        this->ks_stdev = p128.ks_stdev;
        this->bk_stdev = p128.bk_stdev;
        this->Bg = p128.Bg;
        this->Msize = p128.Msize;
        this->inter = p128.inter;

        free(this->H);
        this->H = (double *)malloc(sizeof(double) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->H[i] = pow(this->Bg, (-(i + 1)));
        }

        this->offset = 0;
        for (int i = 0; i < this->bk_l; i++)
        {
            this->offset += _two32 * this->H[i];
        }
        this->offset = this->Bg / 2 * this->offset;

        free(this->decbit);
        this->decbit = (int *)malloc(sizeof(int) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->decbit[i] = 32 - (i + 1) * this->bk_Bgbit;
        }

        free(this->twist);
        this->twist = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * this->N / 2);
        for (int k = 0; k < this->N / 2; k++)
        {
            twist[k] = cexp(make_cuDoubleComplex(0, 2 * double(k) * PI / this->N));
        }

        return *this;
    }
    ~Params128()
    {
        if (this->H)
        {
            free(this->H);
            this->H = nullptr;
        }
        if (this->decbit)
        {
            free(this->decbit);
            this->decbit = nullptr;
        }
        if (this->twist)
        {
            free(this->twist);
            this->twist = nullptr;
        }
    }
};

double sign(double d)
{
    if (d > EPSILON)
    {
        return 1;
    }
    else if (d < -EPSILON)
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

Torus32 dtot32(double d)
{
    double dsign = sign(d);
    return Torus32(round(fmod(d * dsign, 1) * _two32) * dsign);
}

uint32_t Ttomu(uint32_t phase, uint32_t inter)
{
    uint32_t half = uint32_t(inter / 2);
    return uint32_t(uint32_t(phase + half) / inter);
}

void gaussian32(uint32_t *vecmu, uint32_t *ga, double alpha, int size = 1)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, alpha);

    for (size_t i = 0; i < size; i++)
    {
        ga[i] = dtot32(dis(gen)) + vecmu[i];
    }
}

void trlweKeyGen(int32_t *trlwekey, Params128 &p128)
{
    srand((int)time(NULL));
    for (int i = 0; i < p128.N; i++)
    {
        trlwekey[i] = rand() % 2;
    }
}

void TwistFFT(int32_t *a, Params128 &p128, cuDoubleComplex *h_Comp_a)
{
    int M = p128.N / 2;
    cuDoubleComplex *d_Comp_a;
    CHECK(cudaMalloc((void **)&d_Comp_a, M * sizeof(cuDoubleComplex)));
    for (int i = 0; i < M; i++)
    {
        h_Comp_a[i].x = a[i];
        h_Comp_a[i].y = a[i + M];
        h_Comp_a[i] = cuCmul(h_Comp_a[i], p128.twist[i]);
    }
    CHECK(cudaMemcpy(d_Comp_a, h_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_a, (cuDoubleComplex *)d_Comp_a, CUFFT_FORWARD);
    // CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_Comp_a, d_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    CHECK(cudaFree(d_Comp_a));
}



void TwistIFFT(cuDoubleComplex *h_Comp_a, Params128 &p128, double *product)
{
    int M = p128.N / 2;
    cuDoubleComplex *d_Comp_a;
    CHECK(cudaMalloc((void **)&d_Comp_a, M * sizeof(cuDoubleComplex)));
    CHECK(cudaMemcpy(d_Comp_a, h_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_a, (cuDoubleComplex *)d_Comp_a, CUFFT_INVERSE);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_Comp_a, d_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; i++)
    {
        cuDoubleComplex twist;
        twist.x = p128.twist[i].x;
        twist.y = (-1) * p128.twist[i].y;
        // normalize
        h_Comp_a[i].x = h_Comp_a[i].x / M;
        h_Comp_a[i].y = h_Comp_a[i].y / M;
        h_Comp_a[i] = cuCmul(h_Comp_a[i], twist);
        product[i] = h_Comp_a[i].x;
        product[i + M] = h_Comp_a[i].y;
    }

    cufftDestroy(plan);
    CHECK(cudaFree(d_Comp_a));
}


void PolyMul(uint32_t *a, int32_t *trlwekey, uint32_t *product, Params128 &p128)
{
    int M = p128.N / 2;
    cuDoubleComplex *h_Comp_a = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    cuDoubleComplex *h_Comp_trlwekey = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    cuDoubleComplex *h_Comp_product = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    double *result = (double *)malloc(sizeof(double) * p128.N);

    TwistFFT((int32_t *)a, p128, h_Comp_a);
    TwistFFT(trlwekey, p128, h_Comp_trlwekey);

    for (int i = 0; i < M; i++)
    {
        h_Comp_product[i] = cuCmul(h_Comp_a[i], h_Comp_trlwekey[i]);
    }
    TwistIFFT(h_Comp_product, p128, result);
    for (int i = 0; i < p128.N; i++)
    {
        // attention!
        // (uint64_t)(round(result[i])) : double --> int , because module only support integers.
        product[i] = (uint32_t)((uint64_t)(round(result[i])) % _two32);
    }
    

    free(h_Comp_a);
    free(h_Comp_trlwekey);
    free(h_Comp_product);
    free(result);
}

void trlweSymEnc(uint32_t *vecmu, int32_t *trlwekey, double stdev, Params128 &p128, uint32_t **c)
{
    uint32_t *a = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    for (int i = 0; i < p128.N; i++)
    {
        a[i] = g();
        c[1][i] = a[i];
    }

    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    PolyMul(a, trlwekey, product, p128);

    uint32_t *ga = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    gaussian32(vecmu, ga, stdev, p128.N);

    for (int i = 0; i < p128.N; i++)
    {
        // attention!
        // python code:  np.array([ga - mul, a], dtype=np.uint32)
        // c[0][i] = (ga[i] - product[i]) % (_two32);
        c[0][i] = (uint32_t)(ga[i] - product[i]);
    }

    free(a);
    free(product);
    free(ga);
}


void trlweSymDec(uint32_t **c, int32_t *trlwekey, Params128 &p128, uint32_t *mu)
{
    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    PolyMul(c[1], trlwekey, product, p128);

    uint32_t *phase = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        phase[i] = c[0][i] + product[i];
    }
    for (int i = 0; i < p128.N; i++)
    {
        mu[i] = Ttomu(phase[i], p128.inter);
    }

    free(product);
    free(phase);
}

void Test_TRLWE()
{
    Params128 p128 = Params128(1024, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);
    // generate message
    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 1)
        {
            vecmu[i] = mutoT(1, p128.Msize);
        }
        else
        {
            vecmu[i] = mutoT(0, p128.Msize);
        }
    }
    // vecmu[p128.N - 1] = mutoT(1, p128.Msize);

    // trlwekey generation
    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));
    trlweKeyGen(trlwekey, p128);

    // encryption
    uint32_t **c = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(vecmu, trlwekey, p128.ks_stdev, p128, c);
    

    // decryption
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trlweSymDec(c, trlwekey, p128, mu);
    cout << "\ntrlwe decryption result: " << endl;
    for (int i = 0; i < p128.N; i++)
    {
        cout << mu[i] << " ";
        if(i % 2 == 1 && mu[i] != 1)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
        if(i % 2 == 0 && mu[i] != 0)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
    }

    free(vecmu);
    free(trlwekey);
    for (int i = 0; i < 2; i++)
    {
        free(c[i]);
    }
    free(c);
    free(mu);
}



void trgswSymEnc(uint32_t *vecmu, int32_t *trlwekey, Params128 &p128, uint32_t ***c)
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
        }
    }

    int lines = 2 * p128.bk_l;
    

    uint32_t *vec_zero = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    for (int i = 0; i < p128.N; i++)
    {
        vec_zero[i] = 0;
    }

    for (int i = 0; i < lines; i++)
    {
        trlweSymEnc(vec_zero, trlwekey, p128.bk_stdev, p128, c[i]); 
    }

    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            c[i][0][j] += muh[i][j];
        }
    }
    for (int i = p128.bk_l; i < lines; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            c[i][1][j] += muh[i - p128.bk_l][j];
        }
    }

    for (int i = 0; i < p128.bk_l; i++)
    {
        free(muh[i]);
    }
    free(muh);
    free(vec_zero);
}

void trgswSymEnc_2(uint64_t *vecmu, int32_t *trlwekey, Params128 &p128, uint32_t*** c)
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
        }
    }

    int lines = 2 * p128.bk_l;
    

    uint32_t *vec_zero = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    for (int i = 0; i < p128.N; i++)
    {
        vec_zero[i] = 0;
    }

    for (int i = 0; i < lines; i++)
    {
        trlweSymEnc(vec_zero, trlwekey, p128.bk_stdev, p128, c[i]); 
    }

    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            c[i][0][j] += muh[i][j];
        }
    }
    for (int i = p128.bk_l; i < lines; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            c[i][1][j] += muh[i - p128.bk_l][j];
        }
    }

    for (int i = 0; i < p128.bk_l; i++)
    {
        free(muh[i]);
    }
    free(muh);
    free(vec_zero);
}

void trgswSymDec(uint32_t ***c, int32_t *trlwekey, Params128 &p128, uint32_t* mu)
{
    uint32_t *phase = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    PolyMul(c[0][1], trlwekey, product, p128);
    for (int i = 0; i < p128.N; i++)
    {
        phase[i] = (c[0][0][i] + product[i]) * p128.Bg;
        mu[i] = Ttomu(phase[i], p128.inter);
    }
    free(phase);
    free(product);
}

void Test_TRGSW()
{   
    Params128 p128 = Params128(1024, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);

    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 1)
        {
            vecmu[i] = mutoT(1, p128.Msize);
        }
        else
        {
            vecmu[i] = mutoT(0, p128.Msize);
        }
    }
    // for (int i = p128.N / 2; i < p128.N; i++)
    // {
    //     vecmu[i] = mutoT(1, p128.Msize);
    // }
    // vecmu[p128.N-1] = 0;
    

    // trlwekey generation
    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));
    trlweKeyGen(trlwekey, p128);

    
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
    trgswSymEnc(vecmu, trlwekey, p128, c);


    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trgswSymDec(c, trlwekey, p128, mu);
    cout << "\ntrgsw decryption result:" << endl;
    for (int i = 0; i < p128.N; i++)
    {
        cout << mu[i] << " ";
        if(i % 2 == 1 && mu[i] != 1)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
        if(i % 2 == 0 && mu[i] != 0)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
    }
    

    free(vecmu);
    free(trlwekey);
    free(mu);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(c[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(c[i]);
    }
    free(c);
}


void trgswfftSymEnc(uint64_t *mu, int32_t *trlwekey, Params128 &p128, cuDoubleComplex ***trgswfftcipher)
{
    int lines = 2 * p128.bk_l;
    uint32_t ***trgsw;
    trgsw = (uint32_t ***)malloc(lines * sizeof(uint32_t **));

    for (int i = 0; i < lines; i++)
    {
        trgsw[i] = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgsw[i][j] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
        }
    }
    trgswSymEnc_2(mu, trlwekey, p128, trgsw);

    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            // attention!
            TwistFFT((int32_t *)trgsw[i][j], p128, trgswfftcipher[i][j]);
        }   
    }

    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(trgsw[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(trgsw[i]);
    }
    free(trgsw);
}

__global__ void externalproduct(cudaPitchedPtr devPitchedPtr, cudaPitchedPtr devPitchedPtr_2, cudaPitchedPtr devPitchedPtr_3, cudaExtent extent)
{
    cuDoubleComplex *devPtr = (cuDoubleComplex *)devPitchedPtr.ptr;
    cuDoubleComplex *sliceHead, *rowHead;
    cuDoubleComplex *devPtr_2 = (cuDoubleComplex *)devPitchedPtr_2.ptr;
    cuDoubleComplex *sliceHead_2, *rowHead_2;
    cuDoubleComplex *devPtr_3 = (cuDoubleComplex *)devPitchedPtr_3.ptr;
    cuDoubleComplex *sliceHead_3, *rowHead_3;

    for (int z = 0; z < extent.depth; z++)
    {
        sliceHead = (cuDoubleComplex *)((char *)devPtr + z * devPitchedPtr.pitch * extent.height);
        sliceHead_2 = (cuDoubleComplex *)((char *)devPtr_2 + z * devPitchedPtr_2.pitch * extent.height);
        sliceHead_3 = (cuDoubleComplex *)((char *)devPtr_3 + z * devPitchedPtr_3.pitch * extent.height);
        for (int y = 0; y < extent.height; y++)
        {
            rowHead = (cuDoubleComplex*)((char *)sliceHead + y * devPitchedPtr.pitch);
            rowHead_2 = (cuDoubleComplex*)((char *)sliceHead_2 + y * devPitchedPtr_2.pitch);
            rowHead_3 = (cuDoubleComplex*)((char *)sliceHead_3 + y * devPitchedPtr_3.pitch);
            for (int x = 0; x < extent.width / sizeof(cuDoubleComplex); x++)
            {
                rowHead_3[x] = cuCmul(rowHead[x], rowHead_2[x]);
            }
        }
    }

}

void trgswfftExternalProduct(cuDoubleComplex ***trgswfftcipher, uint32_t **trlwecipher, Params128 p128, uint32_t **res)
{
    int lines = 2 * p128.bk_l;
    int M = p128.N / 2;

    // t = np.uint32((trlwecipher + p128.offset)%2**32)
    // t size: 2 * p128.N      ok!
    uint32_t **t = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        t[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            // trlwecipher + offset may surpass uint32_t ??
            t[i][j] = (uint32_t)((trlwecipher[i][j] + (uint64_t)p128.offset) % _two32);
        }
    }

    // t1 = np.array([t >> i for i in p128.decbit]) 
    // t1 size: p128.bk_l * 2 * p128.N        ok!
    uint32_t ***t1;
    t1 = (uint32_t ***)malloc(p128.bk_l * sizeof(uint32_t **));
    for (int i = 0; i < p128.bk_l; i++)
    {
        t1[i] = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    }
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t1[i][j] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
        }
    }


    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t[j][k] >> p128.decbit[i];
            }
        }
    }

    // t2=t1&(p128.Bg - 1)
    // t2 size: p128.bk_l * 2 * p128.N        ok!
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t1[i][j][k] & (p128.Bg - 1);
            }
        }
    }

    // t3=t2-p128.Bg // 2
    // t3 size: p128.bk_l * 2 * p128.N        ok!
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t1[i][j][k] - p128.Bg / 2;
            }   
        }
    }


    // decvec = np.concatenate([t3[:, 0], t3[:, 1]])
    // decvec size: (p128.bk_l * 2) * p128.N        ok!
    uint32_t **decvec = (uint32_t **)malloc(p128.bk_l * 2 * sizeof(uint32_t *));
    for (int i = 0; i < p128.bk_l * 2; i++)
    {
        decvec[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < p128.bk_l; i++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                decvec[i+j*2][k] = t1[i][j][k];
            }
        }   
    }


    // decvecfft = TwistFFT(np.int32(decvec), p128.twist, dim=2)
    // decvecfft size: (p128.bk_l * 2) * (p128.N / 2)        ok!
    cuDoubleComplex **decvecfft = (cuDoubleComplex **)malloc(sizeof(cuDoubleComplex)*2*p128.bk_l);
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        decvecfft[i] = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * p128.N / 2);
    }
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        TwistFFT((int32_t *)decvec[i], p128, decvecfft[i]);
    }
    
    
    
    // t4 = decvecfft.reshape(2 * p128.bk_l, 1, p128.N // 2) * trgswfftcipher
    // t4 size: (2 * p128.bk_l) * 2 * (p128.N / 2)     ok!
    cuDoubleComplex ***t4;
    t4 = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));

    for (int i = 0; i < lines; i++)
    {
        t4[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t4[i][j] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
        }
    }

    size_t width = p128.N / 2;
    size_t height = 2;
    size_t depth = 2 * p128.bk_l;
    cuDoubleComplex *h_trgswfftcipher, *h_decvecfft, *h_result;
    cudaPitchedPtr d_trgswfftcipher, d_decvecfft, d_result;
    cudaExtent extent;
    cudaMemcpy3DParms cpyParm;


    h_trgswfftcipher = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * width * height * depth);
    h_result = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * width * height * depth);
    for (int i = 0; i < depth; i++)
    {
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                h_trgswfftcipher[i*height*width+j*width+k] = trgswfftcipher[i][j][k];
            }
        }   
    }
    // process trgswfftcipher
    // alloc device memory
    extent = make_cudaExtent(sizeof(cuDoubleComplex) * width, height, depth);
    cudaMalloc3D(&d_trgswfftcipher, extent);

    cpyParm = {0};
    cpyParm.srcPtr = make_cudaPitchedPtr((void*)h_trgswfftcipher, sizeof(cuDoubleComplex) * width, width, height);
    cpyParm.dstPtr = d_trgswfftcipher;
    cpyParm.extent = extent;
    cpyParm.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cpyParm);


    // process decvecfft
    h_decvecfft = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * width * height * depth);
    for (int i = 0; i < depth; i++)
    {
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                h_decvecfft[i*height*width+j*width+k] = decvecfft[i][k];
            }
        }   
    }
    
    cudaMalloc3D(&d_decvecfft, extent);
    cpyParm = {0};
    cpyParm.srcPtr = make_cudaPitchedPtr((void*)h_decvecfft, sizeof(cuDoubleComplex) * width, width, height);
    cpyParm.dstPtr = d_decvecfft;
    cpyParm.extent = extent;
    cpyParm.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cpyParm);

    cudaMalloc3D(&d_result, extent);

    // call kernel 
    externalproduct<<<1,1>>>(d_trgswfftcipher, d_decvecfft, d_result, extent);
    cudaDeviceSynchronize();
    cpyParm = { 0 };
    cpyParm.srcPtr = d_result;
    cpyParm.dstPtr = make_cudaPitchedPtr((void*)h_result, sizeof(cuDoubleComplex) * width, width, height);
    cpyParm.extent = extent;
    cpyParm.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&cpyParm);

    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                t4[i][j][k] = h_result[i*width*height+j*width+k];
            }
        }
    }



    
    // t5 = t4.sum(axis=0)
    // t5 size: 2 * (p128.N / 2)    ok!
    cuDoubleComplex **t5 = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    for (int i = 0; i < 2; i++)
    {
        t5[i] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            t5[i][j].x = 0;
            t5[i][j].y = 0;
            for (int k = 0; k < lines; k++)
            {
                t5[i][j] = cuCadd(t5[i][j], t4[k][i][j]);
            }
        }
    }

    // t6 = TwistIFFT(t5, p128.twist, axis=1)
    // t6 size : 2 * p128.N
    double **t6 = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        t6[i] = (double *)malloc(p128.N * sizeof(double));
    }

    for (int i = 0; i < 2; i++)
    {
        TwistIFFT(t5[i], p128, t6[i]);
    }

    // res=np.array(t6, dtype=np.uint32)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            res[i][j] = (uint32_t)t6[i][j];
        }
    }
    
    

    for (int i = 0; i < 2; i++)
    {
        free(t[i]);
    }
    free(t);
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t1[i][j]);
        }
    }
    for (int i = 0; i < p128.bk_l; i++)
    {
        free(t1[i]);
    }
    free(t1);
    for (int i = 0; i < p128.bk_l * 2; i++)
    {
        free(decvec[i]);
    }
    free(decvec);
    for (int i = 0; i < p128.bk_l * 2; i++)
    {
        free(decvecfft[i]);
    }
    free(decvecfft);
    for (int i = 0; i < 2 * p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t4[i][j]);
        }
    }
    for (int i = 0; i < 2 * p128.bk_l; i++)
    {
        free(t4[i]);
    }
    free(t4);
    free(h_trgswfftcipher);
    free(h_decvecfft);
    free(h_result);
    CHECK(cudaFree(d_trgswfftcipher.ptr));
    CHECK(cudaFree(d_decvecfft.ptr));
    CHECK(cudaFree(d_result.ptr));
    for (int i = 0; i < 2; i++)
    {
        free(t5[i]);
    }
    free(t5);
    for (int i = 0; i < 2; i++)
    {
        free(t6[i]);
    }
    free(t6);
}

void Test_ExternalProduct()
{
    Params128 p128 = Params128(16, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);

    // generate message
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 0)
        {
            mu[i] = mutoT(0, p128.Msize);
        }
        else
        {
            mu[i] = mutoT(1, p128.Msize);
        }
    }
    // mu[0] = 0;
    // mu[2] = 0;
    // mu[p128.N-3]=mutoT(0, p128.Msize);
    uint64_t *pgsw = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        pgsw[i] = 0;
    }
    pgsw[0] = pow(2, 32);

    // trlwekey generation
    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));
    trlweKeyGen(trlwekey, p128);

    // encryption
    int lines = 2 * p128.bk_l;
    cuDoubleComplex ***trgswfftcipher;
    trgswfftcipher = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));

    for (int i = 0; i < lines; i++)
    {
        trgswfftcipher[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgswfftcipher[i][j] = (cuDoubleComplex *)malloc(p128.N / 2 * sizeof(cuDoubleComplex));
        }
    }
    trgswfftSymEnc(pgsw, trlwekey, p128, trgswfftcipher);


    uint32_t **trlwecipher = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu, trlwekey, p128.ks_stdev, p128, trlwecipher);
    

    uint32_t **cprod = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        cprod[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trgswfftExternalProduct(trgswfftcipher, trlwecipher, p128, cprod);
    
    uint32_t *msg = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(cprod, trlwekey, p128, msg);

    cout << "\n\nexternal product decryption result:\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
        if(i % 2 == 1 && msg[i] != 1)
        {
            cout << "\ni: " << i << " get wrong!" << endl;
        }
        if(i % 2 == 0 && msg[i] != 0)
        {
            cout << "\ni: " << i << " get wrong!" << endl;
        }
    }
    


    free(mu);
    free(pgsw);
    free(trlwekey);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(trgswfftcipher[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(trgswfftcipher[i]);
    }
    free(trgswfftcipher);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(cprod[i]);
    }
    free(cprod);
    free(msg);
}

__global__ void Reduce(cudaPitchedPtr db_ptr, cudaExtent db_extent, cudaPitchedPtr id_ptr, cudaExtent id_extent, cudaPitchedPtr t1_ptr, cudaExtent t1_extent, Params128 p128, int cnt)
{
	uint32_t *dbPtr = (uint32_t *)db_ptr.ptr;
    uint32_t *t1Ptr = (uint32_t *)t1_ptr.ptr;
    uint32_t *db_sliceHead0, *db_sliceHead1, *t1_sliceHead, *db_rowHead0, *db_rowHead1, *t1_rowHead;

    // i = 0, deal with: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    // i = 1, deal with: (1,3), (5,7), (9,11), (13,15)
    // i = 2, deal with: (3,7), (11,15)
    // i = 3, deal with: (7,15)
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int z = int(pow(2, cnt) - 1)+ (int)pow(2, cnt+1) * tid;

    db_sliceHead0 = (uint32_t *)((char *)dbPtr + z * db_ptr.pitch * db_extent.height);
    db_sliceHead1 = (uint32_t *)((char *)dbPtr + (z + (int)pow(2, cnt)) * db_ptr.pitch * db_extent.height);
    // t1_sliceHead = (uint32_t *)((char *)t1Ptr + tid * t1_ptr.pitch * t1_extent.height);
    for (int y = 0; y < db_extent.height; y++)
    {
        db_rowHead0 = (uint32_t*)((char *)db_sliceHead0 + y * db_ptr.pitch);
        db_rowHead1 = (uint32_t*)((char *)db_sliceHead1 + y * db_ptr.pitch);
        // t1_rowHead = (uint32_t*)((char *)t1_sliceHead + y * t1_ptr.pitch);
        for (int x = 0; x < db_extent.width / sizeof(uint32_t); x++)
        {
            db_rowHead1[x] = db_rowHead1[x] - db_rowHead0[x];    // ok!
            db_rowHead1[x] = (uint32_t)((db_rowHead1[x] + (uint64_t)p128.offset) % _two32);    // ok!
        }
    }
    

    // for (int d = 0; d < t1_extent.depth; d++)
    // {
    //     t1_sliceHead = (uint32_t *)((char *)t1Ptr + d * t1_ptr.pitch * t1_extent.height);
    //     db_sliceHead1 = (uint32_t *)((char *)dbPtr + (z + (int)pow(2, cnt)) * db_ptr.pitch * db_extent.height);
    //     for (int y = 0; y < t1_extent.height; y++)
    //     {
    //         t1_rowHead = (uint32_t*)((char *)t1_sliceHead + y * t1_ptr.pitch);
    //         db_rowHead1 = (uint32_t*)((char *)db_sliceHead1 + y * db_ptr.pitch);
    //         for (int x = 0; x < t1_extent.width / sizeof(uint32_t); x++)
    //         {
    //             t1_rowHead[x] = db_rowHead1[x] >> p128.decbit[d];
    //         }
    //     }
    // }
}


__global__ void STLU_GPU(uint32_t ***database, int flen, uint32_t *d_params, uint32_t *d_decbit, int cnt, uint32_t ****t1, uint32_t ***decvec, cuDoubleComplex ***decvecfft)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // i = 0, deal with: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    // i = 1, deal with: (1,3), (5,7), (9,11), (13,15)
    // i = 2, deal with: (3,7), (11,15)
    // i = 3, deal with: (7,15)
    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)
	int d0 = int(pow(2, cnt) - 1)+ (int)pow(2, cnt+1) * tid;
    int d1 = d0 + (int)pow(2, cnt);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < d_params[3]; j++)
        { 
            database[d1][i][j] = database[d1][i][j] - database[d0][i][j];    // ok!
            database[d1][i][j] = (uint32_t)((database[d1][i][j] + (uint64_t)d_params[0]) % _two32);   // ok!
        }
    }
    // t1 size : threadnum * p128.bk_l * 2 * p128.N
    // printf("tid: %d\n", tid);
    for (int i = 0; i < d_params[2]; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < d_params[3]; k++)
            {
                t1[tid][i][j][k] = database[d1][j][k] >> d_decbit[i];   // ok!
                t1[tid][i][j][k] = t1[tid][i][j][k] & (d_params[1] - 1);    // ok!
                t1[tid][i][j][k] = t1[tid][i][j][k] - d_params[1] / 2;   // ok!
            } 
        }
    } 
    // decvec size: threadnum * (p128.bk_l * 2) * p128.N
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < d_params[2]; i++)
        {
            for (int k = 0; k < d_params[3]; k++)
            {
                // attention!
                decvec[tid][i+j*d_params[2]][k] = t1[tid][i][j][k];
            }
        }   
    }
}


__global__ void STLU_GPU_2(uint32_t ***database, int flen, cuDoubleComplex ***id, int dlen, uint32_t *d_params, int cnt, cuDoubleComplex ***decvecfft, cuDoubleComplex ****t4, cuDoubleComplex ***t5)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // i = 0, deal with: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    // i = 1, deal with: (1,3), (5,7), (9,11), (13,15)
    // i = 2, deal with: (3,7), (11,15)
    // i = 3, deal with: (7,15)
    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)

    // t4 size: threadnum * (2 * p128.bk_l) * 2 * (p128.N / 2)
	int M = d_params[3] / 2;
    for (int i = 0; i < 2*d_params[2]; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                t4[tid][i][j][k] = cuCmul(id[i][j][k], decvecfft[tid][i][k]);
            }
        }
    }
    // t5 size: threadnum * 2 * (p128.N / 2)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            t5[tid][i][j].x = 0;
            t5[tid][i][j].y = 0;
            for (int k = 0; k < 2*d_params[2]; k++)
            {
                t5[tid][i][j] = cuCadd(t5[tid][i][j], t4[tid][k][i][j]);
            }
        }
    }
}



int GetNum(uint32_t *num_bits, int len)
{
    int num = 0;
    int i = 0;
    while (i < len)
    {
        num = num + num_bits[i] * pow(2, i);
        i++;
    }
    return num;
}

void GetBits(int num, uint32_t *num_bits, int len)
{
    for (int i = 0; i < len; i++)
    {
        num_bits[i] = 0;
    }
    
    int j = 0;
    while (num)
    {
        num_bits[j] = num % 2;
        num = num / 2;
        j++;
    }
}


void Database(Params128 p128, uint32_t ***database, int32_t *trlwekey)
{
    int dlen = p128.N;
    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    uint32_t **c = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }

    // trlwekey generation
    trlweKeyGen(trlwekey, p128);

    for (int i = 0; i < 16; i++)
    {
        // generate message       
        GetBits(i, vecmu, dlen);
        for (int j = 0; j < dlen; j++)
        {
            vecmu[j] = mutoT(vecmu[j], p128.Msize);
        }

        // encryption
        trlweSymEnc(vecmu, trlwekey, p128.ks_stdev, p128, c);
        
        // database[i] = c;
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                database[i][j][k] = c[j][k];
            }
        }

        // trlweSymDec(c, trlwekey, p128, vecmu);
        // cout << GetNum(vecmu, p128.N) << endl;
    }
    free(vecmu);
    for (int i = 0; i < 2; i++)
    {
        free(c[i]);
    }
    free(c);
}


void Test_STLU()
{
    Params128 p128 = Params128(4, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);
    int dlen = p128.N;
    int flen = pow(2, dlen);
    // database size: flen * 2 * p128.N
    uint32_t ***database = (uint32_t ***)malloc(sizeof(uint32_t **) * flen);
    for (int i = 0; i < flen; i++)
    {
        database[i] = (uint32_t **)malloc(sizeof(uint32_t *) * 2);
    }
    for (int i = 0; i < flen; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            database[i][j] = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
        }    
    }

    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));

    Database(p128, database, trlwekey);

    int idnum = 14;
    uint32_t *idbits = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    GetBits(idnum, idbits, dlen);

    uint64_t *mu0 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    uint64_t *mu1 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        mu0[i] = 0;
        mu1[i] = 0;
    }
    mu1[0] = pow(2, 32);

    int lines = 2 * p128.bk_l;
    cuDoubleComplex ****id;
    id = (cuDoubleComplex ****)malloc(sizeof(cuDoubleComplex ***) * dlen);
    for (int i = 0; i < dlen; i++)
    {
        id[i] = (cuDoubleComplex ***)malloc(sizeof(cuDoubleComplex **) * lines);
    }
    for (int i = 0; i < dlen; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            id[i][j] = (cuDoubleComplex **)malloc(sizeof(cuDoubleComplex *) * 2);
        }
    }
    for (int i = 0; i < dlen; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                id[i][j][k] = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * p128.N / 2);
            }
        }
    }
    
    for (int i = 0; i < dlen; i++)
    {
        if(idbits[i] == 1)
        {
            trgswfftSymEnc(mu1, trlwekey, p128, id[i]);
        }
        else{
            trgswfftSymEnc(mu0, trlwekey, p128, id[i]);
        }
    }


    // convert database (3 dim) to 1 dimension.
    uint32_t *h_database = (uint32_t *)malloc(sizeof(uint32_t) * p128.N * 2 * flen);
    for (int i = 0; i < flen; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                h_database[i * 2 * p128.N + j * p128.N + k] = database[i][j][k];
            }
        }    
    }
    
    for (int i = 0; i < flen; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                cout << h_database[i * 2 * p128.N + j * p128.N + k] << ",";
            }
        }    
        cout << endl;
    }
    cout <<"\n-----------------------------------------\n\n";


    cudaPitchedPtr db_ptr;
    cudaExtent db_extent;
    cudaMemcpy3DParms cpyParm;
    db_extent = make_cudaExtent(sizeof(uint32_t) * p128.N, 2, flen);
    cudaMalloc3D(&db_ptr, db_extent);
    cpyParm = {0};
    cpyParm.srcPtr = make_cudaPitchedPtr((void*)h_database, sizeof(uint32_t) * p128.N, p128.N, 2);
    cpyParm.dstPtr = db_ptr;
    cpyParm.extent = db_extent;
    cpyParm.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cpyParm);


    // t1 size: p128.bk_l * 2 * p128.N 
    cudaPitchedPtr t1_ptr;
    cudaExtent t1_extent;
    t1_extent = make_cudaExtent(sizeof(uint32_t) * p128.N, 2, p128.bk_l);
    cudaMalloc3D(&t1_ptr, t1_extent);
    uint32_t *h_t1 = (uint32_t *)malloc(sizeof(uint32_t) * p128.N * 2 * p128.bk_l);
    
    // t1 size: p128.bk_l * 2 * p128.N


    int threadnum = flen / 2;
    for (int i = 0; i < dlen; i++)
    {
        cout << "i = " << i << endl;
        cuDoubleComplex *h_id = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * lines * 2 * p128.N / 2);
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                for (int t = 0; t < p128.N / 2; t++)
                {
                    h_id[j * p128.N + k * p128.N / 2 + t] = id[i][j][k][t];
                }
            }
        }

        cudaPitchedPtr id_ptr;
        cudaExtent id_extent;
        id_extent = make_cudaExtent(sizeof(uint32_t) * p128.N / 2, 2, lines);
        cudaMalloc3D(&id_ptr, id_extent);
        cpyParm = {0};
        cpyParm.srcPtr = make_cudaPitchedPtr((void*)h_id, sizeof(uint32_t) * p128.N / 2, p128.N / 2, 2);
        cpyParm.dstPtr = id_ptr;
        cpyParm.extent = id_extent;
        cpyParm.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&cpyParm);
        cudaDeviceSynchronize();

        // db1 -db0
        Reduce<<<1,threadnum>>>(db_ptr, db_extent, id_ptr, id_extent, t1_ptr, t1_extent, p128, i);
        cudaDeviceSynchronize();

        cpyParm = {0};
        cpyParm.srcPtr = db_ptr;
        cpyParm.dstPtr = make_cudaPitchedPtr((void*)h_database, sizeof(uint32_t) * p128.N, p128.N, 2);
        cpyParm.extent = db_extent;
        cpyParm.kind = cudaMemcpyDeviceToHost;
        cudaMemcpy3D(&cpyParm);
        for (int t = (int)pow(2,i+1)-1; t < flen; t=t+(int)pow(2, i+1))
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < p128.N; k++)
                {
                    cout << h_database[t * 2 * p128.N + j * p128.N + k] << ",";
                }
            }    
            cout << endl;
        }
        cout << "\n----------------------------------------------" << endl;
        
        cpyParm = {0};
        cpyParm.srcPtr = t1_ptr;
        cpyParm.dstPtr = make_cudaPitchedPtr((void*)h_t1, sizeof(uint32_t) * p128.N, p128.N, 2);
        cpyParm.extent = t1_extent;
        cpyParm.kind = cudaMemcpyDeviceToHost;
        cudaMemcpy3D(&cpyParm);

        for (int x = 0; x < p128.bk_l; x++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int z = 0; z < p128.N; z++)
                {
                    cout << h_t1[x * 2 * p128.N + y * p128.N + z] << ",";
                }
            }
            cout << endl;
        }
        cout << "\n------------------------------------------------------------------------" << endl;
        
        threadnum = threadnum / 2;
    }
    
}


__global__ void ExternalProduct_1(uint32_t ***database, uint32_t *d_params, uint32_t *d_decbit, uint32_t ***t1, uint32_t **decvec, cuDoubleComplex **decvecfft)
{
    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < d_params[3]; j++)
        { 
            database[1][i][j] = database[1][i][j] - database[0][i][j];    // ok!
            database[1][i][j] = (uint32_t)((database[1][i][j] + (uint64_t)d_params[0]) % _two32);   // ok!
        }
    }
    // t1 size : threadnum * p128.bk_l * 2 * p128.N
    for (int i = 0; i < d_params[2]; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < d_params[3]; k++)
            {
                t1[i][j][k] = database[1][j][k] >> d_decbit[i];   // ok!
                t1[i][j][k] = t1[i][j][k] & (d_params[1] - 1);    // ok!
                t1[i][j][k] = t1[i][j][k] - d_params[1] / 2;   // ok!
            } 
        }
    } 
    // // decvec size: threadnum * (p128.bk_l * 2) * p128.N
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < d_params[2]; i++)
        {
            for (int k = 0; k < d_params[3]; k++)
            {
                // attention!
                decvec[i+j*d_params[2]][k] = t1[i][j][k];
            }
        }   
    }
}


__global__ void ExternalProduct_2(uint32_t ***database, cuDoubleComplex ***id, uint32_t *d_params, cuDoubleComplex **decvecfft, cuDoubleComplex ***t4, cuDoubleComplex **t5)
{
    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)

    // t4 size: threadnum * (2 * p128.bk_l) * 2 * (p128.N / 2)
	int M = d_params[3] / 2;
    for (int i = 0; i < 2*d_params[2]; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                t4[i][j][k] = cuCmul(id[i][j][k], decvecfft[i][k]);
            }
        }
    }
    // t5 size: threadnum * 2 * (p128.N / 2)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            t5[i][j].x = 0;
            t5[i][j].y = 0;
            for (int k = 0; k < 2*d_params[2]; k++)
            {
                t5[i][j] = cuCadd(t5[i][j], t4[k][i][j]);
            }
        }
    }
}

void Test_CMUXFFT_2()
{
    Params128 p128 = Params128(2, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);
    int M = p128.N / 2;
    // generate message
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 0)
        {
            mu[i] = mutoT(1, p128.Msize);
        }
        else
        {
            mu[i] = mutoT(0, p128.Msize);
        }
    }
    uint32_t *mu2 = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        mu2[i] = 0;
    }
    mu2[2] = mutoT(1, p128.Msize);
    mu2[1] = mutoT(1, p128.Msize);
    uint64_t *pgsw = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        pgsw[i] = 0;
    }
    pgsw[0] = pow(2, 32);
    

    // trlwekey generation
    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));
    trlweKeyGen(trlwekey, p128);


    // encryption
    int lines = 2 * p128.bk_l;
    cuDoubleComplex ***trgswfftcipher;
    trgswfftcipher = create_3d_array<cuDoubleComplex>(lines, 2, p128.N / 2);

    trgswfftSymEnc(pgsw, trlwekey, p128, trgswfftcipher);


    uint32_t **trlwecipher = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu, trlwekey, p128.ks_stdev, p128, trlwecipher);
    

    uint32_t **trlwecipher2 = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher2[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu2, trlwekey, p128.ks_stdev, p128, trlwecipher2);

    uint32_t ***database;
    database = create_3d_array<uint32_t>(2, 2, p128.N);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            database[0][i][j] = trlwecipher[i][j];
            database[1][i][j] = trlwecipher2[i][j];
        }
    }

    cout << "\n\n database:\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                cout << database[i][j][k] << ",";
            }   
        }
        cout << endl;
    }
    cout << setiosflags(ios::scientific) << setprecision(8);
    cout << "\noffset:" << p128.offset << endl;
    cout << "\n decbit: " << endl;
    for (int i = 0; i < p128.bk_l; i++)
    {
        cout << p128.decbit[i] << ",";
    }
    cout << "\nBg:" << p128.Bg << endl;
    cout << "\n twist: " << endl;
    for (int i = 0; i < p128.N / 2; i++)
    {
        cout << p128.twist[i].x << "," << p128.twist[i].y << "\t";
    }

    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)
    uint32_t *h_params = (uint32_t *)malloc(4 * sizeof(uint32_t));
    h_params[0] = p128.offset;
    h_params[1] = p128.Bg;
    h_params[2] = p128.bk_l;
    h_params[3] = p128.N;
    uint32_t *d_params;
    CHECK(cudaMalloc((void **)&d_params, 4 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_params, h_params, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t *d_decbit;
    CHECK(cudaMalloc((void **)&d_decbit, p128.bk_l * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_decbit, p128.decbit, p128.bk_l * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // t1 size: threadnum * p128.bk_l * 2 * p128.N
    uint32_t ***t1;
    t1 = create_3d_array<uint32_t>(p128.bk_l, 2, p128.N);
    // decvec size: threadnum * (p128.bk_l * 2) * p128.N
    uint32_t **decvec;
    decvec = create_2d_array<uint32_t>(p128.bk_l * 2, p128.N);
    // decvecfft size: threadnum * (p128.bk_l * 2) * (p128.N / 2)
    cuDoubleComplex **decvecfft;
    decvecfft = create_2d_array<cuDoubleComplex>(p128.bk_l * 2, p128.N / 2);
    // t4 size: threadnum * (2 * p128.bk_l) * 2 * (p128.N / 2)
    cuDoubleComplex ***t4;
    t4 = create_3d_array<cuDoubleComplex>(2 * p128.bk_l, 2, p128.N / 2);
    // t5 size: threadnum * 2 * (p128.N / 2)
    cuDoubleComplex **t5;
    t5 = create_2d_array<cuDoubleComplex>(2, p128.N / 2);
    // t6 size : threadnum * 2 * p128.N
    uint32_t **t6;
    t6 = create_2d_array<uint32_t>(2, p128.N);


    // db1 -db0
    ExternalProduct_1<<<1,1>>>(database, d_params, d_decbit, t1, decvec, decvecfft);
    cudaDeviceSynchronize();

    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        for (int j = 0; j < M; j++)
        {
            decvecfft[i][j].x = (int32_t)(decvec[i][j]);
            decvecfft[i][j].y = (int32_t)(decvec[i][j+M]);
            decvecfft[i][j] = cuCmul(decvecfft[i][j], p128.twist[j]);
        }
        cufftHandle plan;
        cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
        cufftExecZ2Z(plan, (cuDoubleComplex *)decvecfft[i], (cuDoubleComplex *)decvecfft[i], CUFFT_FORWARD);
        CHECK(cudaDeviceSynchronize());
    }
    cout << "\n\n decvecfft:\n";
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        for (int k = 0; k < p128.N / 2; k++)
        {
            cout << decvecfft[i][k].x << "," << decvecfft[i][k].y << "\t";
        }   
        cout << endl;
    }

    cout << "\n\n trgswfftcipher:\n";
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                cout << trgswfftcipher[i][j][k].x << "+" << trgswfftcipher[i][j][k].y << "j, ";
            }
            cout << endl;
        }
        cout << endl;
    }
    ExternalProduct_2<<<1, 1>>>(database, trgswfftcipher, d_params, decvecfft, t4, t5);
    cudaDeviceSynchronize();
    cout << "\n\n t4:\n";
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                cout << t4[i][j][k].x << "+" << t4[i][j][k].y << "j, ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "\n\n t5:\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N / 2; j++)
        {
            cout << t5[i][j].x << "+" << t5[i][j].y << ", ";
        }
        cout << endl;
    }
    

    for (int i = 0; i < 2; i++)
    {
        cufftHandle plan;
        cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
        cufftExecZ2Z(plan, (cuDoubleComplex *)t5[i], (cuDoubleComplex *)t5[i], CUFFT_INVERSE);
        CHECK(cudaDeviceSynchronize());
    }


    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            cuDoubleComplex twist;
            twist.x = p128.twist[j].x;
            twist.y = (-1) * p128.twist[j].y;
            // normalize
            t5[i][j].x = t5[i][j].x / M;
            t5[i][j].y = t5[i][j].y / M;
            t5[i][j] = cuCmul(t5[i][j], twist);
            t6[i][j] = (uint32_t)t5[i][j].x;
            t6[i][j+M] = (uint32_t)t5[i][j].y;
        }
    }

    cout << "\n\n t5 IFFT:\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N / 2; j++)
        {
            cout << t5[i][j].x << "+" << t5[i][j].y << ", ";
        }
        cout << endl;
    }

    cout << "\n\n t6:\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            cout << t6[i][j] << ", ";
        }
        cout << endl;
    }
    

    cout << "\n\ndatabase:\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            database[1][i][j] = t6[i][j] + database[0][i][j];
            cout << database[1][i][j] << ",";
        }
    }
    cout << endl;


    uint32_t *msg = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trlweSymDec(database[1], trlwekey, p128, msg);
    cout << "\nmsg:" << endl;
    for (int i = 0; i < p128.N; i++)
    {
       printf("%d,", msg[i]);
    }
    
}


void Test_STLU_2()
{
    Params128 p128 = Params128(4, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);
    int dlen = p128.N;
    int flen = pow(2, dlen);
    // database size: flen * 2 * p128.N
    uint32_t ***database;
    database = create_3d_array<uint32_t>(flen, 2, p128.N);

    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));

    Database(p128, database, trlwekey);
    cout << "database: " << endl;
    for (int i = 0; i < flen; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                cout << database[i][j][k] << ",";
            }    
        }  
        cout << endl;  
    }
    cout << "---------------------------------------------" << endl;

    int idnum = 10;
    uint32_t *idbits = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    GetBits(idnum, idbits, dlen);
    

    uint64_t *mu0 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    uint64_t *mu1 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        mu0[i] = 0;
        mu1[i] = 0;
    }
    mu1[0] = pow(2, 32);

    int lines = 2 * p128.bk_l;
    // id size : dlen * lines * 2 * (p128.N / 2)
    cuDoubleComplex ****id;
    id = create_4d_array<cuDoubleComplex>(dlen, lines, 2, p128.N / 2);
    
    for (int i = 0; i < dlen; i++)
    {
        if(idbits[i] == 1)
        {
            trgswfftSymEnc(mu1, trlwekey, p128, id[i]);
        }
        else{
            trgswfftSymEnc(mu0, trlwekey, p128, id[i]);
        }
    }    
    
    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)
    uint32_t *h_params = (uint32_t *)malloc(4 * sizeof(uint32_t));
    h_params[0] = p128.offset;
    h_params[1] = p128.Bg;
    h_params[2] = p128.bk_l;
    h_params[3] = p128.N;
    uint32_t *d_params;
    CHECK(cudaMalloc((void **)&d_params, 4 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_params, h_params, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t *d_decbit;
    CHECK(cudaMalloc((void **)&d_decbit, p128.bk_l * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_decbit, p128.decbit, p128.bk_l * sizeof(uint32_t), cudaMemcpyHostToDevice));


    int threadnum = flen / 2;
    int M = p128.N / 2;
    for (int cnt = 0; cnt < dlen; cnt++)
    {
        cout << "cnt = " << cnt << endl;
        
        // t1 size: threadnum * p128.bk_l * 2 * p128.N
        uint32_t ****t1;
        t1 = create_4d_array<uint32_t>(threadnum, p128.bk_l, 2, p128.N);
        // decvec size: threadnum * (p128.bk_l * 2) * p128.N
        uint32_t ***decvec;
        decvec = create_3d_array<uint32_t>(threadnum, p128.bk_l * 2, p128.N);
        // decvecfft size: threadnum * (p128.bk_l * 2) * (p128.N / 2)
        cuDoubleComplex ***decvecfft;
        decvecfft = create_3d_array<cuDoubleComplex>(threadnum, p128.bk_l * 2, p128.N / 2);
        // t4 size: threadnum * (2 * p128.bk_l) * 2 * (p128.N / 2)
        cuDoubleComplex ****t4;
        t4 = create_4d_array<cuDoubleComplex>(threadnum, 2 * p128.bk_l, 2, p128.N / 2);
        // t5 size: threadnum * 2 * (p128.N / 2)
        cuDoubleComplex ***t5;
        t5 = create_3d_array<cuDoubleComplex>(threadnum, 2, p128.N / 2);
        // t6 size : threadnum * 2 * p128.N
        uint32_t ***t6;
        t6 = create_3d_array<uint32_t>(threadnum, 2, p128.N);


        // db1 -db0
        STLU_GPU<<<1, threadnum>>>(database, flen, d_params, d_decbit, cnt, t1, decvec, decvecfft);
        cudaDeviceSynchronize();

        for (int tid = 0; tid < threadnum; tid++)
        {
            for (int i = 0; i < 2*p128.bk_l; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    decvecfft[tid][i][j].x = (int32_t)(decvec[tid][i][j]);
                    decvecfft[tid][i][j].y = (int32_t)(decvec[tid][i][j+M]);
                    decvecfft[tid][i][j] = cuCmul(decvecfft[tid][i][j], p128.twist[j]);

                }
                cufftHandle plan;
                cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
                cufftExecZ2Z(plan, (cuDoubleComplex *)decvecfft[tid][i], (cuDoubleComplex *)decvecfft[tid][i], CUFFT_FORWARD);
                CHECK(cudaDeviceSynchronize());
            }
        }

        STLU_GPU_2<<<1, threadnum>>>(database, flen, id[cnt], dlen, d_params, cnt, decvecfft, t4, t5);
        cudaDeviceSynchronize();

        for (int tid = 0; tid < threadnum; tid++)
        {
            for (int i = 0; i < 2; i++)
            {
                cufftHandle plan;
                cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
                cufftExecZ2Z(plan, (cuDoubleComplex *)t5[tid][i], (cuDoubleComplex *)t5[tid][i], CUFFT_INVERSE);
                CHECK(cudaDeviceSynchronize());
            }
        }

        for (int tid = 0; tid < threadnum; tid++)
        {
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    cuDoubleComplex twist;
                    twist.x = p128.twist[j].x;
                    twist.y = (-1) * p128.twist[j].y;
                    // normalize
                    t5[tid][i][j].x = t5[tid][i][j].x / M;
                    t5[tid][i][j].y = t5[tid][i][j].y / M;
                    t5[tid][i][j] = cuCmul(t5[tid][i][j], twist);
                    t6[tid][i][j] = (uint32_t)t5[tid][i][j].x;
                    t6[tid][i][j+M] = (uint32_t)t5[tid][i][j].y;
                }
            }
        }
        
        for (int tid = 0; tid < threadnum; tid++)
        {
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < p128.N; j++)
                {
                    int d0 = int(pow(2, cnt) - 1)+ (int)pow(2, cnt+1) * tid;
                    int d1 = d0 + (int)pow(2, cnt);
                    database[d1][i][j] = ((uint64_t)t6[tid][i][j] + (uint64_t)database[d0][i][j]) % _two32;
                    // cout << database[d1][i][j] << ",";
                }
            }
            // cout << endl;
        }
        // cout << "\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;




        cout << "\ndatabase:\n";
        for (int i = 0; i < flen; i += 1)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < p128.N; k++)
                {
                    cout << database[i][j][k] << ",";
                }
            }    
            cout << endl;
        }
        // cout << "\ndatabase:\n";
        // for (int i = int(pow(2, cnt+1) - 1); i < flen; i += pow(2, cnt+1))
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N; k++)
        //         {
        //             cout << database[i][j][k] << ",";
        //         }
        //     }    
        //     cout << endl;
        // }
        cout << "\n***************************************" << endl;
        // cout << "\nt1:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int t = 0; t < p128.bk_l; t++)
        //     {
        //         for (int j = 0; j < 2; j++)
        //         {
        //             for (int k = 0; k < p128.N; k++)
        //             {
        //                 cout << t1[i][t][j][k] << ",";
        //             }
        //         }    
        //         cout << endl;
        //     } 
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\ndecvec:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int j = 0; j < 2*p128.bk_l; j++)
        //     {
        //         for (int k = 0; k < p128.N; k++)
        //         {
        //             cout << decvec[i][j][k] << ",";
        //         }
        //         cout << endl;
        //     }    
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\ndecvecfft:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int j = 0; j < 2*p128.bk_l; j++)
        //     {
        //         for (int k = 0; k < M; k++)
        //         {
        //             cout << decvecfft[i][j][k].x << "," << decvecfft[i][j][k].y << "\t\t";
        //         }
        //         cout << endl;
        //     }    
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\nid[cnt]:\n";
        // for (int t = 0; t < 2*p128.bk_l; t++)
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N/2; k++)
        //         {
        //             cout << id[cnt][t][j][k].x << "," << id[cnt][t][j][k].y << "\t";
        //         }
        //         cout << endl;
        //     }    
        //     cout << endl;
        // } 
        // cout << "\n***************************************" << endl;
        // cout << "\nt4:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int t = 0; t < 2*p128.bk_l; t++)
        //     {
        //         for (int j = 0; j < 2; j++)
        //         {
        //             for (int k = 0; k < p128.N/2; k++)
        //             {
        //                 cout << t4[i][t][j][k].x << "," << t4[i][t][j][k].y << "\t";
        //             }
        //             cout << endl;
        //         }    
        //         cout << endl;
        //     } 
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\nt5:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N / 2; k++)
        //         {
        //             cout << t5[i][j][k].x << "," << t5[i][j][k].y << "\t";
        //         }
        //         cout << endl;
        //     }   
        //     cout << endl; 
        // }

        cout << "\n----------------------------------------------" << endl;
        threadnum = threadnum / 2;
    }

    
    uint32_t *msg = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(database[15], trlwekey, p128, msg);
    cout << "\nTable Lookup result is:\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
    }
    cout << endl;
}

void CMUXFFT(cuDoubleComplex ***CFFT, uint32_t **d1, uint32_t **d0, uint32_t **cprod, Params128 &p128)
{
    // return trgswfftExternalProduct(CFFT, d1 - d0, p128) + d0
    uint32_t **tmp = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        tmp[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            tmp[i][j] = d1[i][j] - d0[i][j];
        }
    }
    
    trgswfftExternalProduct(CFFT, tmp, p128, cprod);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            cprod[i][j] = cprod[i][j] + d0[i][j];
        }
    }

    for (int i = 0; i < 2; i++)
    {
        free(tmp[i]);
    }
    free(tmp);
}

void Test_CMUXFFT()
{
    Params128 p128 = Params128(1024, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 8);
    // generate message
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 0)
        {
            mu[i] = mutoT(1, p128.Msize);
        }
        else
        {
            mu[i] = mutoT(0, p128.Msize);
        }
    }
    uint32_t *mu2 = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        mu2[i] = 0;
    }
    mu2[2] = mutoT(6, p128.Msize);
    uint64_t *pgsw = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        pgsw[i] = 0;
    }
    pgsw[0] = pow(2, 32);
    

    // trlwekey generation
    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));
    trlweKeyGen(trlwekey, p128);


    // encryption
    int lines = 2 * p128.bk_l;
    cuDoubleComplex ***trgswfftcipher;
    trgswfftcipher = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));

    for (int i = 0; i < lines; i++)
    {
        trgswfftcipher[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgswfftcipher[i][j] = (cuDoubleComplex *)malloc(p128.N / 2 * sizeof(cuDoubleComplex));
        }
    }
    trgswfftSymEnc(pgsw, trlwekey, p128, trgswfftcipher);


    uint32_t **trlwecipher = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu, trlwekey, p128.ks_stdev, p128, trlwecipher);
    

    uint32_t **trlwecipher2 = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher2[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu2, trlwekey, p128.ks_stdev, p128, trlwecipher2);

    uint32_t **cprod = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        cprod[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }

    // when trgswfftcipher == 0, choose trlwecipher, otherwise trlwecipher2
    CMUXFFT(trgswfftcipher, trlwecipher2, trlwecipher, cprod, p128);
    
    uint32_t *msg = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(cprod, trlwekey, p128, msg);

    cout << "\n\nTest CMUXFFT result:\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
    }


    free(mu);
    free(mu2);
    free(pgsw);
    free(trlwekey);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher2[i]);
    }
    free(trlwecipher2);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(trgswfftcipher[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(trgswfftcipher[i]);
    }
    free(trgswfftcipher);
    for (int i = 0; i < 2; i++)
    {
        free(cprod[i]);
    }
    free(cprod);
    free(msg);
}


    
int main()
{

    Test_CMUXFFT_2();

    return 0;
}