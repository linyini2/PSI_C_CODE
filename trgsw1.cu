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
using namespace std;

typedef std::complex<double> Complex;
#define PI acos(-1)
static const int64_t _two31 = INT64_C(1) << 31; // 2^31
static const int64_t _two32 = INT64_C(1) << 32; // 2^32
typedef uint32_t Torus32;
#define BATCH 1
#define NRANK 2

__host__ cuDoubleComplex cexp(const cuDoubleComplex &z)
{
    Complex stl_complex(cuCreal(z), cuCimag(z));
    stl_complex = exp(stl_complex);
    return make_cuDoubleComplex(real(stl_complex), imag(stl_complex));
}
__host__ __device__ double carg(const cuDoubleComplex &z) { return (double)atan2(cuCimag(z), cuCreal(z)); } // polar angle
__host__ __device__ double cabs(const cuDoubleComplex &z) { return (double)cuCabs(z); }
__host__ __device__ cuDoubleComplex cp2c(const double d, const double a) { return make_cuDoubleComplex(d * cos(a), d * sin(a)); }
__host__ __device__ cuDoubleComplex cpow(const cuDoubleComplex &z, const int &n) { return make_cuDoubleComplex((pow(cabs(z), n) * cos(n * carg(z))), (pow(cabs(z), n) * sin(n * carg(z)))); }

class Params128
{
public:
    int N;
    int n;
    int bk_l;
    int bk_Bgbit;
    int bk_Bgbitbar;
    float ks_stdev;
    float bk_stdev;
    int Bg;
    int Bgbar;
    int Msize;
    double *H;
    double offset;
    int *decbit;
    uint32_t inter;
    cuDoubleComplex *twist;

public:
    Params128(int N, int n, int bk_l, int bk_Bgbit, int bk_Bgbitbar, float ks_stdev, float bk_stdev, int Msize) : N(N),
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
            twist[k] = cexp(make_cuDoubleComplex(0, 2 * float(k) * PI / this->N));
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
            twist[k] = cexp(make_cuDoubleComplex(0, 2 * float(k) * PI / this->N));
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
            twist[k] = cexp(make_cuDoubleComplex(0, 2 * float(k) * PI / this->N));
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

uint32_t Ttomu(uint32_t phase, uint32_t inter)
{
    uint32_t half = uint32_t(inter / 2);
    return uint32_t(uint32_t(phase + half) / inter);
}

void gaussian32(uint32_t *vecmu, uint32_t *ga, Params128 p128, int size = 1)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, p128.ks_stdev);

    for (size_t i = 0; i < size; i++)
    {
        ga[i] = dtot32(dis(gen)) + vecmu[i];
    }
}

void trlweKeyGen(int32_t *h_trlwekey, Params128 p128)
{
    srand((int)time(NULL));
    for (int i = 0; i < p128.N; i++)
    {
        h_trlwekey[i] = rand() % 2;
    }
}

void TwistFFT(int32_t *a, Params128 p128, cuDoubleComplex *h_Comp_a)
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


void TwistIFFT(cuDoubleComplex *h_Comp_a, Params128 p128, uint32_t *product)
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


void PolyMul(uint32_t *a, int32_t *trlwekey, uint32_t *product, Params128 p128)
{
    int M = p128.N / 2;
    cuDoubleComplex *h_Comp_a = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    cuDoubleComplex *h_Comp_trlwekey = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    cuDoubleComplex *h_Comp_product = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    uint32_t *result = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);

    TwistFFT((int32_t *)a, p128, h_Comp_a);
    TwistFFT(trlwekey, p128, h_Comp_trlwekey);

    for (int i = 0; i < M; i++)
    {
        h_Comp_product[i] = cuCmul(h_Comp_a[i], h_Comp_trlwekey[i]);
    }
    TwistIFFT(h_Comp_product, p128, result);
    for (int i = 0; i < p128.N; i++)
    {
        product[i] = (uint32_t)(result[i] % _two32);
    }
    

    free(h_Comp_a);
    free(h_Comp_trlwekey);
    free(h_Comp_product);
}

void trlweSymEnc(uint32_t *vecmu, int32_t *trlwekey, Params128 p128, uint32_t **c)
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
    gaussian32(vecmu, ga, p128, p128.N);

    for (int i = 0; i < p128.N; i++)
    {
        c[0][i] = (ga[i] - product[i]) % (_two32);
    }

    free(a);
    free(product);
    free(ga);
}

void trlweSymEnc_2(uint32_t *vecmu, int32_t *trlwekey, Params128 p128, uint64_t **c)
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
    gaussian32(vecmu, ga, p128, p128.N);

    for (int i = 0; i < p128.N; i++)
    {
        c[0][i] = (ga[i] - product[i]) % (_two32);
    }

    free(a);
    free(product);
    free(ga);
}

void trlweSymDec(uint32_t **c, int32_t *trlwekey, Params128 p128, uint32_t *mu)
{
    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    PolyMul(c[1], trlwekey, product, p128);

    uint32_t *phase = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        phase[i] = c[0][i] + product[i];
    }
    cout << "\n mu begin:\n";
    for (int i = 0; i < p128.N; i++)
    {
        mu[i] = Ttomu(phase[i], p128.inter);
        cout << mu[i] << endl;
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
        if (i % 2 == 0)
        {
            vecmu[i] = mutoT(1, p128.Msize);
        }
        else
        {
            vecmu[i] = mutoT(0, p128.Msize);
        }
    }
    vecmu[p128.N - 1] = mutoT(1, p128.Msize);

    // trlwekey generation
    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));
    trlweKeyGen(trlwekey, p128);

    // encryption
    uint32_t **c = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(vecmu, trlwekey, p128, c);
    

    // decryption
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trlweSymDec(c, trlwekey, p128, mu);
    cout << "\ntrlwe decryption result: " << endl;
    for (int i = 0; i < p128.N; i++)
    {
        cout << mu[i] << " ";
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



void trgswSymEnc(uint32_t *vecmu, int32_t *trlwekey, Params128 p128, uint32_t*** c)
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
        trlweSymEnc(vec_zero, trlwekey, p128, c[i]); 
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
}

void trgswSymEnc_2(uint64_t *vecmu, int32_t *trlwekey, Params128 p128, uint32_t*** c)
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
        trlweSymEnc(vec_zero, trlwekey, p128, c[i]); 
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
}

void trgswSymDec(uint32_t ***c, int32_t *trlwekey, Params128 p128, uint32_t* mu)
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

void trgswSymDec_2(uint32_t ***c, int32_t *trlwekey, Params128 p128, uint32_t* mu)
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
    
    Params128 p128 = Params128(16, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);

    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N / 2; i++)
    {
        if (i % 2 == 0)
        {
            vecmu[i] = pow(2, 32);
        }
        else
        {
            vecmu[i] = mutoT(1, p128.Msize);
        }
    }
    for (int i = p128.N / 2; i < p128.N; i++)
    {
        vecmu[i] = mutoT(1, p128.Msize);
    }
    // vecmu[p128.N-1] = 0;
    for (int i = 0; i < p128.N; i++)
    {
        cout << vecmu[i] << " ";
    }
    

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
    }
    

    free(vecmu);
    free(mu);
    // how to free dim3 pointer ? (for example, c)
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

void Test_TRGSW_2()
{
    
    Params128 p128 = Params128(8, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);

    uint64_t *vecmu = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 0)
        {
            vecmu[i] = pow(2, 32);
        }
        else
        {
            vecmu[i] = 0;
        }
    }
    
    for (int i = 0; i < p128.N; i++)
    {
        cout << vecmu[i] << " ";
    }
    

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
    trgswSymEnc_2(vecmu, trlwekey, p128, c);


    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trgswSymDec_2(c, trlwekey, p128, mu);
    cout << "\ntrgsw decryption result:" << endl;
    for (int i = 0; i < p128.N; i++)
    {
        cout << mu[i] << " ";
    }
    

    free(vecmu);
    free(mu);
    // how to free dim3 pointer ? (for example, c)
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

void trgswfftSymEnc(uint32_t *mu, int32_t *trlwekey, Params128 p128, cuDoubleComplex ***trgswfftcipher)
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
    trgswSymEnc(mu, trlwekey, p128, trgsw);

    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            TwistFFT((int32_t *)trgsw[i][j], p128, trgswfftcipher[i][j]);
        }   
    }
}


void trgswfftSymEnc_2(uint64_t *mu, int32_t *trlwekey, Params128 p128, cuDoubleComplex ***trgswfftcipher)
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
            TwistFFT((int32_t *)trgsw[i][j], p128, trgswfftcipher[i][j]);
        }   
    }
}

void trgswfftExternalProduct(cuDoubleComplex ***trgswfftcipher, uint32_t **trlwecipher, Params128 p128, uint32_t **t6)
{
    int lines = 2 * p128.bk_l;
    int M = p128.N / 2;
    cout << "\ntrlwecipher begin:\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            cout << trlwecipher[i][j] << " ";
        }
        cout << endl;
    }

    // t = np.uint32((trlwecipher + p128.offset)%2**32)
    // t size: 2 * p128.N      ok!
    uint32_t **t = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        t[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    cout << "offset: " << (int64_t)p128.offset << endl;
    cout << "\nt begin:\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            // trlwecipher + offset may surpass uint32_t ??
            t[i][j] = (uint32_t)((trlwecipher[i][j] + (int64_t)p128.offset) % _two32);
            cout << t[i][j] << " ";
        }
        cout << endl;
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

    cout << "\nt1 begin\n";
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t[j][k] >> p128.decbit[i];
                cout << t1[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    // t2=t1&(p128.Bg - 1)
    // t2 size: p128.bk_l * 2 * p128.N        ok!
    cout << "\nt2 begin:\n";
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t1[i][j][k] & (p128.Bg - 1);
                cout << t1[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    // t3=t2-p128.Bg // 2
    // t3 size: p128.bk_l * 2 * p128.N        ok!
    cout << "\n t3 begin:\n";
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t1[i][j][k] - p128.Bg / 2;
                cout << t1[i][j][k] << " ";
            } 
            cout << endl;  
        }
        cout << endl;
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
    cout << "\ndecvec begin:\n";
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            cout << decvec[i][j] << " ";
        }
        cout << endl;
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
    
    cout << "\ndecvecfft begin:\n";
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        for (int j = 0; j < p128.N / 2; j++)
        {
            cout << decvecfft[i][j].x << "," << decvecfft[i][j].y << "  ";
        }
        cout << endl;
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
    cout << "\ntrgswfftcipher begin\n";
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                cout << trgswfftcipher[i][j][k].x << ", " << trgswfftcipher[i][j][k].y << "  ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "\nt4 begin\n";
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                t4[i][j][k] = cuCmul(decvecfft[i][k], trgswfftcipher[i][j][k]);
                cout << t4[i][j][k].x << ", " << t4[i][j][k].y << "  ";
            }
            cout << endl;
        }
        cout << endl;
    }

    
    // t5 = t4.sum(axis=0)
    // t5 size: 2 * (p128.N / 2)    ok!
    cuDoubleComplex **t5 = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    for (int i = 0; i < 2; i++)
    {
        t5[i] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
    }
    cout << "t5 begin:\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            t5[i][j].x = 0;
            t5[i][j].y = 0;
            for (int k = 0; k < lines; k++)
            {
                t5[i][j] = cuCadd(t5[i][j], t4[k][i][j]);
                cout << t5[i][j].x << "," << t5[i][j].y << "  ";
            }
            cout << endl;
        }
    }

    // t6 = TwistIFFT(t5, p128.twist, axis=1)
    // t6 size : 2 * p128.N
    // TwistIFFT_2(t5, p128, t6);
    for (int i = 0; i < 2; i++)
    {
        TwistIFFT(t5[i], p128, t6[i]);
    }
    
    cout << "\nt6 begin:" << endl;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            cout << t6[i][j] << " ";
        }
        cout << endl;
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
    for (int i = 0; i < 2; i++)
    {
        free(t5[i]);
    }
    free(t5);
}

void Test_ExternalProduct()
{
    Params128 p128 = Params128(8, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);
    // generate message
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    cout << "\nmu begin\n";
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
        cout << mu[i] << " ";
    }
    mu[p128.N-1]=mutoT(1, p128.Msize);
    uint64_t *pgsw = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        pgsw[i] = 0;
        // pgsw[i] = pow(2, 32);
    }
    pgsw[0] = pow(2, 32);

    cout << "\npgsw beigin\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << pgsw[i] << " ";
    }

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
    // trgswfftSymEnc((uint32_t *)pgsw, trlwekey, p128, trgswfftcipher);
    trgswfftSymEnc_2(pgsw, trlwekey, p128, trgswfftcipher);


    uint32_t **trlwecipher = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu, trlwekey, p128, trlwecipher);
    

    uint32_t **cprod = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        cprod[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trgswfftExternalProduct(trgswfftcipher, trlwecipher, p128, cprod);
    
    uint32_t *msg = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(cprod, trlwekey, p128, msg);

    // cout << "\ncprod begin:\n";
    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < p128.N; j++)
    //     {
    //         cout << cprod[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    
    cout << "external product decryption result:\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
    }
    


    free(mu);
    free(pgsw);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
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
    // Test_TRLWE();
    // cout << "\n\nsuccess" << endl;
    // Test_TRGSW_2();
    Test_ExternalProduct();
    // TestIFFT();

    return 0;
}