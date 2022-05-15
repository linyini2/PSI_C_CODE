#include "function.cuh"
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


__global__ void STLU_GPU(uint32_t ***database, uint32_t *d_params, uint32_t *d_decbit, int cnt, uint32_t ****t1, uint32_t ***decvec, cuDoubleComplex ***decvecfft)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // i = 0, deal with: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    // i = 1, deal with: (1,3), (5,7), (9,11), (13,15)
    // i = 2, deal with: (3,7), (11,15)
    // i = 3, deal with: (7,15)
    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)
	// int d0 = int(pow(2, cnt) - 1)+ (int)pow(2, cnt+1) * tid;
    // int d1 = d0 + (int)pow(2, cnt);
    int d0 = (1 << cnt) - 1+ (1 << (cnt+1)) * tid;
    int d1 = d0 + (1 << cnt);
    // printf("\nd0: %d", d0);
    // printf("\nd1: %d\n", d1);
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


__global__ void STLU_GPU_2(uint32_t ***database, cuDoubleComplex ***id, uint32_t *d_params, int cnt, cuDoubleComplex ***decvecfft, cuDoubleComplex ****t4, cuDoubleComplex ***t5)
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

    Database(p128, database, flen, trlwekey);

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
    Params128 p128 = Params128(1024, 630, 2, 10, 9, pow(2.0, -15.4), pow(2.0, -28), 2);
    int dlen = 13;
    int flen = 1 << dlen;
    // attention! host and device should have a backup of database respectively.
    // database size: flen * 2 * p128.N
    uint32_t ***database;
    database = create_3d_array<uint32_t>(flen, 2, p128.N);

    int32_t *trlwekey = (int32_t *)malloc(p128.N * sizeof(int32_t));

    Database(p128, database, flen, trlwekey);
    // cout << "database: " << endl;
    // for (int i = 0; i < flen; i++)
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
    // cout << "---------------------------------------------" << endl;

    int idnum = 23;
    uint32_t *idbits = (uint32_t *)malloc(dlen * sizeof(uint32_t));
    GetBits(idnum, idbits, dlen); 
    for (int i = 0; i < dlen; i++)
    {
        cout << idbits[i] << " ";
    }
    cout << endl << endl;
    

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
        // cout << "\n\ncnt = " << cnt << endl;
        
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
        // STLU_GPU<<<1, threadnum>>>(database, flen, d_params, d_decbit, cnt, t1, decvec, decvecfft);
        if (threadnum > 1024)
        {
            STLU_GPU<<<threadnum/1024, 1024>>>(database, d_params, d_decbit, cnt, t1, decvec, decvecfft);
            cudaDeviceSynchronize();
        }
        else{
            // 1, threadnum
            STLU_GPU<<<1, threadnum>>>(database, d_params, d_decbit, cnt, t1, decvec, decvecfft);
            cudaDeviceSynchronize();
        }

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

        if (threadnum > 1024)
        {
            STLU_GPU_2<<<threadnum / 1024, 1024>>>(database, id[cnt], d_params, cnt, decvecfft, t4, t5);
            cudaDeviceSynchronize();
        }
        else{
            STLU_GPU_2<<<1, threadnum>>>(database, id[cnt], d_params, cnt, decvecfft, t4, t5);
            cudaDeviceSynchronize();
        }
        

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
                    int d0 = (1 << cnt) - 1+ (1 << (cnt+1)) * tid;
                    int d1 = d0 + (1 << cnt);
                    database[d1][i][j] = ((uint64_t)t6[tid][i][j] + (uint64_t)database[d0][i][j]) % _two32;
                    // cout << database[d1][i][j] << ",";
                }
            }
            // cout << endl;
        }

        // cout << setiosflags(ios::scientific) << setprecision(8);
        // cout << "\ndatabase:\n";
        // for (int i = 0; i < flen; i += 1)
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
        // cout << "database:\n";
        // for (int i = (1 << (cnt + 1)) - 1; i < flen; i += (1 << (cnt + 1)))
        // {
        //     cout << "i: " << i << endl;
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N; k++)
        //         {
        //             cout << database[i][j][k] << ",";
        //         }
        //     }    
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
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
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl; 
        // cout << "\ndecvecfft:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int j = 0; j < 2*p128.bk_l; j++)
        //     {
        //         for (int k = 0; k < M; k++)
        //         {
        //             cout << decvecfft[i][j][k].x << "+" << decvecfft[i][j][k].y << "j, ";
        //         }
        //         cout << endl;
        //     }    
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\n id:\n";
        // for (int i = 0; i < lines; i++)
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N / 2; k++)
        //         {
        //             cout << id[cnt][i][j][k].x << "+" << id[cnt][i][j][k].y << "j, ";
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
        // cout << "\n***************************************" << endl;
        // cout << "\nt6:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N; k++)
        //         {
        //             cout << t6[i][j][k] << "\t";
        //         }
        //         cout << endl;
        //     }   
        //     cout << endl; 
        // }

        // cout << "\n----------------------------------------------" << endl;
        threadnum = threadnum / 2;
    }

    
    uint32_t *msg = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(database[flen - 1], trlwekey, p128, msg);
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

    Test_STLU_2();

    return 0;
}