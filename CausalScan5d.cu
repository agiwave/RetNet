#ifndef __COMMON_H__
#define __COMMON_H__
template <typename scalar_t> struct wrap_t{
    int x, y, z, l, n, s;
    scalar_t* p;
};
typedef struct{
    int x, y, z;
}INDICS;
#define SHAPE5D(t) {\
    (int)t.size(0), (int)t.size(2), (int)t.size(3),\
    (int)t.size(1), \
    (int)t.size(4), \
    t.size(1) == 1 ? 0 : ((int)t.size(2) * (int)t.size(3) * (int)t.size(4)),\
    (scalar_t*)t.data_ptr()\
}
#define IDX5D(shape) ((((blockIdx.x % shape.x) * shape.l * shape.y + blockIdx.y % shape.y) * shape.z + blockIdx.z % shape.z) * shape.n + threadIdx.x % shape.n)
#define Ptr5D(shape) (shape.p + ((((blockIdx.x % shape.x) * shape.l * shape.y + blockIdx.y % shape.y ) * shape.z + blockIdx.z % shape.z) * shape.n + threadIdx.x % shape.n))
#define GROUP_SIZE 1023
#endif//__COMMON_H__

#ifndef __DISABLE_CUDA__
    #define DEVICEINDICS 
    #define CAUSAL_FORWARD causalScan5d_Forward_cuda
    #define CAUSAL_BACKWARD causalScan5d_Backward_cuda
    #define atomAdd atomicAdd
    #include <cuda.h>
    #include <cuda_runtime.h>
#else//__DISABLE_CUDA__
    #ifdef DEVICEINDICS
        #undef DEVICEINDICS
    #endif//
    #ifdef CAUSAL_FORWARD
        #undef CAUSAL_FORWARD
    #endif//
    #ifdef CAUSAL_BACKWARD
        #undef CAUSAL_BACKWARD
    #endif//
    #ifdef __global__
        #undef __global__
    #endif//
    #ifdef atomAdd
        #undef atomAdd
    #endif//
    #define DEVICEINDICS ,const INDICS& blockIdx, const INDICS& threadIdx
    #define CAUSAL_FORWARD causalScan5d_Forward_cpu
    #define CAUSAL_BACKWARD causalScan5d_Backward_cpu
    #define __global__
    #define atomAdd(p,b) (*(p) = *(p) + (b))
#endif//__DISABLE_CUDA__

namespace { namespace device {
    template <typename scalar_t> __global__ void CAUSAL_FORWARD(
        const wrap_t<scalar_t> shapeX,
        const wrap_t<scalar_t> shapeZ,
        const wrap_t<scalar_t> shapeA,
        const wrap_t<scalar_t> shapeB,
        const wrap_t<scalar_t> shapeC,
        const wrap_t<scalar_t> shapeO
        DEVICEINDICS
    )
    {
        scalar_t * pX = Ptr5D(shapeX);
        scalar_t * pZ = Ptr5D(shapeZ);
        scalar_t * pA = Ptr5D(shapeA);
        scalar_t * pB = Ptr5D(shapeB);
        scalar_t * pC = Ptr5D(shapeC);
        scalar_t * pO = Ptr5D(shapeO);
        scalar_t * pH = pZ;
        scalar_t zh = *pZ;
        int i = 0;
        while(i++<shapeO.l) {
            if( i % 1024 == 0 ) {
                pH += shapeZ.s;
                *pH = zh;
            }
            zh = (*pA) * zh + (*pB) * (*pX);
            atomAdd(pO, ((*pC) * zh));
            pX += shapeX.s;
            pA += shapeA.s;
            pB += shapeB.s;
            pC += shapeC.s;
            pO += shapeO.s;
        }
        pZ[(shapeZ.l-1)*shapeZ.s] = zh;
    }

    template <typename scalar_t> __global__ void CAUSAL_BACKWARD(
        scalar_t * pX,
        scalar_t * pZ,
        scalar_t * pA,
        scalar_t * pB,
        scalar_t * pC,
        const wrap_t<scalar_t> gradO,
        const wrap_t<scalar_t> gradX,
        const wrap_t<scalar_t> gradZ,
        const wrap_t<scalar_t> gradA,
        const wrap_t<scalar_t> gradB,
        const wrap_t<scalar_t> gradC
        DEVICEINDICS
    )
    {
        int length = gradO.l;
        int sx = IDX5D(gradX);
        int sz = IDX5D(gradZ);
        int sa = IDX5D(gradA);
        int sb = IDX5D(gradB);
        int sc = IDX5D(gradC);
        pX += sx;
        pZ += sz;
        pA += sa;
        pB += sb;
        pC += sc;
        scalar_t * pGradO = Ptr5D(gradO);
        scalar_t * pGradX = gradX.p + sx;
        scalar_t * pGradZ = gradZ.p + sz;
        scalar_t * pGradA = gradA.p + sa;
        scalar_t * pGradB = gradB.p + sb;
        scalar_t * pGradC = gradC.p + sc;

        scalar_t gradh = 0.0;
        scalar_t zhs[GROUP_SIZE+1];
        int groups = (length + GROUP_SIZE - 1) / GROUP_SIZE;
        for(int igroups=groups-1; igroups>=0; igroups--){
            int ibegin = igroups * GROUP_SIZE;
            int group_length = (igroups==groups-1)?(length-ibegin):GROUP_SIZE;

            scalar_t * pIX = pX + ibegin*gradX.s;
            scalar_t * pIA = pA + ibegin*gradA.s;
            scalar_t * pIB = pB + ibegin*gradB.s;
            zhs[0] = pZ[igroups*gradZ.s];
            for(int i=0; i<group_length; i++) {
                zhs[i+1] = (*pIA) * zhs[i] + (*pIB) * (*pIX);
                pIA += gradA.s;
                pIB += gradB.s;
                pIX += gradX.s;
            }

            int iend = ibegin + group_length;
            scalar_t * pIC = pC + iend * gradC.s;
            scalar_t * pIGradO = pGradO + iend * gradO.s;
            scalar_t * pIGradX = pGradX + iend * gradX.s;
            scalar_t * pIGradA = pGradA + iend * gradA.s;
            scalar_t * pIGradB = pGradB + iend * gradB.s;
            scalar_t * pIGradC = pGradC + iend * gradC.s;
            while(group_length-->0) {
                pIA -= gradA.s;
                pIB -= gradB.s;
                pIX -= gradX.s;
                pIC -= gradC.s;
                pIGradO -= gradO.s;
                pIGradA -= gradA.s;
                pIGradB -= gradB.s;
                pIGradX -= gradX.s;
                pIGradC -= gradC.s;

                atomAdd(pIGradC, (*pIGradO) * zhs[group_length+1]);
                gradh += (*pIGradO) * (*pC);
                atomAdd(pIGradB, gradh * (*pX));
                atomAdd(pIGradX, gradh * (*pB));
                atomAdd(pIGradA, zhs[group_length] * gradh);
                gradh *= (*pIA);
            }
        }
        *pGradZ = gradh;
    }
}}

#ifndef __TORCH_INLINE__
#define __TORCH_INLINE__

#ifndef __DISABLE_CUDA__
#define __DISABLE_CUDA__
#include "CausalScan5d.cu"
#undef __DISABLE_CUDA__
#endif//__DISABLE_CUDA__

#include <torch/extension.h>
#include <vector>
torch::Tensor causalScan5d_Forward(
    torch::Tensor X, 
    torch::Tensor Z, 
    torch::Tensor A,
    torch::Tensor B, 
    torch::Tensor C
) {
    auto O = torch::zeros_like(X);
    if(X.is_cuda()) {
        #ifndef __DISABLE_CUDA__
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan5d_Forward", ([&] {
            wrap_t<scalar_t> shapeX = SHAPE5D(X);
            wrap_t<scalar_t> shapeZ = SHAPE5D(Z);
            wrap_t<scalar_t> shapeA = SHAPE5D(A);
            wrap_t<scalar_t> shapeB = SHAPE5D(B);
            wrap_t<scalar_t> shapeC = SHAPE5D(C);
            wrap_t<scalar_t> shapeO = SHAPE5D(O);
            int threads = shapeZ.n;
            const dim3 blocks(O.size(0), O.size(2), O.size(3));    
            device::causalScan5d_Forward_cuda<scalar_t><<<blocks, threads>>>(
                shapeX,
                shapeZ,
                shapeA,
                shapeB,
                shapeC,
                shapeO
            );
        }));
        #else
        AT_ASSERT(false);
        #endif//__DISABLE_CUDA__
    }
    else{
        AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan5d_cpu_Forward", ([&] {
            wrap_t<scalar_t> shapeX = SHAPE5D(X);
            wrap_t<scalar_t> shapeZ = SHAPE5D(Z);
            wrap_t<scalar_t> shapeA = SHAPE5D(A);
            wrap_t<scalar_t> shapeB = SHAPE5D(B);
            wrap_t<scalar_t> shapeC = SHAPE5D(C);
            wrap_t<scalar_t> shapeO = SHAPE5D(O);
            for(int ib=0; ib<shapeZ.x; ib++)
            for(int ih=0; ih<shapeZ.y; ih++)
            for(int id=0; id<shapeZ.z; id++)
            for(int in=0; in<shapeZ.n; in++)
            {
                INDICS indics[] = {
                    {ib, ih, id},
                    {in}
                };
                device::causalScan5d_Forward_cpu<scalar_t>(
                    shapeX,
                    shapeZ,
                    shapeA,
                    shapeB,
                    shapeC,
                    shapeO,
                    indics[0],
                    indics[1]
                );
            }
            // int stepy = shapeZ.z * shapeZ.n;
            // int stepx = stepy * shapeZ.y;
            // at::parallel_for(0, shapeZ.x * stepx, 0, [&](int64_t start, int64_t end){
            //     while(start<end){
            //         INDICS indics[] = {
            //             {(int)(start/stepx), (int)((start/stepy)%shapeZ.y), (int)((start/shapeZ.n)%shapeZ.z)},
            //             {(int)(start%shapeZ.n)}
            //         };
            //         start++;
            //     };
            // });
        }));
    }
    return O;
}

std::vector<torch::Tensor> causalScan5d_Backward(
    torch::Tensor gradO,
    torch::Tensor X,
    torch::Tensor Z,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
) {
    auto gradX = torch::zeros_like(X);
    auto gradZ = torch::zeros_like(Z);
    auto gradA = torch::zeros_like(A);
    auto gradB = torch::zeros_like(B);
    auto gradC = torch::zeros_like(C);
    if(gradO.is_cuda()){
        #ifndef __DISABLE_CUDA__
        AT_DISPATCH_FLOATING_TYPES(gradO.scalar_type(), "causalScan5d_Backward", ([&] {
            wrap_t<scalar_t> deltaO = SHAPE5D(gradO);
            wrap_t<scalar_t> deltaX = SHAPE5D(gradX);
            wrap_t<scalar_t> deltaZ = SHAPE5D(gradZ);
            wrap_t<scalar_t> deltaA = SHAPE5D(gradA);
            wrap_t<scalar_t> deltaB = SHAPE5D(gradB);
            wrap_t<scalar_t> deltaC = SHAPE5D(gradC);
            int threads = deltaZ.n;
            const dim3 blocks(gradO.size(0), gradO.size(2), gradO.size(3));
            device::causalScan5d_Backward_cuda<scalar_t><<<blocks, threads>>>(
                (scalar_t*)X.data_ptr(),
                (scalar_t*)Z.data_ptr(),
                (scalar_t*)A.data_ptr(),
                (scalar_t*)B.data_ptr(),
                (scalar_t*)C.data_ptr(),
                deltaO,
                deltaX,
                deltaZ,
                deltaA,
                deltaB,
                deltaC
            );
        }));
        #else
        AT_ASSERT(false);
        #endif//__DISABLE_CUDA__
    }
    else{
        AT_DISPATCH_FLOATING_TYPES(gradO.scalar_type(), "causalScan5d_cpu_Backward", ([&] {
            wrap_t<scalar_t> deltaX = SHAPE5D(gradX);
            wrap_t<scalar_t> deltaO = SHAPE5D(gradO);
            wrap_t<scalar_t> deltaZ = SHAPE5D(gradZ);
            wrap_t<scalar_t> deltaA = SHAPE5D(gradA);
            wrap_t<scalar_t> deltaB = SHAPE5D(gradB);
            wrap_t<scalar_t> deltaC = SHAPE5D(gradC);

            for(int ib=0; ib<deltaZ.x; ib++)
            for(int ih=0; ih<deltaZ.y; ih++)
            for(int id=0; id<deltaZ.z; id++)
            for(int in=0; in<deltaZ.n; in++)
            {
                INDICS indics[] = {
                    {ib, ih, id},
                    {in}
                };
                device::causalScan5d_Backward_cpu<scalar_t>(
                    (scalar_t*)X.data_ptr(),
                    (scalar_t*)Z.data_ptr(),
                    (scalar_t*)A.data_ptr(),
                    (scalar_t*)B.data_ptr(),
                    (scalar_t*)C.data_ptr(),
                    deltaO,
                    deltaX,
                    deltaZ,
                    deltaA,
                    deltaB,
                    deltaC,
                    indics[0],
                    indics[1]
                );
            }
            // int stepy = deltaZ.z * deltaZ.n;
            // int stepx = stepy * deltaZ.y;
            // at::parallel_for(0, deltaZ.x * stepx, 0, [&](int64_t start, int64_t end){
            //     while(start<end){
            //         INDICS indics[] = {
            //             {(int)(start/stepx), (int)((start/stepy)%deltaZ.y), (int)((start/deltaZ.n)%deltaZ.z)},
            //             {(int)(start%deltaZ.n)}
            //         };
            //         start++;
            //     };
            // });
        }));
    }

    return {gradX, gradZ, gradA, gradB, gradC};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan5d_Forward, "");
    m.def("backward", &causalScan5d_Backward, "");
}
#endif//__TORCH_INLINE__