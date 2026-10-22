#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "exl3_devctx.cuh"
#include "../util.h"
#include "../util.cuh"

//DevCtx::DevCtc()
//{
//    int num_sms[MAX_DEVICES] = {};
//    int cc[MAX_DEVICES] = {};
//    void* locks[MAX_DEVICES] = {};
//    std::mutex mtx;
//}

DevCtx& DevCtx::instance()
{
    static DevCtx ctx;
    return ctx;
}

int DevCtx::get_num_sms(int device)
{
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "EXL3 device index out of range");
    std::lock_guard<std::mutex> lock(mtx);
    if (!num_sms[device])
        cuda_check(cudaDeviceGetAttribute(&num_sms[device], cudaDevAttrMultiProcessorCount, device));
    return num_sms[device];
}

int DevCtx::get_cc(int device)
{
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "EXL3 device index out of range");
    std::lock_guard<std::mutex> lock(mtx);
    if (!cc[device])
    {
        cudaDeviceProp prop;
        cuda_check(cudaGetDeviceProperties(&prop, device));
        if (prop.major >= 10) cc[device] = CC_BLACKWELL;
        else if (prop.major >= 9) cc[device] = CC_HOPPER;
        else if (prop.major >= 8 && prop.minor >= 9) cc[device] = CC_ADA;
        else if (prop.major >= 8) cc[device] = CC_AMPERE;
        else cc[device] = CC_OLD;
    }
    return cc[device];
}

void* DevCtx::get_ws(int device)
{
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "EXL3 device index out of range");
    std::lock_guard<std::mutex> lock(mtx);
    if (!ws[device].defined())
    {
        c10::cuda::CUDAGuard guard(device);
        ws[device] = at::empty({WORKSPACE_SIZE}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA, device));
    }
    return ws[device].data_ptr();
}

int* DevCtx::get_locks(int device)
{
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "EXL3 device index out of range");
    std::lock_guard<std::mutex> lock(mtx);
    if (!locks[device].defined())
    {
        c10::cuda::CUDAGuard guard(device);
        size_t elements = MAX_TILES_C + MAX_BARRIERS * 2;
        auto initialized = at::empty({static_cast<int64_t>(elements)}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, device));
        auto stream = at::cuda::getCurrentCUDAStream(device).stream();
        cuda_check(cudaMemsetAsync(initialized.data_ptr(), 0, elements * sizeof(int), stream));
        cuda_check(cudaStreamSynchronize(stream));
        locks[device] = std::move(initialized);
    }
    return locks[device].data_ptr<int>();
}

int g_get_cc(int device)
{
    return DevCtx::instance().get_cc(device);
}

int g_get_num_sms(int device)
{
    return DevCtx::instance().get_num_sms(device);
}

void prepare_ctx(int device)
{
    DevCtx::instance().get_num_sms(device);
    DevCtx::instance().get_cc(device);
    DevCtx::instance().get_locks(device);
}
