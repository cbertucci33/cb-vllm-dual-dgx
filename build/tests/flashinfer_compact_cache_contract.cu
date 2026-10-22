#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/kv_cache_io.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/smem_layout.cuh>
using GLM = KVCacheTraits<ModelType::GLM53_NOPE>;
static_assert(bytes_per_token(ModelType::GLM53_NOPE) == 528);
static_assert(GLM::KV_GMEM_STRIDE == 528);
static_assert(GLM::KV_SMEM_COPY_BYTES == 528);
static_assert(GLM::KV_ROPE_GMEM_OFFSET == 528);
static_assert(GLM::SCALE_BYTES_PER_TOKEN == 16);
static_assert(GLM::D_ROPE == 0);
static_assert(KVIOTraits<ModelType::GLM53_NOPE>::IO_STRIDE == 528);
static_assert(SmemLayoutSwapAB<ModelType::GLM53_NOPE>::KV_STRIDE == 528);
static_assert(bytes_per_token(ModelType::DSV3_2) == 656);
static_assert(bytes_per_token(ModelType::GLM_NSA) == 656);
static_assert(KVCacheTraits<ModelType::GLM_NSA>::KV_GMEM_STRIDE == 656);
