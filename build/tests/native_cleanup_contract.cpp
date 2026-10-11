#include <instant_tensor/loader.hpp>
#include <cassert>
#include <algorithm>
#include <cerrno>
using namespace instanttensor;
static std::vector<std::string> releases;
static int failures = 0;
static void* host_pointer = nullptr;
extern "C" void __real_free(void*);
extern "C" void __wrap_free(void* pointer) {
  if (pointer && pointer == host_pointer) {
    releases.push_back("host-free");
    host_pointer = nullptr;
  }
  __real_free(pointer);
}
static cudaError_t allocate(void** pointer, size_t size) {
  *pointer = std::malloc(size);
  return cudaSuccess;
}
static cudaError_t release_device(void* pointer) {
  releases.push_back("device-free");
  std::free(pointer);
  return failures & 4 ? static_cast<cudaError_t>(704) : cudaSuccess;
}
static cudaError_t register_host(void* pointer, size_t, unsigned int) {
  host_pointer = pointer;
  return cudaSuccess;
}
static cudaError_t unregister_host(void*) {
  releases.push_back("host-unregister");
  return failures & 8 ? static_cast<cudaError_t>(708) : cudaSuccess;
}
static cudaError_t destroy_stream(cudaStream_t stream) {
  int id = static_cast<int>(reinterpret_cast<uintptr_t>(stream));
  releases.push_back("stream" + std::to_string(id));
  return failures & id ? static_cast<cudaError_t>(700 + id) : cudaSuccess;
}
static const char* error_text(cudaError_t) { return "injected CUDA failure"; }
int main(int argc, char** argv) {
  failures = argc > 1 ? std::atoi(argv[1]) : 0;
  cuda_binding::cudaMalloc_fn = allocate;
  cuda_binding::cudaFree_fn = release_device;
  cuda_binding::cudaHostRegister_fn = register_host;
  cuda_binding::cudaHostUnregister_fn = unregister_host;
  cuda_binding::cudaStreamDestroy_fn = destroy_stream;
  cuda_binding::cudaGetErrorString_fn = error_text;
  auto input = std::make_unique<SPSCQueue<RPCRequest>>();
  auto output = std::make_shared<SPSCQueue<RPCResponse>>();
  Loader loader(std::move(input), output);
  loader.backend = Backend::AIO;
  loader.world_size = 1;
  loader.rank_chunk_size = PAGE_SIZE;
  loader.io_depth = 1;
  loader.need_host_buffer = true;
  loader.init_buffer();
  loader.initialize_aio_context();
  loader.cuda_stream = reinterpret_cast<void*>(1);
  loader.nccl_stream = reinterpret_cast<void*>(2);
  int pipe_fds[2];
  assert(pipe(pipe_fds) == 0);
  FileInfo file;
  file.fd = pipe_fds[0];
  loader.file_info.push_back(file);
  std::string first_error;
  try {
    loader.close(CloseArgs{});
  } catch (const std::exception& error) {
    first_error = error.what();
  }
  const std::vector<std::string> expected = {
      "stream1", "stream2", "device-free", "host-unregister", "host-free"};
  assert(releases == expected);
  assert(loader.threads_joined);
  assert(loader.device_buffer == nullptr);
  assert(loader.host_buffer_entry.ptr == nullptr);
  errno = 0;
  assert(fcntl(pipe_fds[0], F_GETFD) == -1 && errno == EBADF);
  close(pipe_fds[1]);
  if (!failures)
    assert(first_error.empty());
  else {
    int expected_code = failures & 1   ? 701
                        : failures & 2 ? 702
                        : failures & 4 ? 704
                                       : 708;
    assert(first_error.find(std::to_string(expected_code)) !=
           std::string::npos);
  }
  // Mapped-file cleanup must close its fd even when unregister fails.
  loader.use_internal_memory_register = true;
  assert(pipe(pipe_fds) == 0);
  file.fd = pipe_fds[0];
  file.mapped_memory = mmap(nullptr, PAGE_SIZE, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(file.mapped_memory != MAP_FAILED);
  file.size = PAGE_SIZE;
  try {
    loader.close_file_inmem(file);
  } catch (const std::exception&) {
    assert(failures & 8);
  }
  errno = 0;
  assert(fcntl(pipe_fds[0], F_GETFD) == -1 && errno == EBADF);
  close(pipe_fds[1]);
  std::cout << "PASS fault-mask=" << failures
            << " releases=5 first-error-preserved=1 fd-closed=1\n";
}
