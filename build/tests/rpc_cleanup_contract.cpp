#include "main.cpp"
#include <cassert>
int main() {
  using namespace instanttensor;
  LoaderManager rpc_manager;
  auto input = std::make_unique<SPSCQueue<RPCRequest>>();
  auto output = std::make_shared<SPSCQueue<RPCResponse>>();
  rpc_manager.loader_handle_to_queue[41] = RPCHandle{0, input.get(), output};
  std::thread responder([&] {
    RPCRequest request;
    input->pop(request);
    assert(request.op == CLOSE);
    auto error = std::make_exception_ptr(std::runtime_error("cleanup failure"));
    output->push(RPCResponse{request.id, std::any(error)});
  });
  bool received_error = false;
  try {
    rpc_manager.close(41);
  } catch (const std::exception& error) {
    received_error = std::string(error.what()) == "cleanup failure";
  }
  responder.join();
  assert(received_error);
  assert(rpc_manager.loader_handle_to_queue.empty());
  std::cout << "PASS CLOSE error reply consumed; handle registry erased\n";
}
