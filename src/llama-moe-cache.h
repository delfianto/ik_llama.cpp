#pragma once
#include "ggml-backend.h"
#include <map>
#include <memory>
#include <vector>
struct llama_model;
// Context-owned device banks; host weights remain authoritative.
class llama_moe_cache {
public:
    llama_moe_cache(const llama_model &, const std::vector<ggml_backend_t> &,
                    const std::vector<ggml_backend_buffer_type_t> &, size_t);
    ~llama_moe_cache();
    ggml_backend_t backend(int32_t il) const;
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_used) const;
    ggml_tensor * get_experts(const ggml_tensor * w) const;
    bool copy(ggml_backend_t, const ggml_tensor *, ggml_tensor *, ggml_cgraph *);
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
