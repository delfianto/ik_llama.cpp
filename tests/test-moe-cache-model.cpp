#include "common.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include "ggml-backend.h"

// Teacher-forced comparison: both contexts consume identical fixed tokens.
// Pass the usual model/placement options and a nonzero --moe-cache-mib budget.
int main(int argc, char ** argv) {
    gpt_params params;
    params.n_ctx = 2048;
    params.n_batch = 64;
    params.n_ubatch = 64;
    params.n_predict = 256;
    params.prompt = "Summarize why a persistent cache helps repeated matrix operations. "
        "Write Python and SQL to count distinct customers. Explain 17 times 23. "
        "Jelaskan manfaat cache dalam bahasa Indonesia. Voici une petite histoire. ";
    if (!gpt_params_parse(argc, argv, params) || params.moe_cache_size == 0) {
        fprintf(stderr, "Provide a model and a nonzero --moe-cache-mib budget\n");
        return 1;
    }
    llama_backend_init();
    auto mp = common_model_params_to_llama(params);
    auto * model = llama_model_load_from_file(params.model.c_str(), mp);
    if (!model) return 1;
    auto cp = common_context_params_to_llama(params);
    struct capture { bool cached = false; std::unordered_map<std::string,std::vector<float>> values; } observed;
    if (std::getenv("MOE_TEST_DEBUG")) {
        cp.cb_eval = [](ggml_tensor * t, bool ask, void * user) -> int {
            if (t->type != GGML_TYPE_F32 || std::strncmp(t->name,"ffn",3)) return 0;
            if (ask) return 1;
            auto & c = *static_cast<capture *>(user);
            std::vector<float> values(ggml_nelements(t));
            ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t));
            if (!c.cached) c.values[t->name] = values;
            else if (c.values.count(t->name) && c.values[t->name].size() == values.size()) {
                double err=0,norm=0;auto & ref=c.values[t->name];
                for(size_t i=0;i<values.size();++i) {err+=double(values[i]-ref[i])*(values[i]-ref[i]);norm+=double(ref[i])*ref[i];}
                if(err/std::max(norm,1e-20)>1e-5) fprintf(stderr,"DIFF %s %.8g\n",t->name,err/std::max(norm,1e-20));
            }
            return 1;
        };
        cp.cb_eval_user_data = &observed;
    }
    const auto budget = cp.moe_cache_size;
    cp.moe_cache_size = 0;
    auto * control = llama_init_from_model(model, cp);
    cp.moe_cache_size = budget;
    auto * cached = llama_init_from_model(model, cp);
    if (!control || !cached) {
        llama_free(cached); llama_free(control);
        llama_free_model(model);
        return 1;
    }
    std::string text;
    for (int i = 0; i < 32; ++i) text += params.prompt;
    auto tokens = common_tokenize(control, text, true);
    tokens.resize(std::min(tokens.size(), size_t(std::min(params.n_predict, params.n_ctx))));
    auto batch = llama_batch_init(64, 0, 1);
    const int sizes[] = {1, 1, 1, 2, 8, 32, 33, 64};
    size_t offset = 0, step = 0;
    double maximum = 0;
    int greedy_differences = 0;
    bool ok = true;
    while (offset < tokens.size() && ok) {
        int count = std::min<size_t>(sizes[step++ % 8], tokens.size() - offset);
        common_batch_clear(batch);
        for (int i = 0; i < count; ++i) common_batch_add(batch, tokens[offset + i], offset + i, {0}, true);
        observed.cached = false;
        if (llama_decode(control, batch)) { ok = false; break; }
        observed.cached = true;
        if (llama_decode(cached, batch)) { ok = false; break; }
        const int nv = llama_n_vocab(model);
        for (int i = 0; i < count; ++i) {
            auto * a = llama_get_logits_ith(control, i);
            auto * b = llama_get_logits_ith(cached, i);
            double error = 0, norm = 0;
            for (int j = 0; j < nv; ++j) {
                if (!std::isfinite(a[j]) || !std::isfinite(b[j])) { ok = false; break; }
                error += double(a[j] - b[j]) * (a[j] - b[j]); norm += double(a[j]) * a[j];
            }
            double nmse = error / std::max(norm, 1e-20);
            maximum = std::max(maximum, nmse);
            ok &= nmse <= 5e-4;
            greedy_differences += (std::max_element(a, a + nv) - a) != (std::max_element(b, b + nv) - b);
        }
        offset += count;
    }
    printf("{\"tokens\":%zu,\"max_nmse\":%.9g,\"greedy_differences\":%d,\"passed\":%s}\n",
           offset, maximum, greedy_differences, ok ? "true" : "false");
    llama_batch_free(batch); llama_free(cached); llama_free(control);
    llama_free_model(model); llama_backend_free();
    return ok ? 0 : 1;
}
