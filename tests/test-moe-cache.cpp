#include "llama-moe-cache-lru.h"
#include "llama-moe-cache.h"
#include "llama-model.h"
#include "ggml-cpp.h"
#include "moe-cache-args.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static void test_lru() {
    moe_cache_lru lru;
    lru.init(2, 8, 3);
    std::vector<int32_t> a(8,-1), b(8,-1);
    lru.slot_map = {a.data(),b.data()};
    std::vector<moe_cache_lru::fill> fills;
    size_t hits;
    auto plan = [&](int layer, std::vector<int32_t> ids) { return lru.plan(layer,ids.data(),ids.size(),fills,hits); };
    check(plan(0,{1,1,2}) && fills.size()==2 && hits==0,"deduplicate misses");
    check(plan(0,{1,2}) && fills.empty() && hits==2,"repeated hits");
    check(plan(0,{1,3,4}) && hits==1 && fills.size()==2 && a[1]>=0 && a[2]<0,"protect selected hits during eviction");
    check(plan(1,{5,6}) && b[5]>=0 && b[6]>=0 && lru.evictions>=2,"cross-layer eviction");
    const auto before=lru.key_of;
    check(!plan(0,{0,1,2,3}) && lru.key_of==before,"overflow does not mutate residency");
    lru.seen_gen=UINT32_MAX;
    check(plan(1,{5,5,6}) && fills.empty() && hits==2,"generation wrap");
    moe_cache_lru other; other.init(1,8,1);
    std::vector<int32_t> c(8,-1); other.slot_map[0]=c.data();
    int32_t id=7;check(other.plan(0,&id,1,fills,hits) && c[7]==0,"one slot");
    check(b[5]>=0 && b[6]>=0,"pool isolation");
}
static void test_args() {
    check(parse_moe_cache_mib("0")==0 && parse_moe_cache_mib("512")==512u*1024*1024,"MiB conversion");
    for (auto s : {"-1","1.5","+2","", "18446744073709551615"}) {
        bool rejected=false;try { (void)parse_moe_cache_mib(s); } catch (const std::exception &) { rejected=true; }
        check(rejected,"invalid budget accepted");
    }
}
#ifdef GGML_USE_CUDA
static void test_cuda(int device, bool roomy = false, ggml_type type = GGML_TYPE_Q4_0) {
    ggml_backend_ptr gpu(ggml_backend_cuda_init(device,nullptr,nullptr));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    check(bool(gpu),"initialize CUDA");
    llama_model model;
    model.split_mode=LLAMA_SPLIT_MODE_LAYER;
    model.hparams.n_layer=2;model.hparams.n_expert=40;model.hparams.n_expert_used=8;
    model.devices={device};model.default_layer_device={0,0};model.layers.resize(2);
    ggml_init_params params={64*ggml_tensor_overhead(),nullptr,true};
    auto * weight_ctx=ggml_init(params);model.ctxs.push_back(weight_ctx);
    for (int il=0;il<2;++il) {
        auto & l=model.layers[il];
        l.ffn_gate_exps=ggml_new_tensor_3d(weight_ctx,type,1536,512,40);
        l.ffn_up_exps=ggml_new_tensor_3d(weight_ctx,type,1536,512,40);
        l.ffn_down_exps=ggml_new_tensor_3d(weight_ctx,type,512,1536,40);
    }
    auto * weight_buf=ggml_backend_alloc_ctx_tensors(weight_ctx,cpu.get());model.bufs.push_back(weight_buf);
    ggml_backend_buffer_set_usage(weight_buf,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> values;
    int wi=0;
    for(auto * w=ggml_get_first_tensor(weight_ctx);w;w=ggml_get_next_tensor(weight_ctx,w)) {
        values.resize(ggml_nelements(w));
        for(size_t i=0;i<values.size();++i) values[i]=0.05f*std::sin(float(i+17*wi)*0.071f);
        ggml_quantize_chunk(w->type,values.data(),w->data,0,w->ne[1]*w->ne[2],w->ne[0],nullptr,nullptr);++wi;
    }
    std::vector<ggml_backend_t> backends={gpu.get(),cpu.get()};
    std::vector<ggml_backend_buffer_type_t> bufts={ggml_backend_get_default_buffer_type(gpu.get()),ggml_backend_cpu_buffer_type()};
    // A decode working set plus a zero guard, shared across two layers.
    llama_moe_cache cache(model,backends,bufts,(roomy ? 61 : 9)*(model.layers[0].ffn_up_exps->nb[2]+model.layers[0].ffn_gate_exps->nb[2]+model.layers[0].ffn_down_exps->nb[2])+1024);
    check(!cache.get_slot_map(0,33,8),"batch 33 bypass");
    for (int n : {2,8,32}) check(!cache.get_slot_map(0,n,8),"multi-token bypass");
    check(cache.get_slot_map(0,1,8)!=nullptr,"one-token map");
    auto run=[&](int il,int e0,int e1,bool fused,int nt = 1) {
        ggml_context_ptr ctx(ggml_init({1024*1024,nullptr,true}));
        auto * x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,1536,1,nt);ggml_set_input(x);
        auto * ids_storage=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,40,nt);ggml_set_input(ids_storage);
        auto * ids=ggml_view_2d(ctx.get(),ids_storage,8,nt,ids_storage->nb[1],0);
        // Inputs get host storage independently of scheduler graph allocations.
        ggml_backend_buffer_ptr input_buf(ggml_backend_alloc_ctx_tensors(ctx.get(),cpu.get()));
        std::vector<float> input(1536*nt);for(int i=0;i<1536*nt;++i) input[i]=std::cos(i*0.12f);
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        std::vector<int32_t> selected(40*nt);for(int i=0;i<nt;++i) for(int j=0;j<8;++j) selected[40*i+j]=(e0+j+(roomy ? i : 0))%40;
        (void)e1;
        ggml_backend_tensor_set(ids_storage,selected.data(),0,selected.size()*sizeof(int32_t));
        auto * flat=ggml_reshape_1d(ctx.get(),ggml_cont(ctx.get(),ids),8*nt);
        auto * lookup=ggml_get_rows(ctx.get(),cache.get_slot_map(il,nt,8),flat);
        lookup->op_params[GGML_MAX_OP_PARAMS/sizeof(int32_t)-1]=0xff;
        auto * slots=ggml_reshape_2d(ctx.get(),lookup,8,nt);
        auto & l=model.layers[il];
        auto * output=fused ? ggml_moe_up_gate(ctx.get(),cache.get_experts(l.ffn_up_exps),cache.get_experts(l.ffn_gate_exps),x,slots,GGML_UNARY_OP_SILU)
                            : ggml_mul_mat_id(ctx.get(),cache.get_experts(l.ffn_up_exps),x,slots);
        auto * reference=fused ? ggml_moe_up_gate(ctx.get(),l.ffn_up_exps,l.ffn_gate_exps,x,ids,GGML_UNARY_OP_SILU)
                               : ggml_mul_mat_id(ctx.get(),l.ffn_up_exps,x,ids);
        auto * reference_up = reference;
        output = ggml_mul_mat_id(ctx.get(),cache.get_experts(l.ffn_down_exps),output,slots);
        reference = ggml_mul_mat_id(ctx.get(),l.ffn_down_exps,reference,ids);
        auto * graph=ggml_new_graph_custom(ctx.get(),128,false);
        ggml_build_forward_expand(graph,output);ggml_build_forward_expand(graph,reference);
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),bufts.data(),2,128,false));
        ggml_backend_sched_set_tensor_backend(sched.get(),lookup,gpu.get());
        ggml_backend_sched_set_tensor_backend(sched.get(),reference_up,gpu.get());
        ggml_backend_sched_set_tensor_backend(sched.get(),reference,gpu.get());
        ggml_backend_sched_set_copy_callback(sched.get(),[](ggml_backend_t backend,const ggml_tensor * src,ggml_tensor * dst,ggml_cgraph * g,void * user) {
            return static_cast<llama_moe_cache *>(user)->copy(backend,src,dst,g);
        },&cache);
        check(ggml_backend_sched_alloc_graph(sched.get(),graph),"allocate cached graph");
        check(ggml_backend_sched_graph_compute(sched.get(),graph)==GGML_STATUS_SUCCESS,"compute cached graph");
        std::vector<float> a(ggml_nelements(output)),b(a.size());
        ggml_backend_tensor_get(output,a.data(),0,a.size()*sizeof(float));ggml_backend_tensor_get(reference,b.data(),0,b.size()*sizeof(float));
        double err=0,norm=0;for(size_t i=0;i<a.size();++i) {check(std::isfinite(a[i]),"nonfinite result");err+=(a[i]-b[i])*(a[i]-b[i]);norm+=b[i]*b[i];}
        fprintf(stderr,"NMSE device=%d type=%s roomy=%d fused=%d batch=%d value=%.9g\n",device,ggml_type_name(type),roomy,fused,nt,err/std::max(norm,1e-20));
        check(err/std::max(norm,1e-20)<=5e-4,"cached CUDA NMSE");
        // Every resident matrix must be byte-identical to the authoritative host slice.
        auto * map=cache.get_slot_map(il,1,8); const auto * mapping=static_cast<const int32_t *>(map->data);
        for(auto * w : {l.ffn_up_exps,l.ffn_gate_exps,l.ffn_down_exps}) {
            auto * bank=cache.get_experts(w);std::vector<char> raw(w->nb[2]);
            for(int e : {e0,e0+1}) {ggml_backend_tensor_get(bank,raw.data(),mapping[e]*bank->nb[2],raw.size());
                check(std::memcmp(raw.data(),static_cast<const char *>(w->data)+e*w->nb[2],raw.size())==0,"expert byte identity");}
        }
    };
    for(int pass=0;pass<2;++pass) for(bool fused : {false,true}) {
        run(0,1,2,fused);run(0,1,2,fused);run(1,3,4,fused);run(0,1,5,fused);
    }

    ggml_backend_synchronize(gpu.get());
}
#endif
int main() {
    try {
        test_lru();test_args();
#ifdef GGML_USE_CUDA
        for(int device=0;device<ggml_backend_cuda_get_device_count();++device) {for(auto type : {GGML_TYPE_Q4_0,GGML_TYPE_Q4_K,GGML_TYPE_Q6_K}) {test_cuda(device,false,type);test_cuda(device,true,type);}}
#endif
        puts("MoE cache tests passed");return 0;
    } catch(const std::exception & e) {fprintf(stderr,"MoE cache test failed: %s\n",e.what());return 1;}
}
