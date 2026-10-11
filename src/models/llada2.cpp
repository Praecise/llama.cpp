#include "models.h"

void llama_model_llada2::load_arch_hparams(llama_model_loader & ml) {
    llama_model_bailingmoe2::load_arch_hparams(ml);
    hparams.causal_attn = false;
}
