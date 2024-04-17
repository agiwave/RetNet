import aka.nn as nn
import aka.numpy as np
try:
    from CausalScan5d import CausalScan
    causalScan = CausalScan.apply
except ImportError:
    causalScan = None
    print('Warn: CausalScan4d import failured.')

def RetentionBlock(**kwargs):
    def __init__(self,**kwargs):
        args = nn.Object(**kwargs)
        self.embed_dim = args.latent_dim
        self.value_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.num_heads = args.num_heads
        self.scaling = (self.embed_dim // self.num_heads)**-0.5

        if getattr(args, 'xproj', True):
            from Xproj import XprojBlock
            self.xproj = XprojBlock(**dict(kwargs, kv_dims=[self.embed_dim, self.embed_dim, self.value_dim]))
        else:
            self.xproj = None
        self.group_norm = nn.RMSNorm(self.value_dim // self.num_heads)
        self.decay = nn.Parameter(
            data=np.log(1 - 2 ** (-5 - np.arange(self.num_heads, dtype=np.float))),
            requires_grad=getattr(args,'lr',False)
        )
        return self

    def apply_rotary_emb(x, cache, pos=0):
        _,L,_,D = x.shape
        slen = pos+L
        emb = cache.get('rotary_emb', None)
        if emb is None or len(emb[0]) < slen:
            angle = 1.0 / (10000 ** np.linspace(0, 1, D//2, dtype=x.dtype, device=x.device))
            angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
            index = np.arange(slen, dtype=x.dtype, device=x.device)
            sin = np.sin(index[:, None, None] * angle[None, None, :])
            cos = np.cos(index[:, None, None] * angle[None, None, :])
            cache['rotary_emb'] = (sin, cos)
        else:
            (sin,cos) = emb
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        y = np.stack((-x2, x1), dim=-1).flatten(-2)
        return (x * cos[pos:pos+L]) + (y * sin[pos:pos+L])

    def forward(self, x, kv=None, cache={}, state=None, **kwargs): 
        if self.xproj is not None:
            ((C, B), x, go) = self.xproj.proj_in(x, state=state)
        else:
            ((C, B), x) = (kv, x)

        b, l, _ = x.size()
        # q, k, v, g = [proj(x) for proj in [self.q_proj, self.k_proj, self.v_proj, self.g_proj]]
        C, B, x = [t.view(b, l, self.num_heads, -1) for t in [C,B,x]]
        B *= self.scaling

        # -- rotary embedding --
        C = apply_rotary_emb(C, cache)
        B = apply_rotary_emb(B, cache)

        if causalScan is not None:
            ssm_state = None if state is None else state.get('ssm_state',None)
            ssm_state = ssm_state if ssm_state is not None else np.zeros(b, 1, self.num_heads, self.value_dim//self.num_heads, self.embed_dim//self.num_heads, dtype=x.dtype, device=x.device)
            A = np.exp(self.decay.view(1,1,self.num_heads,1,1))
            B = B.unsqueeze(-2)
            C = C.unsqueeze(-2)
            x, ssm_state = causalScan(x, ssm_state, A, B, C)
            if state is not None:
                state['ssm_state'] = ssm_state.detach()
        else:
            C, B = [np.rearrange('b l h d -> b h l d', t) for t in [C, B]]
            x = np.rearrange('b l n d -> b n l d', x)

            # Pure py implementation. Pool performance and Memory efficiency.
            def compute_mask(decay, qlen, klen):
                if qlen == 1:
                    return np.exp(decay)[None, :, None, None]
                else:
                    index = np.arange(klen).to(decay)
                    mask = np.tril(np.ones(qlen, klen), diagonal=klen-qlen).to(decay)
                    mask = np.masked_fill(
                        index[-qlen:, None] - index[None, :], ~mask.bool(), float("inf")
                    )
                    mask = np.exp(mask * decay[:, None, None])
                    mask = np.nan_to_num(mask)
                    mask = mask.unsqueeze(0)  # [1, h, t, t]
                    mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
                    mask = np.nan_to_num(mask, nan=0.0)
                    return mask
                
            # -- qkv (Q @ K * D) @ V --
            decay_mask = compute_mask(self.decay, C.size(2), B.size(2))
            y = np.einsum('bhld,bhmd,bhlm,bhmv->blhv', C, B, decay_mask, x)

            # -- state --
            if state is not None:
                current_S = np.einsum('bhld,bhlv,bhl->bhvd', B, x, decay_mask[:, :, -1])
                prev_S = state.get('prev_S',None) # ->[b, h, d, v]
                if prev_S is not None:
                    decay = decay_mask[:, :, :, 0] # ->[b, h, t]
                    # S += S0 * (gamma ** n)
                    current_S += np.einsum('bhvd,bh->bhvd', prev_S, decay[:,:,-1])
                    # V += Q @ decay * S0
                    y += np.einsum('bhld,bhvd,bhl->blhv', C, prev_S, decay)
                state["prev_S"] = current_S.detach()
            x = y

        # Gate and Output
        x = self.group_norm(x).reshape(b, x.size(1), self.value_dim)
        if self.xproj is not None:
            return self.xproj.proj_out(x, go)
        else:
            return x
    return __init__(nn.Module(forward=forward), **kwargs)

def RetentionArgs(name):
    args = nn.Object(
        vocab_dim = 32,
        latent_dim = 384,
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    mlp_args = nn.Object(
        name = 'Xproj',
        hidden_dim = 384 * 3
    ),
    attn_args = nn.Object(
        name = 'Retention',
        num_heads = 8,
        num_kv_groups = 8,
        rotary_embedding = True
    ),
    match name:
        case 'Ret':
            args.layers = [attn_args, mlp_args]*3
        case _:
            assert False, f"Unknown Ret name{name}"
    return args

def RetNet(name):
    import aka.repo as repo

    # -- Tokenizer --
    tokenizer = repo.AutoTokenizer(name)
    cfg = repo.fopen(name, 'config.json', ftype='json')
    args = dict(
        tokenizer = tokenizer,
        vocab_size = cfg['vocab_size'],
        latent_dim = cfg['decoder_embed_dim'],
        lm_head = True,
        prev_norm = 'rms',
        bias = False,
        dropout = cfg['dropout'],
        layers = [
            dict(
                name = 'Retention',
                num_heads = cfg['decoder_retention_heads'],
                hidden_dim = cfg['decoder_value_embed_dim'],
                gate = 'gh',
                activation = 'silu'
            ), 
            dict(
                name = 'Xproj',
                hidden_dim = cfg['decoder_ffn_embed_dim'],
                gate = 'gh' if cfg['use_glu'] else None,
                activation = cfg['activation_fn']
            )
        ]*cfg['decoder_layers']
    )

    # -- Model --
    from CausalLM import CausalLM
    m = CausalLM(**args)
    if repo.exist(name, "model.safetensors"):
        with repo.fopen(name, "model.safetensors", ftype='safetensor') as f:
            keys = f.keys()
            with np.no_grad():
                m.lm_head.weight.copy_(f.get_tensor(f'lm_head.weight'))
                m.embedding.weight.copy_(f.get_tensor('model.embed_tokens.weight'))
                m.post_norm.weight.copy_(f.get_tensor(f'model.layer_norm.weight'))
                m.prev_norm.weight.copy_(f.get_tensor(f'model.layernorm_embedding.weight'))
                for i in range(len(m.layers)//2):
                    m.layers[i*2].norm.weight.copy_(f.get_tensor(f'model.layers.{i}.retention_layer_norm.weight'))
                    m.layers[i*2].layer.xproj.copy_xproj_weights(
                        [
                            f.get_tensor(f'model.layers.{i}.retention.q_proj.weight'),
                            f.get_tensor(f'model.layers.{i}.retention.k_proj.weight'),
                            f.get_tensor(f'model.layers.{i}.retention.v_proj.weight'),
                            f.get_tensor(f'model.layers.{i}.retention.g_proj.weight')
                        ],
                        f.get_tensor(f'model.layers.{i}.retention.out_proj.weight')
                    )
                    m.layers[i*2+1].norm.weight.copy_(f.get_tensor(f'model.layers.{i}.final_layer_norm.weight'))

                    # Take care here. gate and fc1 are just swaped.
                    m.layers[i*2+1].layer.copy_xproj_weights(
                        [f.get_tensor(f'model.layers.{i}.ffn.gate.weight'),
                        f.get_tensor(f'model.layers.{i}.ffn.fc1.weight')],
                        f.get_tensor(f'model.layers.{i}.ffn.fc2.weight')
                    )
    return m

if __name__ == "__main__":
    m = RetNet('data/SDPrompt-RetNet-300M')
    print('Model loaded')
    for w in m.generator("1girl"):
        print(w, end='')
# <s> 1girl, absurdres, animal ear fluff, animal ears, bangs, bare shoulders, black hair, blue archive, blunt bangs, blush, closed mouth, collarbone, commentary request, eyes visible through hair, green eyes, hair between eyes, halo, hand on own face, hand up, highres, jacket, kisaki blue archive, long hair, long sleeves, looking at viewer, open clothes, open jacket, shinonome asu, simple background, solo, track jacket, upper body, white background, white jacket</s>
    # from RomeArena import TrainRoles
    # TrainRoles([
    #     # 'Gemma-20m', 
    #     'Retention-Ret',
    # ], lr = 6e-4, epochs=4)
