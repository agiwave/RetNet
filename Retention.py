import aka.nn as nn
import aka.numpy as np

def RetentionBlock(args):
    def RMSNorm(dim: int, eps: float = 1e-5):
        '''
        Reference: LLaMA and Gemma
        '''
        def forward(self, x):
            x = (x.float() * np.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
            return x * self.weight
        return nn.Module(
            forward = forward,
            eps = eps,
            weight = nn.Parameter(np.ones(dim)))

    def get_activation_fn(activation):
        if activation == "relu":
            return np.relu
        elif activation == "gelu":
            return np.gelu
        elif activation == "swish":
            return np.silu
        else:
            raise NotImplementedError

    def __init__(self,args):
        # config: RetNetConfig,
        gate_fn="swish"
        use_bias=args.bias
        tensor_parallel=False
        self.config = args.attn_args
        self.embed_dim = args.latent_dim
        self.value_dim = getattr(args.attn_args, 'hidden_dim', args.latent_dim)
        self.num_heads = args.attn_args.num_heads
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim**-0.5
        self.rot_embedding = getattr(args.attn_args, 'rotary_embedding', False),
        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.value_dim, bias=use_bias)
        self.g_proj = nn.Linear(self.embed_dim, self.value_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.value_dim, self.embed_dim, bias=use_bias)
        self.group_norm = RMSNorm(self.head_dim)
        if tensor_parallel:
            self.decay_proj = nn.Linear(self.num_heads, self.num_heads, bias=False)
        else:
            self.decay_proj = None

        angle = 1.0 / (
            10000 ** np.linspace(0, 1, self.embed_dim // self.num_heads // 2)
        )
        self.angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        return self

    def compute_mask(self, slen, latent_dim, num_heads, get_decay_scale=True, retention_mask=None, mode='parallel'):
        angle = 1.0 / (
            10000 ** np.linspace(0, 1, latent_dim // num_heads // 2)
        )
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = np.log(
            1 - 2 ** (-5 - np.arange(num_heads, dtype=np.float))
        )
        if mode == "recurrent":
            sin = np.sin(self.angle * (slen - 1))
            cos = np.cos(self.angle * (slen - 1))
            retention_rel_pos = (decay.view(1, -1, 1, 1).exp(), None, None)
        else:
            index = np.arange(slen).to(decay)
            sin = np.sin(index[:, None] * angle[None, :])
            cos = np.cos(index[:, None] * angle[None, :])
            mask = np.tril(np.ones(slen, slen)).to(decay)
            mask = np.masked_fill(
                index[:, None] - index[None, :], ~mask.bool(), float("inf")
            )
            mask = np.exp(mask * decay[:, None, None])
            mask = np.nan_to_num(mask)
            mask = mask.unsqueeze(0)  # [1, h, t, t]

            # scaling
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            mask = np.nan_to_num(mask, nan=0.0)

            # decay_scale (used for kv cache)
            if get_decay_scale:
                exponent = np.arange(slen, device=decay.device).float()
                decay_scale = decay.exp().view(-1, 1) ** exponent.view(1, -1)  # [h, t]
                if retention_mask is not None:
                    seqlen = retention_mask.sum(dim=-1)  # [b,]
                    bsz = seqlen.size(0)
                    decay_scale = decay_scale.unsqueeze(0).repeat(bsz, 1, 1)  # [b, h, t]
                    for i, pos in enumerate(seqlen):
                        decay_scale[i, :, pos.item() :] = 0
                else:
                    bsz = 1
                decay_scale = decay_scale.sum(-1).view(bsz, -1, 1, 1)  # [b, h, 1, 1]
            else:
                decay_scale = None

            # mask processing for intra decay
            if retention_mask is not None:
                max_non_zero = (
                    np.cumsum(retention_mask, dim=-1).max(dim=-1).indices
                )  # [b,]
                intra_decay = mask[range(mask.shape[0]), :, max_non_zero]
            else:
                intra_decay = mask[:, :, -1]
            retention_rel_pos = (mask, intra_decay, decay_scale)
        return (cos,sin),retention_rel_pos

    def recurrent_retention(
        self, q, k, v, decay, kv_cache=None, retention_mask=None
    ):
        """
        q, k, v, # bsz * num_head * 1 * qkv_dim
        kv_cache:
            - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
            - "scale"  # (1 or bsz) * num_head * 1 * 1
        decay # (1 or bsz) * num_head * 1 * 1
        retention_mask # bsz * 1
        """
        if retention_mask is not None:
            retention_mask = retention_mask.float().view(-1, 1, 1, 1).to(decay)
        else:
            retention_mask = np.ones(k.size(0), 1, 1, 1).to(decay)
        # (b, h, v_dim, qk_dim)
        current_kv = k * v.transpose(-1, -2) * retention_mask

        if kv_cache is not None and "prev_key_value" in kv_cache:
            prev_kv = kv_cache["prev_key_value"]
            prev_scale = kv_cache["scale"]
            scale = np.where(retention_mask == 0, prev_scale, prev_scale * decay + 1)
            # connect prev_kv and current_kv
            # how much to decay prev_kv
            decay_amount = prev_scale.sqrt() * decay / scale.sqrt()
            decay_amount = np.where(retention_mask == 0, 1, decay_amount)
            prev_kv = prev_kv * decay_amount  # decay prev_kv
            current_kv = current_kv / scale.sqrt()  # scale current_kv
            current_kv = np.nan_to_num(
                current_kv, nan=0.0
            )  # remove nan, scale might be 0

            current_kv = prev_kv + current_kv
        else:
            scale = np.ones_like(decay)
            # when retention_mask is 0 at the beginning, setting scale to 1 will
            # make the first retention to use the padding incorrectly. Hence,
            # setting it to 0 here. This is a little ugly, so we might want to
            # change this later. TODO: improve
            scale = np.where(retention_mask == 0, np.zeros_like(decay), scale)

        output = np.sum(q * current_kv, dim=3).unsqueeze(1)  # (b, 1, h, d_v)
        return output, {"prev_key_value": current_kv, "scale": scale}

    def theta_shift(x, sin, cos):
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        y = np.stack((-x2, x1), dim=-1).flatten(-2)
        return (x * cos) + (y * sin)

    def forward(
        self,
        hidden_states,
        gctx = {},
        state = None,
        **kwargs
    ): 
        B, T, H = hidden_states.size()

        mode = 'parallel' if T > 1 else 'recurrent'
        if state is not None:
            if mode in state:
                ((cos, sin), (decay_mask, intra_decay, scale)) = state[mode]
                if cos.size(0) != T:
                    ((cos, sin), (decay_mask, intra_decay, scale)) = compute_mask(self, T, self.embed_dim, self.num_heads, mode=mode)
                    state[mode] = ((cos, sin), (decay_mask, intra_decay, scale))
            else:
                ((cos, sin), (decay_mask, intra_decay, scale)) = compute_mask(self, T, self.embed_dim, self.num_heads, mode=mode)
                state[mode] = ((cos, sin), (decay_mask, intra_decay, scale))
        else:
            ((cos, sin), (decay_mask, intra_decay, scale)) = compute_mask(self, T, self.embed_dim, self.num_heads, mode=mode)

        q, k, v, g = [proj(hidden_states) for proj in [self.q_proj, self.k_proj, self.v_proj, self.g_proj]]
        q, k, v = [t.view(B, T, self.num_heads, -1) for t in [q,k,v]]
        k *= self.scaling  # for scaled dot product
        q, k, v = [np.einsum('blnd->bnld', t) for t in [q, k, v]]
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        # -- Cache load --
        kv_cache = None if state is None else state.get('kv_cache', None)

        if T > 1:
            if self.decay_proj is not None:
                decay_mask = self.decay_proj(decay_mask.transpose(-1, -3)).transpose(-3, -1)

            # retention(q,k,v)
            retention = np.einsum('bnld,bnmd->bnlm', q, k)
            retention = retention * decay_mask 
            retention = retention / retention.detach().abs().sum(
                dim=-1, keepdim=True
            ).clamp(min=1, max=5e4)
            retention_out = np.einsum('bnlm,bnmd->blnd', retention, v)

            if self.decay_proj is not None:
                intra_decay = self.decay_proj(intra_decay.transpose(-1, -2)).transpose(
                    -2, -1
                )

            # kv cache: [b, h, t, v_dim, qk_dim]
            current_kv = k.unsqueeze(-2) * v.unsqueeze(-1)
            intra_decay = intra_decay[:, :, :, None, None]  # [b, h, t, 1, 1]
            current_kv = (current_kv * intra_decay).sum(2)  # [b, h, v_dim, qk_dim]
            kv_cache = {"prev_key_value": current_kv, "scale": scale}
        else:
            retention_out, kv_cache = self.recurrent_retention(q,k,v,
                decay_mask,
                kv_cache=kv_cache,
                retention_mask=None #??????
            )

        # -- Cache load --
        if state is not None:
            state['kv_cache'] = kv_cache

        # norm
        normed = self.group_norm(retention_out).reshape(B, T, self.value_dim)
        # out gate & proj
        out = self.gate_fn(g) * normed
        return self.out_proj(out)

    return __init__(nn.Module(forward=forward,recurrent_retention=recurrent_retention), args)

class Args():
    def __init__(self, **kwargs): 
        for key in kwargs: setattr(self, key, kwargs[key])

def RetNet(name):
    import aka.repo as repo

    # -- Tokenizer --
    tokenizer = repo.AutoTokenizer(name)
    cfg = repo.jsonload(name, 'config.json')
    args = Args(
        tokenizer = tokenizer,
        vocab_size = cfg['vocab_size'],
        embedding_scale = True,
        latent_dim = cfg['decoder_embed_dim'],
        lm_head = True,
        prev_norm = 'rms',
        layers = ['Retention', 'MLP']*cfg['decoder_layers'],
        mlp_args = Args(
            kv_size = cfg['decoder_ffn_embed_dim'],
            kv_gate = cfg['use_glu'],
            activation = cfg['activation_fn']    # gelu, rms, layernorm
        ),
        attn_args = Args(
            num_heads = cfg['decoder_retention_heads'],
            hidden_dim = cfg['decoder_value_embed_dim'],
        ),
        bias = False,
        dropout = cfg['dropout']
    )

    # -- Model --
    from CausalLM import CausalLM
    model = CausalLM(args)
    if repo.exists(name, "model.safetensors"):
        with repo.safeopen(name, "model.safetensors") as f:
            keys = f.keys()
            with np.no_grad():
                model.lm_head.weight.copy_(f.get_tensor(f'lm_head.weight'))
                model.embedding.weight.copy_(f.get_tensor('model.embed_tokens.weight'))
                model.post_norm.weight.copy_(f.get_tensor(f'model.layer_norm.weight'))
                model.prev_norm.weight.copy_(f.get_tensor(f'model.layernorm_embedding.weight'))
                for i in range(len(model.layers)//2):
                    model.layers[i*2].norm.weight.copy_(f.get_tensor(f'model.layers.{i}.retention_layer_norm.weight'))
                    model.layers[i*2].layer.q_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.q_proj.weight'))
                    model.layers[i*2].layer.g_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.g_proj.weight'))
                    model.layers[i*2].layer.k_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.k_proj.weight'))
                    model.layers[i*2].layer.v_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.v_proj.weight'))
                    model.layers[i*2].layer.out_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.retention.out_proj.weight'))
                    model.layers[i*2+1].norm.weight.copy_(f.get_tensor(f'model.layers.{i}.final_layer_norm.weight'))

                    # Take care here. gate and fc1 are just swaped.
                    model.layers[i*2+1].layer.gate_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.ffn.fc1.weight'))
                    model.layers[i*2+1].layer.up_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.ffn.gate.weight'))
                    model.layers[i*2+1].layer.down_proj.weight.copy_(f.get_tensor(f'model.layers.{i}.ffn.fc2.weight'))
    return model

if __name__ == "__main__":
    retnet = RetNet('../aka/data/SDPrompt-RetNet-300M')
    print('Model loaded')
    for w in retnet.generator("<s>1girl"):
        print(w, end='')
# <s> 1girl, absurdres, animal ear fluff, animal ears, bangs, bare shoulders, black hair, blue archive, blunt bangs, blush, closed mouth, collarbone, commentary request, eyes visible through hair, green eyes, hair between eyes, halo, hand on own face, hand up, highres, jacket, kisaki blue archive, long hair, long sleeves, looking at viewer, open clothes, open jacket, shinonome asu, simple background, solo, track jacket, upper body, white background, white jacket</s>

