import aka.nn as nn
import aka.numpy as np

def MLPBlock(args):
    '''
    Reference: Gemma, LLaMA
    Common ver:
        (b,l,latent_dim) --up--> (b,l,kv_size, kv_size) --down--> (b, l, latent_dim)
    Full ver:
        (b,l,latent_dim) --in--> (b,l,qk_dim) --up--> (b,l,kv_size, kv_size) 
        --down--> (b, l, hidden_dim) --out--> (b,l,latent_dim)
    Args:
        args.latent_dim,    Required
        args.bias = False,  Optional(False)
        args.mlp_args.kv_size = 384*4, Optional(latent_dim)
        args.mlp_args.kv_gate = False, Optional(False)
        args.mlp_args.qk_dim = 384,  Optional(latetn_dim)
        args.mlp_args.hidden_dim = 384,Optional(latetn_dim)
        args.mlp_args.num_heads = 6,      # not support.
        args.mlp_args.num_kv_groups = 6,  # not support.
    Examples:
        args.mlp_args,kv_gate == True ==> GateMLP
    '''
    def __init__(self, args):
        
        # -- Global Args --
        latent_dim = args.latent_dim
        bias = getattr(args,'bias', False)

        # -- MLP Args
        args = args.mlp_args
        kv_size = getattr(args, 'kv_size', latent_dim)
        kv_gate = getattr(args, 'kv_gate', False)
        qk_dim = getattr(args, 'qk_dim', latent_dim)
        hidden_dim = getattr(args, 'hidden_dim', latent_dim)
        act = getattr(args, 'activation', 'gelu')

        def Swish():
            def forward(self, x):
                return np.silu(x)
            return nn.Module(forward=forward)

        match act:
            case 'gelu':
                act = nn.GELU()
            case 'swish':
                act = Swish()
            case 'layernorm':
                act = nn.LayerNorm(kv_size)
            case _:
                act = None

        self.act = act
        self.in_proj = None if qk_dim == latent_dim else nn.Linear(latent_dim, qk_dim, bias=bias)   # Q
        self.up_proj = nn.Linear(qk_dim, kv_size, bias=bias)                                        # K(reversed)
        self.gate_proj = None if not kv_gate else nn.Linear(qk_dim, kv_size, bias=bias)             # G or mask
        self.down_proj = nn.Linear(kv_size, hidden_dim, bias=bias)                                  # V
        self.out_proj = None if hidden_dim == latent_dim else nn.Linear(hidden_dim, latent_dim, bias=bias)
        return self

    def forward(self, x, **kwargs):
        if self.in_proj is not None:
            x = self.in_proj(x)
        att = self.up_proj(x)
        if(self.gate_proj is not None):
            gate = self.gate_proj(x)
            if self.act is not None:
                gate = self.act(gate)    # silu LLaMA ?
            att = gate * att
        elif self.act is not None:
            att = self.act(att)
        down = self.down_proj(att)
        if self.out_proj is not None:
            down = self.out_proj(down)
        return down
    return __init__(nn.Module(forward=forward), args)
