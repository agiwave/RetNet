import aka.nn as nn
import aka.numpy as np

def XprojBlock(**kwargs):
    '''
    FFN/MLP/Hawk/SSM/Attention/.....
    Args:
        latent_dim:       (required)
        hidden_dim:       latent_dim (default)
        xproj_heads:        1 (default)
    Example:
        1, FFN/MLP: XprojBlock(xproj_heads = 1, gate='gh', latent_dim, hidden_dim)
        2, Attention: 
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        bias = getattr(args, 'bias', False)
        dropout = getattr(args, 'dropout', 0.2)

        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.xproj_heads = getattr(args, 'xproj_heads', 1)
        self.xproj_swapd = getattr(args, 'xproj_swapd', True)
        self.kv_dims = getattr(args, 'kv_dims', [self.hidden_dim])
        match getattr(args, 'gate', None):
            case 'gh':
                (self.hg_dim, self.og_dim) = (self.hidden_dim, 0)
            case 'go':
                (self.hg_dim, self.og_dim) = (0, args.latent_dim)
            case _:
                (self.hg_dim, self.og_dim) = (0, 0)

        act = getattr(args, 'activation', None)
        self.act = None if act is None else getattr(np, act, None)
        assert args.latent_dim % self.xproj_heads == 0
        assert self.hidden_dim % self.xproj_heads == 0

        # ik, vk, v, hg, og
        k_sum_dim = 0
        for k_dim in self.kv_dims:
            k_sum_dim += k_dim
            assert k_dim % self.xproj_heads == 0
        self.in_proj = nn.Parameter(shape=(
            self.xproj_heads,
            (k_sum_dim + self.hg_dim + self.og_dim)//self.xproj_heads,
            args.latent_dim//self.xproj_heads)
        )

        # mixers
        def createMixer(name, **kwargs):
            import importlib
            module = importlib.import_module(name)
            short_name = name.split('./\\')[-1]
            m = getattr(module, short_name+"Block", None)
            assert m is not None, f"Unknown layer:{name}"
            return m(**kwargs)
        mixers = getattr(args, 'mixers', None)
        if mixers is not None:
            self.mixers = nn.ModuleList([createMixer(**dict(
                kwargs,
                xproj = False,
                latent_dim = self.hidden_dim,
                **mixerArgs
            )) for mixerArgs in mixers])
        else:
            self.mixers = None

        # o
        self.out_proj = nn.Parameter(shape=(
            self.xproj_heads,
            args.latent_dim // self.xproj_heads,
            self.hidden_dim // self.xproj_heads
        ))
        self.dropout = nn.Dropout(dropout)
        return self

    def copy_xproj_weights(self, in_projs, out_proj):
        in_proj = np.cat(in_projs, dim=0)
        in_proj = np.rearrange('(h d) k->h d k', in_proj, h=self.xproj_heads)

        out_proj = np.rearrange('(h d) k->h d k', out_proj, h=self.xproj_heads)
        self.in_proj.copy_(in_proj)
        self.out_proj.copy_(out_proj)

    def proj_in(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        # Inproj
        x = x.view(b, l, self.xproj_heads, -1)
        xprojs = np.einsum('b l h d, h v d->b l h v', x, self.in_proj)
        split_dims = self.kv_dims + [self.hg_dim, self.og_dim]
        splits = xprojs.split([dim//self.xproj_heads for dim in split_dims], dim=-1)
        splits = [np.rearrange('b l h d->b l (h d)', item) for item in splits]
        (x, hg, og) = splits[-3], splits[-2], splits[-1]
        kv = splits[:-3]

        # mixers
        if self.mixers is not None:
            if state is not None:
                mixer_states = state.get('mixer_state', None)
                if mixer_states is None:
                    mixer_states = [{} for _ in self.mixers]
                    state['mixer_state'] = mixer_states
            for i, mixer in enumerate(self.mixers):
                x = mixer(x, kv=kv, state=None if state is None else mixer_states[i])
                
        return (kv, x, (hg, og, (b,l,d)))

    def proj_out(self, x, g, **kwargs):
        (hg, og, shape) = g
        (b, l, _) = shape
        x = x if self.dropout is None else self.dropout(x)
        if self.hg_dim == 0:
            x = x if self.act is None else self.act(x)
        else:
            x = self.act(hg) * x
        if self.xproj_swapd:
            x = x.view(b, l, -1, self.xproj_heads)    # mix heads
            x = np.einsum('b l v h , h d v -> b l h d', x, self.out_proj)
        else:
            x = x.view(b, l, self.xproj_heads, -1)
            x = np.einsum('b l h v , h d v -> b l h d', x, self.out_proj)
        x = np.reshape(x, shape)
        return x if self.og_dim == 0 else self.act(og) * x

    def forward(self, x, state=None, **kwargs):
        (_, x, g) = self.proj_in(x)
        return self.proj_out(x, g)
    return __init__(nn.Module(forward = forward, proj_in=proj_in, proj_out=proj_out, copy_xproj_weights=copy_xproj_weights),**kwargs)

if __name__ == "__main__":
    def XprojArgs(name):
        layer = dict(
            mixers = [
                dict(
                    name = 'Conv1d'
                )
            ],
            conv_kernel_size = 4
        )
        args = dict(
            name = name,
            vocab_dim = 64,
            latent_dim = 512,
            resident_gate = True,
            prev_norm = 'rms',
            dropout = 0.1,
            bias = False # bias in Linear?
        )
        match(name):
            case 'Full':
                return dict(
                    args,
                    latent_dim = 512,
                    xproj_heads = 2,
                    layers = [
                        dict(
                            name='Attention',
                            rotary_embedding = True,
                            num_heads = 16
                        ),
                        dict(
                            name='Xproj',
                            hidden_dim = 512 * 3
                        )
                    ]*8,
                )
                return args
            case 'Tiny':
                return dict(
                    args,
                    latent_dim = 1024,
                    xproj_heads = 8,
                    layers = [
                        dict(
                            name='Attention',
                            rotary_embedding = True,
                            num_heads = 16
                        ),
                        dict(
                            name='Xproj',
                            hidden_dim = 1024 * 3
                        )
                    ]*8,
                )
                return args
            case _:
                assert False

    from RomeArena import TrainRoles, RunRoles
    roles = [
        XprojArgs('Full'),
        XprojArgs('Tiny')
    ]
    TrainRoles(roles, lr=6e-3, epochs=1)
    # RunRoles(roles, 'My lord Sebastian')
