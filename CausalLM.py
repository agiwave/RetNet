import aka.nn as nn
import aka.numpy as np

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

def MetaLayer(name, args):
    '''
    Build resident meta layer by name. Include: GQA(Group-Query Attention), MLP, GateMLP, ...
    '''
    def forward(self, x, **kwargs):
        y = self.norm(x)
        y = self.layer(y, **kwargs)
        if isinstance(y, tuple):
            y, loss = y
            return x+y, loss
        else:
            return x+y

    import importlib
    module = importlib.import_module(name)
    short_name = name.split('./\\')[-1]
    m = getattr(module, short_name+"Block", None)
    assert m is not None, f"Unknown layer:{name}"
    return nn.Module(forward = forward,norm = RMSNorm(args.latent_dim),layer = m(args))

import math
def CausalLM(args):
    '''
    Causal Language Model.
    '''

    def __init__(self, args):
        in_proj, out_proj = None, None
        vocab_dim = getattr(args, 'vocab_dim', args.latent_dim)
        if vocab_dim != args.latent_dim:
            in_proj = nn.Linear(vocab_dim, args.latent_dim, bias=args.bias)
            out_proj = nn.Linear(args.latent_dim, vocab_dim, bias=args.bias)

        pad_x = getattr(args, 'pad_x', False)
        lm_head = getattr(args, 'lm_head', False)
        make_layer = MetaLayer if not hasattr(args, 'MetaLayer') else args.MetaLayer

        prev_norm = getattr(args, 'prev_norm', None)
        if prev_norm is not None:
            match prev_norm:
                case 'gemma':
                    from Gemma import GemmaEmbNorm
                    prev_norm = GemmaEmbNorm()
                case _:
                    prev_norm = RMSNorm(args.latent_dim)

        embedding_scale = getattr(args,'scale_embedding',False)
        self.tokenizer = args.tokenizer
        self.vocab_dim = vocab_dim
        self.latent_dim = args.latent_dim
        self.pad_x = pad_x
        self.embedding_scale = (None if embedding_scale else math.sqrt(vocab_dim))
        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=vocab_dim)
        self.layers = nn.ModuleList([make_layer(key, args) for key in args.layers])
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.lm_head = None if not lm_head else nn.Linear(vocab_dim, args.vocab_size,bias=False)
        self.prev_norm = prev_norm
        self.post_norm = RMSNorm(args.latent_dim)
        self.gctx = {}
        return self

    def forward(self, inputs, targets=None, state=None):
        # -- Embedding and layers
        x = self.embedding(inputs)

        # -- vocab_dim --> latent_dim
        if self.vocab_dim != self.latent_dim:
            if self.pad_x:
                x = np.pad(x, (self.latent_dim-self.vocab_dim,0), mode='constant', value=float(0.0))
            else:
                x = self.in_proj(x)
        if self.embedding_scale is not None:
            x = x * self.embedding_scale

        # -- layers --
        if self.prev_norm is not None:
            x = self.prev_norm(x)
        gctx = self.gctx
        if(state is not None):
            layer_states = state.get('layer_states', None)
            if layer_states is None:
                layer_states = [{} for _ in self.layers]
                state['layer_states'] = layer_states

        layer_losses = []
        for i in range(len(self.layers)):
            l_state = None if state is None else layer_states[i]
            x = self.layers[i](x, targets=targets, gctx=gctx, state=l_state)
            if isinstance(x, tuple):
                (x, loss) = x
                if loss is not None:
                    layer_losses.append(loss)

        if self.post_norm is not None:
            x = self.post_norm(x)

        # -- latent_dim --> vocab_dim
        if self.vocab_dim != self.latent_dim:
            if self.pad_x:
                x = np.pad(x, (self.vocab_dim-self.latent_dim,0), mode='constant', value=float(0.0))
            else:
                x = self.out_proj(x)

        # -- vocab_dim --> logits
        if self.lm_head is not None:
            y = self.lm_head(x)    # -- LLaMA vs embedding.weight ? --
        else:
            y = np.einsum('bld,nd->bln', x, self.embedding.weight)

        # -- logits --> output
        if(targets is not None):
            loss = np.cross_entropy(y.view(-1, y.size(-1)), targets.reshape(-1), ignore_index=-1)
            vocab_max = np.max(self.embedding.weight, dim=1)[0]-1.
            vocab_min = np.min(self.embedding.weight, dim=1)[0]
            loss += np.mean(vocab_max**2)+np.mean(vocab_min**2)
            if len(layer_losses) > 0:
                loss += np.sum(np.stack(layer_losses, dim=-1)) / len(layer_losses)
            return y, loss
        else:
            return y

    def generator(self, prompts: str, max_length : int = 64):
        prompt_tokens = self.tokenizer.encode(prompts) # [self.tokenizer.bos_token_id]+
        if hasattr(self, 'eval'):
            self.eval()

        with np.no_grad():
            state = {}
            cache = []
            input_token_ids = np.array([prompt_tokens])
            for _ in range(max_length):
                outputs = self(input_token_ids, state=state)
                input_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)
                cache = cache + input_token_ids[0].tolist()
                if self.tokenizer.eos_token_id in input_token_ids:
                    break

                word = self.tokenizer.decode(cache)
                word_token_ids = self.tokenizer.encode(word)
                if cache[-1] == word_token_ids[-1]:
                    cache = []
                    yield word

            if len(cache)>0:
                yield self.tokenizer.decode(cache)

    def generate(self, prompts : str, max_length : int = 64):
        response = ''
        for w in self.generator(prompts,max_length):
            response += w
        return response
        
    return __init__(nn.Module(forward = forward, generate = generate, generator=generator),args)
