import torch
from torch import distributions as td


class Normal:

    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self, eps=None):
        eps = torch.randn_like(self.sigma) if eps is None else eps
        return self.mu + eps * self.sigma

    def sample(self, eps=None):
        return self.rsample(eps)

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu

    @classmethod
    def stack(cls, arr, dim=0):
        mu = torch.stack([x.mu for x in arr], dim=dim)
        logvar = torch.stack([x.logvar for x in arr], dim=dim)
        return cls(mu, logvar)

    @classmethod
    def cat(cls, arr, dim=0):
        mu = torch.cat([x.mu for x in arr], dim=dim)
        logvar = torch.cat([x.logvar for x in arr], dim=dim)
        return cls(mu, logvar)


class Categorical:

    def __init__(self, probs=None, logits=None, temp=0.01):
        super().__init__()
        self.logits = logits
        self.temp = temp
        if probs is not None:
            self.probs = probs
        else:
            assert logits is not None
            self.probs = torch.softmax(logits, dim=-1)
        self.dist = td.OneHotCategorical(self.probs)

    def rsample(self, **kwargs):
        relatex_dist = td.RelaxedOneHotCategorical(self.temp, self.probs)
        return relatex_dist.rsample()

    def sample(self, **kwargs):
        return self.dist.sample()

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            p = Categorical(logits=torch.zeros_like(self.probs))
        kl = td.kl_divergence(self.dist, p.dist)
        return kl

    def mode(self):
        argmax = self.probs.argmax(dim=-1)
        one_hot = torch.zeros_like(self.probs)
        one_hot.scatter_(1, argmax.unsqueeze(1), 1)
        return one_hot


if __name__ == "__main__":
    mu = torch.randn([3, 5])
    logvar = torch.randn([3, 5])
    mu1 = torch.zeros([3, 5])
    logvar1 = torch.zeros([3, 5])
    params = torch.cat([mu, logvar], dim=-1)

    q = Normal(params=params)
    p = Normal(mu1, logvar1)
    q_dist = td.Normal(scale=q.sigma, loc=q.mu)
    p_dist = td.Normal(scale=p.sigma, loc=p.mu)
    kl_tr = td.kl_divergence(q_dist,p_dist)
    kl1 = q.kl(p)
    kl0 = q.kl()
    print(kl1, kl_tr, kl0)

    logits = torch.randn([3, 5])
    logits1 = torch.randn([3, 5])

    q = Categorical(logits=logits, temp=0.5)
    p = Categorical(logits=logits1, temp=0.01)
    q_rsamp = q.rsample()
    p_rsamp = p.rsample()
    kl = q.kl(p)
    q_samp = q.sample()
    mode = q.mode()
    print(kl)