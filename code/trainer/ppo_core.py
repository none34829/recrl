import torch, math

def clipped_surrogate(obj_adv, ratio, eps=0.2):
    return torch.minimum(obj_adv * ratio,
                         obj_adv * torch.clamp(ratio, 1-eps, 1+eps))

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation (vectorised)."""
    adv = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv
