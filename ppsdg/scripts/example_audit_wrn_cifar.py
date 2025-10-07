import sys
import re
import math
import scipy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import numpy as np
import sklearn.metrics
from wideresnet import WideResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())

# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions)
# v = number of correct guesses by auditor
# eps,delta = DP guarantee of null hypothesis
# output: p-value = probability of >=v correct guesses under null hypothesis
def p_value_DP_audit (m, r, v, eps, delta):
	assert 0 <= v <= r <= m
	assert eps >= 0
	assert 0 <= delta <= 1
	q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
	beta = scipy.stats.binom.sf(v-1, r, q).item() # = P[Binomial(r, q) >= v]
	alpha = 0
	sum = 0 # = P[v > Binomial(r, q) >= v - i]
	for i in range(1, v + 1):
		sum = sum + scipy.stats.binom.pmf(v - i, r, q).item()
		if sum > i * alpha:
			alpha = sum / i
	p = beta + alpha * delta * 2 * m
	return min(p, 1)

# m = number of examples, each included independently with probability 0.5
# r = number of guesses (i.e. excluding abstentions)
# v = number of correct guesses by auditor
# p = 1-confidence e.g. p=0.05 corresponds to 95%
# output: lower bound on eps i.e. algorithm is not (eps,delta)-DP
def get_eps_audit (m, r, v, delta, p):
	assert 0 <= v <= r <= m
	assert 0 <= delta <= 1
	assert 0 < p < 1
	eps_min = 0 # maintain p_value_DP(eps_min) < p
	eps_max = 1 # maintain p_value_DP(eps_max) >= p
	while p_value_DP_audit(m, r, v, eps_max, delta) < p: eps_max = eps_max + 1
	for _ in range(30): # binary search
		eps = (eps_min + eps_max) / 2
		if p_value_DP_audit(m, r, v, eps, delta) < p:
			eps_min = eps
		else:
			eps_max = eps
	return eps_min

def audit (model, ymap, n, kp, km, Qp=trainset, Qm=testset, criterion=nn.CrossEntropyLoss(reduction='none')):
    train_samples = int(torch.distributions.Binomial(n, 0.5).sample().item())
    S = torch.ones(n, dtype=torch.int)
    S[train_samples:] = -1

    scores = torch.tensor([]).to(device)
    for subset_args in (Qp, torch.randperm(len(Qp))[:train_samples]), (Qm, torch.randperm(len(Qm))[:n-train_samples]):
        for X, y in torch.utils.data.DataLoader(torch.utils.data.Subset(*subset_args), batch_size=1000):
            scores = torch.cat([scores, criterion(model(X.to(device)), ymap[y].to(device))])
    sorted_ixs = scores.argsort()

    T = torch.zeros(n, dtype=torch.int)
    T[sorted_ixs[:kp]] = 1
    T[sorted_ixs[-km:]] = -1

    return get_eps_audit(n, kp+km, torch.clamp(S*T, 0).sum().item(), 1e-5, 0.5), (S*T+1).bincount().tolist()

def load_model (path):
    ymap = torch.arange(10)
    m = re.search(r'-to(\d{10})-', path)
    if m:
        ymap = torch.tensor([int(x) for x in m.group(1)])
    classes = ymap.max().item()+1
    print(path, classes, ymap.tolist())

    model = WideResNet(classes, 1).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()

    return model, classes, ymap

def audit_model (path):
    model, classes, ymap = load_model(path)

    with torch.no_grad():
        confusion = np.zeros((classes, classes), dtype=int)
        for X, y in torch.utils.data.DataLoader(testset, batch_size=1000):
            ypred = model(X.to(device))
            confusion += sklearn.metrics.confusion_matrix(ymap[y], ypred.argmax(1).cpu())
        print(confusion, np.diag(confusion).sum()/len(testset))

        for n in 500, 1000, 2000, 10000:
            for k in n//50, n//20, n//10, n//5:#, n//2:
                print(f"Full set ({n}/{k})", *audit(model, ymap, n, k, k))

        for label in range(torch.tensor(trainset.targets).max()+1):
            Qp = torch.utils.data.Subset(trainset, torch.where(torch.tensor(trainset.targets) == label)[0])
            Qm = torch.utils.data.Subset(testset, torch.where(torch.tensor(testset.targets) == label)[0])
            print(f'Original label {label} (-> {ymap[label].item()}):', *audit(model, ymap, 1000, 50, 50, Qp, Qm))

        for clas in range(classes):
            Qp = torch.utils.data.Subset(trainset, torch.where(ymap[torch.tensor(trainset.targets)] == clas)[0])
            Qm = torch.utils.data.Subset(testset, torch.where(ymap[torch.tensor(testset.targets)] == clas)[0])
            print(f'Mapped class {clas}:', *audit(model, ymap, 1000, 50, 50, Qp, Qm))

def compute_losses (path):
    model, classes, ymap = load_model(path)
    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        train_loss = []
        train_y = []
        train_correct = []
        test_loss = []
        test_y = []
        test_correct = []
        for X, y in torch.utils.data.DataLoader(testset, batch_size=1000):
            ypred = model(X.to(device))
            yout = ymap[y].to(device)
            test_loss.append(criterion(ypred, yout).cpu())
            test_correct.append((ypred.argmax(1) == yout).cpu())
            test_y.append(y)
        for X, y in torch.utils.data.DataLoader(trainset, batch_size=1000):
            ypred = model(X.to(device))
            yout = ymap[y].to(device)
            train_loss.append(criterion(ypred, yout).cpu())
            train_correct.append((ypred.argmax(1) == yout).cpu())
            train_y.append(y)
        return *[torch.cat(el) for el in (train_loss, train_correct, train_y, test_loss, test_correct, test_y)], ymap

def plot_losses (train_loss, train_correct, train_y, test_loss, test_correct, test_y, ymap=None, xlims=(3, 10, 10, 3), setylim=True):
    import seaborn
    plt.figure(figsize=(12, 8))

    plt.title("Loss KDE plots")
    plt.axis(False)
    plt.subplot(2, 2, 1)
    clip = (0, xlims[0] or 3)
    plt.xlim(clip)
    seaborn.kdeplot(train_loss, clip=clip, label="Train losses")
    seaborn.kdeplot(test_loss, clip=clip, label="Test losses")
    seaborn.kdeplot(train_loss[train_correct], clip=clip, label="Right-class train losses")
    seaborn.kdeplot(test_loss[test_correct], clip=clip, label="Right-class test losses")
    plt.legend()

    plt.subplot(2, 2, 2)
    clip = (0, xlims[1] or 10)
    plt.xlim(clip)
    seaborn.kdeplot(train_loss[~train_correct], clip=clip, label="Wrong-class train losses")
    seaborn.kdeplot(test_loss[~test_correct], clip=clip, label="Wrong-class test losses")
    if setylim:
        plt.ylim(plt.ylim())
    seaborn.kdeplot(train_loss[train_correct], clip=clip, label="Right-class train losses")
    seaborn.kdeplot(test_loss[test_correct], clip=clip, label="Right-class test losses")
    plt.legend()

    plt.subplot(2, 2, 3)
    clip = (0, xlims[2] or 1)
    plt.xlim(clip)
    """
    for label in range(train_y.max()+1):
        seaborn.kdeplot(train_loss[train_y == label], clip=clip, label=f"Label {label} train losses", color='tab:blue')
        seaborn.kdeplot(test_loss[test_y == label], clip=clip, label=f"Label {label} test losses", color='tab:orange')
    """
    for label in range(ymap.max()+1):
        seaborn.kdeplot(train_loss[(ymap[train_y] == label) & ~train_correct], clip=clip, label=f"Label {label} train wrong losses")
        seaborn.kdeplot(test_loss[(ymap[test_y] == label) & ~test_correct], clip=clip, label=f"Label {label} test wrong losses")
    if setylim:
        plt.ylim(plt.ylim())
    for label in range(ymap.max()+1):
        seaborn.kdeplot(train_loss[(ymap[train_y] == label) & train_correct], clip=clip, label=f"Label {label} train right losses")
        seaborn.kdeplot(test_loss[(ymap[test_y] == label) & test_correct], clip=clip, label=f"Label {label} test right losses")
    if ymap.max() < 3:
        plt.legend()
        pass

    plt.subplot(2, 2, 4)
    clip = (0, xlims[3] or 3)
    plt.xlim(clip)
    for label in range(ymap.max()+1):
        seaborn.kdeplot(train_loss[ymap[train_y] == label], clip=clip, label=f"Label {label} train losses")
        seaborn.kdeplot(test_loss[ymap[test_y] == label], clip=clip, label=f"Label {label} test losses")
    if ymap.max() < 6:
        plt.legend()

    plt.show()

if __name__ == "__main__":
    for path in sys.argv[1:]:
        audit_model(path)
