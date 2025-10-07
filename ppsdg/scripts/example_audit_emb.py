from wideresnet import WideResNet
import torch
import torch.nn as nn
import sys
import torchvision
import re
import datetime

class WRNEmb (WideResNet):
    def __init__ (self, path="", k=1):
        #print(path[path.index("-to")+3:].split("-")[0])
        self.ymap = torch.tensor([int(x) for x in re.search(r'-to(\d+)-', path).group(1)])
        self.num_classes = self.ymap.max().item()+1
        super(WRNEmb, self).__init__(self.num_classes, k)
        self.load_state_dict(torch.load(path, weights_only=True))
        self.orig_fc = self.fc
        self.fc = nn.Identity()

    def preprocess (self, dataset):
        """Permute and generate embedding * grad target for each row."""
        loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss(reduction='none')
        databatches = []
        for X, y in loader:
            X = X.to(device)
            yout = self.ymap[y].to(device)
            emb = self(X.to(device)).requires_grad_(True)
            grad, = torch.autograd.grad(criterion(self.orig_fc(emb), yout), emb)
            databatches.append(torch.hstack([X.flatten(1),
                                             emb,
                                             # XXX these shouldn't be here
                                             nn.functional.one_hot(yout, self.num_classes),
                                             self.orig_fc(emb),
                                             criterion2(self.orig_fc(emb), yout)[:, None],
                                             #
                                             grad.norm(dim=1)[:, None], grad,
                                             ]).cpu().detach())
        return torch.cat(databatches)

class EmbeddingAuditor (nn.Module):
    def __init__ (self, w):
        super(EmbeddingAuditor, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.ReLU(),
            nn.Softmax(1))

    def forward (self, x):
        return self.model(x)

def load_preproc (path):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=torchvision.transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())
    pppath = path + "_pprl"
    try:
        X_train, targets, X_test, targets, weights = torch.load(pppath, weights_only=True)
    except FileNotFoundError as e:
        torch.manual_seed(0)
        premodel = WRNEmb(path).to(device)

        traindata = premodel.preprocess(trainset)
        testdata = premodel.preprocess(testset)

        X_train = torch.cat([testdata[len(testset)//2:], traindata[len(trainset)//2:]])
        X_test = torch.cat([testdata[:len(testset)//2], traindata[:len(trainset)//2]])
        targets = torch.zeros(len(X_train), dtype=int)
        targets[:len(testset)//2] = 1
        weights = 1 + targets * (len(trainset) / len(testset) - 1)

        torch.save((X_train, targets, X_test, targets, weights), pppath)

    trainset.data = X_train.numpy()
    trainset.targets = targets
    
    testset.data = X_test.numpy()
    testset.targets = targets

    bs = targets.sum().item()
    sampler = torch.utils.data.WeightedRandomSampler(weights, bs, replacement=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, sampler=sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, sampler=sampler)
    return trainloader, testloader

def audit_model (path, suffix, steps=1000, lr=0.1, w=64*2):
    try:
        return torch.load(path + suffix, weights_only=False)
    except FileNotFoundError as e:
        pass

    trainloader, testloader = load_preproc(path)

    if type(w) == int:
        model = EmbeddingAuditor(w).to(device)
        w = torch.arange(w)
    else:
        model = EmbeddingAuditor(len(w)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    graph_data = []


    for step in range(steps):
        rec = [step]
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for i, (X, y) in enumerate(trainloader):
            X, y = X.flatten(1)[:, w].to(device), y.to(device)
            optimizer.zero_grad()
            ypred = model(X)
            loss = criterion(ypred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (ypred.argmax(1) == y).sum().item()
            total += len(y)
        rec += (total_loss/(i+1), correct, total)
        print(step, datetime.datetime.now(), f'Train set loss: {total_loss/(i+1)} , accuracy: {correct/total}', flush=True)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            total = 0
            for i, (X, y) in enumerate(testloader):
                X, y = X.flatten(1)[:, w].to(device), y.to(device)
                ypred = model(X)
                total_loss += criterion(ypred, y).item()
                correct += (ypred.argmax(1) == y).sum().item()
                total += len(y)
            print(step, datetime.datetime.now(), f'Test set loss: {total_loss/(i+1)} , accuracy: {correct/total}', flush=True)
            #if (step+1) % 100 == 0:
            #    eps_est = get_eps_audit(total, total, correct, 1e-5, 0.5)
            #    print(step, datetime.datetime.now(), f'Implied eps estimate(?): {eps_est}')
            rec += (total_loss/(i+1), correct, total)

        graph_data.append(rec)
    torch.save([model.state_dict(), graph_data], path + suffix)
    return torch.load(path + suffix)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for path in sys.argv[1:]:
    #audit_model(path, "_audit-elcn", w=torch.cat([torch.arange(3072, 3072+64+10+10+2)]))
    #audit_model(path, "_audit-lossonly", w=torch.tensor([-66])) #3072+64+10*2, loss
    #audit_model(path, "_audit-en", w=torch.cat([torch.arange(3072, 3072+64), torch.tensor([-65])]))
    #audit_model(path, "_audit-eng", w=torch.cat([torch.arange(3072, 3072+64), torch.arange(-65, 0)]))
    #audit_model(path, "_audit-ecn", w=torch.cat([torch.arange(3072, 3072+64+10+10+2)]))
    audit_model(path, "_audit-ecn2", w=torch.cat([torch.arange(3072, 3072+64+10), torch.tensor([-65])]))
    audit_model(path, "_audit-ecl", w=torch.cat([torch.arange(3072, 3072+64+10), torch.tensor([-66])]))
    audit_model(path, "_audit-eccl", w=torch.cat([torch.arange(3072, 3072+64+10+10), torch.tensor([-66])]))
    audit_model(path, "_audit-eccn", w=torch.cat([torch.arange(3072, 3072+64+10+10), torch.tensor([-65])]))
