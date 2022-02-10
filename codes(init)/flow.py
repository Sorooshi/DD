import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')
DEVICE



class InvertibleLayer(nn.Module):
    def __init__(self, var_size):
        super(InvertibleLayer, self).__init__()

        self.var_size = var_size

    def f(self, x, y):
        '''
        Implementation of forward pass.

        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition

        Return:
          torch.Tensor of shape [batch_size, var_size], torch.Tensor of shape [batch_size]
        '''
        pass

    def g(self, x, y):
        '''
        Implementation of backward (inverse) pass.

        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition

        Return:
        А  torch.Tensor of shape [batch_size, var_size]
        '''
        pass

    
class NormalizingFlow(nn.Module):
    
    def __init__(self, layers, prior):
        super(NormalizingFlow, self).__init__()

        self.layers = nn.ModuleList(layers)
        self.prior = prior

    def log_prob(self, x, y):
        '''
        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        '''
        log_likelihood = None

        for layer in self.layers:
            x, change = layer.f(x, y)
            if log_likelihood is not None:
                log_likelihood = log_likelihood + change
            else:
                log_likelihood = change
        log_likelihood = log_likelihood + self.prior.log_prob(x)

        return log_likelihood.mean()

    def sample(self, y):
        '''
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        '''
        
        x = self.prior.sample((len(y), ))
        for layer in self.layers[::-1]:
            x = layer.g(x, y)

        return x
    

class RealNVP(InvertibleLayer):
    
    def __init__(self, var_size, cond_size, mask, hidden=10):
        super(RealNVP, self).__init__(var_size=var_size)

        self.mask = mask

        self.nn_t = nn.Sequential(
            nn.Linear(var_size+cond_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, var_size)
            )
        self.nn_s = nn.Sequential(
            nn.Linear(var_size+cond_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, var_size),
            )

    def f(self, x, y):
        '''
        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        '''
        xy = torch.cat((x * self.mask[None, :], y), dim=1)
        t = self.nn_t(xy)
        s = self.nn_s(xy)

        new_x = (x * torch.exp(s) + t) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        log_det = (s * (1 - self.mask[None, :])).sum(dim=-1)
        return new_x, log_det

    def g(self, x, y):
        '''
        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        '''
        xy = torch.cat((x * self.mask[None, :], y), dim=1)
        t = self.nn_t(xy)
        s = self.nn_s(xy)

        new_x = ((x - t) * torch.exp(-s)) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        return new_x


from sklearn.preprocessing import StandardScaler


class NFFitter(object):
    
    def __init__(self, var_size=2, cond_size=2, batch_size=32, n_epochs=10, lr=0.0001):
        
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        
        prior = torch.distributions.MultivariateNormal(torch.zeros(var_size), torch.eye(var_size))

        layers = []
        for i in range(8):
            layers.append(RealNVP(var_size=var_size, cond_size=cond_size, mask=((torch.arange(var_size) + i) % 2)))

        self.nf = NormalizingFlow(layers=layers, prior=prior)
        self.opt = torch.optim.Adam(self.nf.parameters(), lr=self.lr)
    
    
    def fit(self, X, y):
            
        # numpy to tensor
        y_real = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        X_cond = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        
        # tensor to dataset
        dataset_real = TensorDataset(y_real, X_cond)
        
        criterion = nn.MSELoss()
        self.loss_history = []

        # Fit GAN
        for epoch in range(self.n_epochs):
            print("epoch =", epoch)
            for i, (y_batch, x_batch) in enumerate(DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True)):
                
                # caiculate loss
                loss = -self.nf.log_prob(y_batch, x_batch)
                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                    
                # caiculate and store loss
                self.loss_history.append(loss.detach().cpu())

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_pred = self.nf.sample(X).cpu().detach().numpy()
        return y_pred
    
    
    
