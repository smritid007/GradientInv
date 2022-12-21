import time

DEFAULT_CONFIG = dict(wt='equal',
                      lr=0.1,
                      optimizer='adam',
                      lr_decay=True,
                      signal=False,
                      cost_fn='cosine_sim',
                      indices='def',
                      restarts=1,
                      iter=4800,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      boxed=True,
                      scoring_choice='loss')
class recovering_info():
    """Implementing the algorithm to recover data."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, data_points=1):
        self.data_points = data_points
        self.config = config
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        self.mean_std = mean_std
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def data_initialization(self, data_shape):
        """for initializing the possible recovered image"""
        if self.config['init'] == 'randn':
            return torch.randn((self.data_points, *data_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.data_points, *data_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.data_points, *data_shape), **self.setup)
        else:
            raise ValueError()
    
    def reconstruct(self, data_shape, input_data, labels, eval=True):
        """Reconstruct from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()

        cost_val = torch.zeros(1)
        x = self.data_initialization(data_shape)
        
        assert labels.shape[0] == self.data_points

        try:
            x_attempt = self.trial(x, input_data, labels)
            if self.config['scoring_choice'] == 'loss':
              self.model.zero_grad()
              x_attempt.grad = None
              loss = self.loss_fn(self.model(x_attempt), labels)
              print("after loss")
              gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
              cost_val = self.proposed_cost([gradient], input_gradient, cost_fn=self.config['cost_fn'], indices=self.config['indices'], weights=self.config['wt']) + self.config['total_variation']*total_variation(x_attempt)
            x = x_attempt
        except:
            print('Error occured.')
            pass

        score = torch.isfinite(cost_val) 
        print(f'result score: {score}')
        print(f'Total time: {time.time()-start_time}.')
        return x.detach()
    def proposed_cost(self, gradients, input_gradient, cost_fn='biultin_cosine_sim', indices='def', weights='equal'):

        if indices == 'def':
            indices = torch.arange(len(input_gradient))
        elif indices == 'batch':
            indices = torch.randperm(len(input_gradient))[:16]
        else:
            raise ValueError()

        
        if weights == 'linear':
            weights = torch.arange(len(input_gradient), 0, -1, dtype=input_gradient[0].dtype, device=input_gradient[0].device) / len(input_gradient)
        elif weights == 'exp':
            weights = torch.arange(len(input_gradient), 0, -1, dtype=input_gradient[0].dtype, device=input_gradient[0].device)
            weights = weights.softmax(dim=0)
            weights = weights / weights[0]
        else:
            weights = input_gradient[0].new_ones(len(input_gradient))

        total_costs = 0
        for gradient in gradients:
            pnorm = [0, 0]
            costs = 0
            for i in indices:

                if cost_fn == 'cosine_sim':
                    costs += (gradient[i] * input_gradient[i]).sum()
                    pnorm[0] += gradient[i].pow(2).sum() 
                    pnorm[1] += input_gradient[i].pow(2).sum()
                elif cost_fn == 'builtin_cosine_sim':
                    costs += 1 - torch.nn.functional.cosine_similarity(gradient[i].flatten(), input_gradient[i].flatten(),
                                                                      0, 1e-10) 
            if cost_fn == 'cosine_sim':
                costs = 1 - costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        return costs / len(gradients)

    
    def trial(self, x_attempt, input_data, labels):
        x_attempt.requires_grad = True
        if self.config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam([x_attempt], lr=self.config['lr'])
        elif self.config['optimizer'] == 'sgd':  
            optimizer = torch.optim.SGD([x_attempt], lr=0.01, momentum=0.9, nesterov=True)
        else:
            raise ValueError()
        iters = self.config['iter']
        dm, ds = self.mean_std

        if self.config['lr_decay']:
            #keeping the milestones same as the original implementation to check the changes
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[iters // 2.667, iters // 1.6,iters // 1.142], gamma=0.1)   # 3/8 5/8 7/8

        try:
            for iteration in range(iters):
                closure = self.step_closure(optimizer, x_attempt, input_data, labels)
                reconstruction_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()
                with torch.no_grad():
                    
                    if self.config['boxed']:
                        x_attempt.data = torch.max(torch.min(x_attempt, (1 - dm) / ds), -dm / ds)
                    if (iteration + 1 == iters) or iteration % 100 == 0:
                        print(f'It: {iteration}. Rec. loss: {reconstruction_loss.item():2.4f}.')
                    if (iteration + 1) % 100 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'avg':
                            x_attempt.data = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_attempt)
                        else:
                            raise ValueError()
        except:
          print("error ")
          pass
        return x_attempt.detach()

    def step_closure(self, optimizer, x_attempt, input_gradient, label):
        ##optimizer step needs closure to be callable
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            loss= self.loss_fn(self.model(x_attempt), label)
            gradient_attempt = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            
            reconstruction_loss = self.proposed_cost([gradient_attempt], input_gradient, cost_fn=self.config['cost_fn'], indices=self.config['indices'], weights=self.config['wt'])
            if self.config['total_variation'] > 0:
                reconstruction_loss += self.config['total_variation'] * total_variation(x_attempt)
            reconstruction_loss.backward()
            if self.config['signal']:
                x_attempt.grad.sign_()
            return reconstruction_loss
        return closure
