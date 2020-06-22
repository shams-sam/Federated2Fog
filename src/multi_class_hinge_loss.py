import torch
import torch.nn as nn


class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight
        # weight for each class, size=n_class,
        # variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average

    def forward(self, output, y):#output: batchsize*n_class
        output_y=output[
            torch.arange(0, y.size()[0]).long().cuda(),y.data.cuda()].view(
                -1, 1)
        loss=output-output_y+self.margin#contains i=y
        # remove i=y items
        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
        # max(0,_)
        # loss[loss<0]=0 # previous version
        loss = nn.functional.relu(loss)
        # ^p
        if(self.p!=1):
            loss=torch.pow(loss, self.p)
        #add weight
        if(self.weight is not None):
            loss = loss*self.weight
        #sum up
        loss=torch.sum(loss)
        if(self.size_average):
            loss /= output.size()[0]#output.size()[0]
        return loss
