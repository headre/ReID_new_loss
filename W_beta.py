import torch

#Em = WX-Y
#W needs utilization
#beta is a smoothing paramter, and it's range in (0,1)
def get_W_m(Em,beta):
    k = Em.size()[0]
    Wm = torch.zeros(k,k)
    for i in range(0,k):
        W=0
        for t in range(0,k):
            if(Em[t]==0 or beta==1/2):
                W=1
            else:
                W+=(Em[i]/Em[t])**(1/(2*beta-1))
        Wm[i,i] = 1/W
    return Wm
