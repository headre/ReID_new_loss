import torch
import torch.nn as nn


class new_loss(nn.Module):
    def __init__(self) -> None:
        super(new_loss, self).__init__()
        return
    def get_single_W(self,X,Y_row,W_row,p,id):
        numerator=0
        denominator=0
        for i in range(p):
            X_row = X[i]
            denominator+=torch.matmul(X_row,X_row.reshape(-1,1))
        denominator = denominator*W_row*W_row+1
        numerator = W_row*W_row*torch.matmul(Y_row,X[id].reshape(-1,1))
        return numerator/denominator

    def get_row_W(self,X,Y_row,W_row,p):
        return [self.get_single_W(X,Y_row,W_row,p,i) for i in range(p)]

    def get_W_matrix(self,X,Y,W_beta):
        W = W_beta.sum(1)
        size_W = X.size()
        width_W = max(size_W[0],size_W[1])
        print(width_W)
        res = torch.tensor([self.get_row_W(X,Y[i],W[i],width_W) for i in range(width_W)])
        return res
    def forward(self,target,source,W_beta):
        W_matrix = self.get_W_matrix(target,source,W_beta)
        primary_loss = torch.mm(W_matrix,target)-source
        loss = torch.pow(torch.mm(W_beta,primary_loss),2).sum().sqrt()
        return loss,W_matrix

X = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
Y = torch.Tensor([[0,1,2],[3,4,5],[6,7,8]])
W_b = torch.Tensor([0.3,0.3,0.4])
W_z = torch.Tensor([[0.3,0,0],[0,0.3,0],[0,0,0.4]])
# print(W_z.sum(1))
# W = get_W_matrix(X,Y,W_z)
# loss = torch.mm(W,X)-Y
# print(W)
# print(X-Y)
# print(loss)
# print(torch.pow(torch.mm(W_z,loss),2).sum().sqrt())
new = new_loss()
loss,matr = new(X,Y,W_z)
print(loss)
print(matr)

#print(get_W_matrix(X,Y,W_b))