import torch
from torch.autograd import Function
from torch_sparse import spspmm

def autograd():

    class SumTensor(Function):

        @staticmethod
        def forward(ctx, a, b):
            return a + b

        @staticmethod
        def backward(ctx, grad_sum):
            return grad_sum, grad_sum

        a = torch.zeros(3, requires_grad=True)

    sum_ab = SumTensor.apply
    a = torch.zeros(3, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)

    sum = sum_ab(a, b).sum()
    sum.backward()


def autograd_chunk():

    def sparse_sum(tensor_list_chunk, indices):
        return torch.sum(tensor_list_chunk[indices], axis=0)

    tensorlist = [
        torch.randn(3, requires_grad=True),
        torch.randn(3, requires_grad=True)
    ]

    optim = torch.optim.SGD(tensorlist, lr=0.01)
    print(tensorlist)
    for i in range(1000):
        optim.zero_grad()

        sum_res = sparse_sum(
            torch.stack(tensorlist).abs(),
            torch.Tensor([1, 0]).long())

        res = sum_res.sum()
        # print(res.item())
        res.backward()
        optim.step()
    print(tensorlist)


def autograd_list_builtin():

    def abs_tensorlist(*args):
        res = [t.abs() for t in args]
        return tuple(res)

    def sum_tensorlist_auto(*args):
        res = torch.zeros_like(args[0])
        for t in args:
            res += t
        return res

    tensorlist = [
        torch.randn(3, requires_grad=True),
        torch.randn(3, requires_grad=True)
    ]

    print(tensorlist)
    optim = torch.optim.SGD(tensorlist, lr=0.001)
    for i in range(1000):
        optim.zero_grad()

        abs_res = abs_tensorlist(*tensorlist)
        sum_res = sum_tensorlist_auto(*abs_res)
        # print(tensorlist, abs_res, sum_res)
        res = sum_res.sum()
        # print(res.item())
        res.backward()
        optim.step()
    print(tensorlist)


def autograd_list_custom():

    def abs_tensorlist(*args):
        res = [t.abs() for t in args]
        return tuple(res)

    class SumTensorList(Function):

        @staticmethod
        def forward(ctx, *args):
            ctx.save_for_backward(*args)

            s = torch.zeros_like(args[0])
            for tensor in args:
                s += tensor
            return s

        @staticmethod
        def backward(ctx, grad_sum):
            saved_tensors = ctx.saved_tensors
            lst = [grad_sum * torch.ones_like(t) for t in saved_tensors]
            return tuple(lst)

    sum_tensorlist = SumTensorList.apply

    tensorlist = [
        torch.randn(3, requires_grad=True),
        torch.randn(3, requires_grad=True)
    ]

    print(tensorlist)
    optim = torch.optim.SGD(tensorlist, lr=0.001)
    for i in range(1000):
        optim.zero_grad()

        abs_res = abs_tensorlist(*tensorlist)
        sum_res = sum_tensorlist(*abs_res)
        # print(tensorlist, abs_res, sum_res)
        res = sum_res.sum()
        # print(res.item())
        res.backward()
        optim.step()
    print(tensorlist)


if __name__ == '__main__':
    autograd_chunk()
    autograd_list_builtin()
    autograd_list_custom()
