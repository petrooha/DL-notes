x = torch.randn(2,2, requires_grad=True)
print(x)

y = x**2
print(y)

## grad_fn shows the function that generated this variable
print(y.grad_fn)

z = y.mean()
print(z)

print(x.grad)
z.backward()
print(x.grad)
print(x/2)
