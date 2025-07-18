import numpy as np
from timeit import default_timer as  timer
a=np.random.randn(1000)
b=np.random.randn(1000)
A=list(a)
B=list(b)
T=1000
def dot1():
    dot=0
    for i in range(len(A)):

        dot+=A[i]*B[i]
    return dot

def dot2():
    return np.dot(a,b)

start=timer()
for t in range(T):
    dot1()
end=timer()
t1=end-start

start=timer()
for t in range(T):
    dot2()
end=timer()
t2=end-start

print('List calculation', t1)
print('np.dot', t2)
print('ratio',t1/t2)

x=np.array([[1,2],[4,5]])
print(x)
c=np.array(T)
print(c)
m=np.random.random((3,2))
print(m)
n=np.random.randn(1000)
print(n.mean(),n.var())