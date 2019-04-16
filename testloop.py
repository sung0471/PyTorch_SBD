temporal_length=16
image_clip=[]
for i in range(temporal_length -len(image_clip)):
    print(temporal_length -len(image_clip))
    image_clip.append(i)

print(image_clip)

arr, arr2 = list(), list()
for i in range(10):
    arr.append(i)
    arr2.append(10-i)

print(arr)
print(arr[1:-2])

if type(arr) is list:
    print("arr : list")
else:
    print("arr : int")

print("arr[{}:] = {}\n".format(1, arr[1:]))
start_index = 0
for i, _ in enumerate(arr, 2):
    print("arr[{}::{}] = {}".format(start_index,i,arr[start_index::i]))

flag=False
for _ in range(10):
    print((1,3)[flag])
    flag=not flag

tup=(arr,arr2)
for i, arr in enumerate(tup):
    print(tup[i])

tup2=(1,2,3,4,5,6,7,8,9,10)
print(tup2[1::2])

import torch
a=torch.randn(4)
print("a = ", a)
b=a.mul(0.9)           # a에 0.9를 mul
print("b = .a.mul(0.9) = ", b)
print("b.add(1) = ", b.add(1))        # b에 1를 add
print("b.add(0.1,1) = ", b.add(0.1,1))    # 0.1*1을 b에 add
print("a.mul(0.9).add(0.1,1) = ", a.mul(0.9).add(0.1,1))

for i in range(4,6,2):
    print(i)

line='5014499309.mp4 1868 2'
print(line.split(' '))