import torch
from openstl.utils import (reduce_tensor, reshape_patch, reshape_patch_back)

ones_tensor = torch.ones([1,1,1,6,6])

a = ones_tensor
b = ones_tensor * 2
c = ones_tensor * 3
d = ones_tensor * 4
print(a,b,c,d)
concatenate = torch.cat([a,b,c,d], dim=2)
print("Concatenate:", concatenate)

concatenate = concatenate.permute(0, 1, 3, 4, 2).contiguous()
concatenate = reshape_patch(concatenate, 6)
# print(concatenate)

concatenate = concatenate.view(1,1,12,3,2,2).transpose(3,4).contiguous()

Mixer = concatenate.view(1,1,4,6,6) #.transpose(4,6) #.permute(0, 1, 4, 2, 3)
print("Mixer:", Mixer)

"""
e = ones_tensor * 5
f = ones_tensor * 6
g = ones_tensor * 7
h = ones_tensor * 8
i = ones_tensor * 9
#print(a,b,c,d)
concatenate = torch.cat([a,b,c,d,e,f,g,h,i], dim=2)
print(concatenate)

concatenate = concatenate.permute(0, 1, 3, 4, 2).contiguous()
concatenate = reshape_patch(concatenate, 6)
print(concatenate)

concatenate = concatenate.view(1,1,18,2,3,3).transpose(3,4).contiguous()

concatenate = concatenate.view(1,1,9,6,6) #.transpose(4,6) #.permute(0, 1, 4, 2, 3)
print(concatenate)
"""