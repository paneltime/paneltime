import numpy
import ctypes as ct

data=[3*[22.0] for i in range(0,5)]
data=numpy.array(data)
data[0][0]=5.5
data[1][0]=7.7
data[2][0]=8.7

data=numpy.ascontiguousarray(data)
cont=data.flags['C_CONTIGUOUS']
#sys.path.append("src/")

import cfunctions as c
aaa=c.GetMoments(data)
print (data)