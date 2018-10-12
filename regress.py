#lighting the hearth
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot
import numpy

#import data
data = numpy.loadtxt("data/data.txt")
x = data[:,0]
y = data[:,1]
z = numpy.zeros(numpy.size(x))
n = 5


#regress
p = numpy.polyfit(x,y,n)

for i in range(0,n+1):
	
	z[:] = z[:] + p[i]*x**(n-i)

#minimum
m = z[numpy.argmin(z)]

#curvature
a=((z[numpy.argmin(z)]-z[numpy.argmin(z)-1])**2+(x[numpy.argmin(z)]-x[numpy.argmin(z)-1])**2)**(1/2)
b=((z[numpy.argmin(z)-1]-z[numpy.argmin(z)+2])**2+(x[numpy.argmin(z)-1]-x[numpy.argmin(z)+2])**2)*(1/2)
c=((z[numpy.argmin(z)+2]-z[numpy.argmin(z)])*2+(x[numpy.argmin(z)+2]-x[numpy.argmin(z)])**2)*(1/2)
d=(a+b+c)/2
e=(d*(d-a)*(d-b)*(d-c))*(1/2)
k = (4*e)/(a*b*c)

#plot
matplotlib.pyplot.scatter(x,y)
matplotlib.pyplot.plot(x,z,'r')

#results
#'''
print("Y=",end='')
for j in range(0,n+1):
	print("+",p[j],"*(X)^",n-j,sep='',end='')
print("",sep=' ',end='\n')
print("Minimum Value: ", m)
print("Curvature: ", k)
'''
print"Coefficients: ",p
print "Minimum Value: ", m
print "Curvature: ", k
#'''

#show
matplotlib.pyplot.title('regression test')
matplotlib.pyplot.xlabel('random data first column')
matplotlib.pyplot.ylabel('random data second column')
matplotlib.pyplot.show()
