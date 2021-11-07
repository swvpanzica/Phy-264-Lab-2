#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import stuff 

import numpy as np
import matplotlib.pyplot as plot
from PIL import Image
import matplotlib.cm as cm
from scipy.optimize import curve_fit


# In[167]:


#constants

h = 6.626*10**(-34)
k_b = 1.3806*10**(-23)
c = 2.9979*10**8
Tspace = np.linspace(1000,5000,4001)
lambda1 = np.linspace(300,1100,801)*10**(-9)
T = 6000
I = np.zeros((801,401))

#Intensity function
def Intensity(t,wave):
    return (2*np.pi*h*(c**2))/((wave**5)*(np.e**(h*c/(wave*k_b*t))- 1))


# In[146]:


#First intensity graph

plot.plot(lambda1,Intensity(6000,lambda1))
plot.title('Intensity at 6000K')
plot.xlabel('Wavelengths, m')
plot.ylabel('Intensity')


# In[145]:


for i in range(np.linspace(1000,5000,401).size):
    plot.plot(lambda1,Intensity(np.linspace(1000,5000,401)[i],lambda1))
plot.title('Intensity vs. Wavelength')
plot.xlabel('Wavelengths, m')
plot.ylabel('Intensity, W/m^2')


# In[177]:


#Exmaple ratio plot

blue = []
red = []
for i in range(Tspace.size):
    blue.append(np.sum(Intensity(Tspace[i],np.linspace(400,500,100)*10**(-9))))
    red.append(np.sum(Intensity(Tspace[i],np.linspace(600,700,100)*10**(-9))))
blue = np.array(blue)
red = np.array(red)


plot.plot(Tspace,blue/red)
plot.title('Blue/Red Ratio')
plot.xlabel('T, K')
plot.ylabel('Ratios')


# In[50]:


#my 'get' function
def arrayfunc(a):
    d = np.loadtxt(a, dtype = str, delimiter='  ')
    myset = []
    for i in range(0,d[0:,2].size):
        myset.append(float(d[0:,2][i]))
    myset = np.array(myset)
    return myset


#External color filter
red_ex = arrayfunc('FB650-40.txt')
green_ex = arrayfunc('FB550-40.txt')
blue_ex = arrayfunc('FB450-40.txt')
ext = np.array([red_ex,green_ex,blue_ex])

#Camera filter
red_in = arrayfunc('Basler_red.txt')
green_in = arrayfunc('Basler_green.txt')
blue_in = arrayfunc('Basler_blue.txt')
basler = np.array([red_in,green_in,blue_in])

#Neutral density filter
filt06 = arrayfunc('NE06B.txt')
filt10 = arrayfunc('NE10B.txt')
filt20 = arrayfunc('NE20B.txt')
filt30 = arrayfunc('NE30B.txt')
filt40 = arrayfunc('NE40B.txt')
filt = np.array([filt06,filt10,filt20,filt30,filt40])


# In[51]:


#1.5V graph

ext[0]*basler[0]*Intensity(T,lambda1)
ext[1]*basler[1]*Intensity(T,lambda1)
ext[2]*basler[2]*Intensity(T,lambda1)

plot.plot(lambda1,Intensity(T,lambda1))
plot.plot(lambda1,ext[0]*basler[0]*Intensity(T,lambda1),color='red')
plot.plot(lambda1,ext[1]*basler[1]*Intensity(T,lambda1),color='green')
plot.plot(lambda1,ext[2]*basler[2]*Intensity(T,lambda1),color='blue')
plot.title('Intensity at 6000K, 1.5V')
plot.xlabel('Wavelengths')
plot.ylabel('Intensity')


# In[105]:


#2.85V graph

filt[1]*ext[0]*basler[0]*Intensity(T,lambda1)
filt[0]*ext[1]*basler[1]*Intensity(T,lambda1)
ext[2]*basler[2]*Intensity(T,lambda1)

plot.plot(lambda1,Intensity(T,lambda1))
plot.plot(lambda1,filt[1]*ext[0]*basler[0]*Intensity(T,lambda1),color='red')
plot.plot(lambda1,filt[0]*ext[1]*basler[1]*Intensity(T,lambda1),color='green')
plot.plot(lambda1,ext[2]*basler[2]*Intensity(T,lambda1),color='blue')
plot.title('Intensity at 6000K, 2.85V')
plot.xlabel('Wavelengths')
plot.ylabel('Intensity')


# In[104]:


#3V graph

filt[1]*ext[0]*basler[0]*Intensity(T,lambda1)
filt[0]*ext[1]*basler[1]*Intensity(T,lambda1)
ext[2]*basler[2]*Intensity(T,lambda1)

plot.plot(lambda1,Intensity(T,lambda1))
plot.plot(lambda1,filt[1]*ext[0]*basler[0]*Intensity(T,lambda1),color='red')
plot.plot(lambda1,filt[0]*ext[1]*basler[1]*Intensity(T,lambda1),color='green')
plot.plot(lambda1,ext[2]*basler[2]*Intensity(T,lambda1),color='blue')
plot.title('Intensity at 6000K, 3V')
plot.xlabel('Wavelengths')
plot.ylabel('Intensity')


# In[56]:


#1.5 ratio graphs 

def volt1_5red(t,wave):
    return ext[0]*basler[0]*Intensity(t,wave)
def volt1_5green(t,wave):
    return ext[1]*basler[1]*Intensity(t,wave)
def volt1_5blue(t,wave):
    return ext[2]*basler[2]*Intensity(t,wave)

r1_5 = []
for i in range(1000,5001):
    r1_5.append(np.sum(volt1_5red(i,lambda1)))
r1_5 = np.array(r1_5)
g1_5 = []
for i in range(1000,5001):
    g1_5.append(np.sum(volt1_5green(i,lambda1)))
g1_5 = np.array(g1_5)
b1_5 = []
for i in range(1000,5001):
    b1_5.append(np.sum(volt1_5blue(i,lambda1)))
b1_5 = np.array(b1_5)

#2.85 ratio graphs 

def volt2_85red(t,wave):
    return filt[1]*ext[0]*basler[0]*Intensity(t,wave)
def volt2_85green(t,wave):
    return filt[0]*ext[1]*basler[1]*Intensity(t,wave)
def volt2_85blue(t,wave):
    return ext[2]*basler[2]*Intensity(t,wave)

r2_85 = []
for i in range(1000,5001):
    r2_85.append(np.sum(volt2_85red(i,lambda1)))
r2_85 = np.array(r2_85)
g2_85 = []
for i in range(1000,5001):
    g2_85.append(np.sum(volt2_85green(i,lambda1)))
g2_85 = np.array(g2_85)
b2_85 = []
for i in range(1000,5001):
    b2_85.append(np.sum(volt2_85blue(i,lambda1)))
b2_85 = np.array(b2_85)

#3 ratio graphs 

def volt3red(t,wave):
    return filt[1]*ext[0]*basler[0]*Intensity(t,wave)
def volt3green(t,wave):
    return filt[0]*ext[1]*basler[1]*Intensity(t,wave)
def volt3blue(t,wave):
    return ext[2]*basler[2]*Intensity(t,wave)

r3 = []
for i in range(1000,5001):
    r3.append(np.sum(volt3red(i,lambda1)))
r3 = np.array(r3)
g3 = []
for i in range(1000,5001):
    g3.append(np.sum(volt3green(i,lambda1)))
g3 = np.array(g3)
b3 = []
for i in range(1000,5001):
    b3.append(np.sum(volt3blue(i,lambda1)))
b3 = np.array(b3)


# In[103]:


#Ratio plots 

plot.figure(1,(15,18))
plot.subplot(3,3,1)
plot.plot(Tspace,r1_5/g1_5)
plot.title('Red/Green Ratio, 1.5V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,2)
plot.plot(Tspace,g1_5/b1_5)
plot.title('Green/Blue Ratio, 1.5V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,3)
plot.plot(Tspace,b1_5/r1_5)
plot.title('Blue/Red Ratio, 1.5V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,4)
plot.plot(Tspace,r2_85/g2_85)
plot.title('Red/Green Ratio, 2.85V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,5)
plot.plot(Tspace,g2_85/b2_85)
plot.title('Green/Blue Ratio, 2.85V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,6)
plot.plot(Tspace,b2_85/r2_85)
plot.title('Blue/Red Ratio, 2.85V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,7)
plot.plot(Tspace,r3/g3)
plot.title('Red/Green Ratio, 3V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,8)
plot.plot(Tspace,g3/b3)
plot.title('Green/Blue Ratio, 3V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,9)
plot.plot(Tspace,b3/r3)
plot.title('Blue/Red Ratio, 3V')
plot.xlabel('T,K')
plot.ylabel('Ratios')


# In[188]:


#Function to import photos

from PIL import Image
import os
import math
import matplotlib.pyplot as plot


def BaslerBG8(filename):
    Data=np.fromfile(filename, dtype=np.uint8)
    ImageData=Data.reshape((1024,1280));

    red_tile = np.array([[0, 0],[0, 1]], dtype=np.bool_)
    green_tile_1 = np.array([[0, 1],[0, 0]], dtype=np.bool_)
    green_tile_2 = np.array([[0, 0],[1, 0]], dtype=np.bool_)
    blue_tile = np.array([[1, 0],[0, 0]], dtype=np.bool_)
    red_index_array=np.tile(red_tile,(512,640))
    green_index_array_1=np.tile(green_tile_1,(512,640))
    green_index_array_2=np.tile(green_tile_2,(512,640))
    blue_index_array=np.tile(blue_tile,(512,640))

    Red_layer=ImageData[red_index_array].reshape((512,640))
    Green_layer_1=ImageData[green_index_array_1].reshape((512,640))
    Green_layer_2=ImageData[green_index_array_2].reshape((512,640))
    Blue_layer=ImageData[blue_index_array].reshape((512,640))

    Red_Image=np.empty([512,640,3], np.uint8);
    Red_Image[:,:,0]=Red_layer
    Red_Image[:,:,1]=((Green_layer_1.astype(int)+Green_layer_1.astype(int))/2).astype(np.uint8)
    Red_Image[:,:,2]=Blue_layer
    return(Red_Image)

Basler_Image=BaslerBG8('3V_G.raw')


newim=Image.fromarray(Basler_Image)
newim.show()


# In[113]:


#Import all my photos, slice out colors, divide by exposure time

p1 = BaslerBG8('1_5V_B.raw')
p2 = BaslerBG8('1_5V_B_Control.raw')
p3 = BaslerBG8('1_5V_G.raw')
p4 = BaslerBG8('1_5V_G_Control.raw')
p5 = BaslerBG8('1_5V_R.raw')
p6 = BaslerBG8('1_5V_R_Control.raw')
p7 = BaslerBG8('2_85V_B.raw')
p8 = BaslerBG8('2_85V_B_Control.raw')
p9 = BaslerBG8('2_85V_G.raw')
p10 = BaslerBG8('2_85V_G_Control.raw')
p11 = BaslerBG8('2_85V_R.raw')
p12 = BaslerBG8('2_85V_R_Control.raw')
p13 = BaslerBG8('3V_B.raw')
p14 = BaslerBG8('3V_B_Control.raw')
p15 = BaslerBG8('3V_G.raw')
p16 = BaslerBG8('3V_G_Control.raw')
p17 = BaslerBG8('3V_R.raw')
p18 = BaslerBG8('3V_R_Control.raw')

pnew1 = p1[:,:,2]/(50000*10**(-6))
pnew2 = p2[:,:,2]/(50000*10**(-6))
pnew3 = p3[:,:,1]/(3000*10**(-6))
pnew4 = p4[:,:,1]/(3000*10**(-6))
pnew5 = p5[:,:,0]/(500*10**(-6))
pnew6 = p6[:,:,0]/(500*10**(-6))
pnew7 = p7[:,:,2]/(120*10**(-6))
pnew8 = p8[:,:,2]/(120*10**(-6))
pnew9 = p9[:,:,1]/(90*10**(-6))
pnew10 = p10[:,:,1]/(90*10**(-6))
pnew11 = p11[:,:,0]/(80*10**(-6))
pnew12 = p12[:,:,0]/(80*10**(-6))
pnew13 = p13[:,:,2]/(130*10**(-6))
pnew14 = p14[:,:,2]/(130*10**(-6))
pnew15 = p15[:,:,1]/(80*10**(-6))
pnew16 = p16[:,:,1]/(80*10**(-6))
pnew17 = p17[:,:,0]/(80*10**(-6))
pnew18 = p18[:,:,0]/(80*10**(-6))


# In[115]:


#Calculate signals and ratios

signalR1_5 = np.sum(pnew5) - np.sum(pnew6)
signalG1_5 = np.sum(pnew3) - np.sum(pnew4)
signalB1_5 = np.sum(pnew1) - np.sum(pnew2)
signalR2_85 = np.sum(pnew11) - np.sum(pnew12)
signalG2_85 = np.sum(pnew9) - np.sum(pnew10)
signalB2_85 = np.sum(pnew7) - np.sum(pnew8)
signalR3 = np.sum(pnew17) - np.sum(pnew18)
signalG3 = np.sum(pnew15) - np.sum(pnew16)
signalB3 = np.sum(pnew13) - np.sum(pnew14)

Rrg1_5 = signalR1_5/signalG1_5
Rgb1_5 = signalG1_5/signalB1_5
Rbr1_5 = signalB1_5/signalR1_5

Rrg2_85 = signalR2_85/signalG2_85
Rgb2_85 = signalG2_85/signalB2_85
Rbr2_85 = signalB2_85/signalR2_85

Rrg3 = signalR3/signalG3
Rgb3 = signalG3/signalB3
Rbr3 = signalB3/signalR3


# In[189]:


print(Rrg1_5,Rgb1_5,Rbr1_5)
print(Rrg2_85,Rgb2_85,Rbr2_85)
print(Rrg3,Rgb3,Rbr3)


# | Voltage (V)     | Red/Green | Green/Blue          | Blue/Red         |
# |:--------------:|:--------------:|:--------------:|:--------------:|
# | 1.5  | 7.78 | 8.47 | 0.02 |
# | 2.85 | 1.10 | 1.32 | 0.69 |
# | 3  | 1.19 | 1.66 |0.51 |

# In[120]:


#Ratio plots 

plot.figure(1,(15,18))
plot.subplot(3,3,1)
plot.plot(Tspace,r1_5/g1_5)
plot.axhline(y=Rrg1_5)
plot.title('Red/Green Ratio, 1.5V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,2)
plot.plot(Tspace,g1_5/b1_5)
plot.axhline(y=Rgb1_5)
plot.title('Green/Blue Ratio, 1.5V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,3)
plot.plot(Tspace,b1_5/r1_5)
plot.axhline(y=Rbr1_5)
plot.title('Blue/Red Ratio, 1.5V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,4)
plot.plot(Tspace,r2_85/g2_85)
plot.axhline(y=Rrg2_85)
plot.title('Red/Green Ratio, 2.85V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,5)
plot.plot(Tspace,g2_85/b2_85)
plot.axhline(y=Rgb2_85)
plot.title('Green/Blue Ratio, 2.85V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,6)
plot.plot(Tspace,b2_85/r2_85)
plot.axhline(y=Rbr2_85)
plot.title('Blue/Red Ratio, 2.85V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,7)
plot.plot(Tspace,r3/g3)
plot.axhline(y=Rrg3)
plot.title('Red/Green Ratio, 3V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,8)
plot.plot(Tspace,g3/b3)
plot.axhline(y=Rgb3)
plot.title('Green/Blue Ratio, 3V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,9)
plot.plot(Tspace,b3/r3)
plot.axhline(y=Rbr3)
plot.title('Blue/Red Ratio, 3V')
plot.xlabel('T,K')
plot.ylabel('Ratios')


# In[129]:


#Find TEMPERATURES!

temp = []
x = []
for i in range(r1_5.size):
    x.append(abs((r1_5/g1_5)[i] - Rrg1_5))
for i in range(r1_5.size):
    if x[i] == min(x):
        temp.append(i+1000)
        
x = []
for i in range(r1_5.size):
    x.append(abs((g1_5/b1_5)[i] - Rgb1_5))
for i in range(r1_5.size):
    if x[i] == min(x):
        temp.append(i+1000)
        
x = []
for i in range(r1_5.size):
    x.append(abs((b1_5/r1_5)[i] - Rbr1_5))
for i in range(r1_5.size):
    if x[i] == min(x):
        temp.append(i+1000)

x = []
for i in range(r2_85.size):
    x.append(abs((r2_85/g2_85)[i] - Rrg2_85))
for i in range(r2_85.size):
    if x[i] == min(x):
        temp.append(i+1000)
        
x = []
for i in range(r2_85.size):
    x.append(abs((g2_85/b2_85)[i] - Rgb2_85))
for i in range(r2_85.size):
    if x[i] == min(x):
        temp.append(i+1000)
        
x = []
for i in range(r2_85.size):
    x.append(abs((b2_85/r2_85)[i] - Rbr2_85))
for i in range(r2_85.size):
    if x[i] == min(x):
        temp.append(i+1000)
        
x = []
for i in range(r3.size):
    x.append(abs((r3/g3)[i] - Rrg3))
for i in range(r3.size):
    if x[i] == min(x):
        temp.append(i+1000)
        
x = []
for i in range(r3.size):
    x.append(abs((g3/b3)[i] - Rgb3))
for i in range(r3.size):
    if x[i] == min(x):
        temp.append(i+1000)
        
x = []
for i in range(r3.size):
    x.append(abs((b3/r3)[i] - Rbr3))
for i in range(r3.size):
    if x[i] == min(x):
        temp.append(i+1000)


# In[131]:


#Error

plot.figure(1,(15,18))
plot.subplot(3,3,1)
plot.plot(Tspace,r1_5/g1_5)
plot.axhline(y=Rrg1_5)
plot.axhline(y=1.28*Rrg1_5, color = 'red')
plot.axhline(y=0.72*Rrg1_5, color = 'red')
plot.title('Red/Green Ratio, 1.5V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,2)
plot.plot(Tspace,g1_5/b1_5)
plot.axhline(y=Rgb1_5)
plot.axhline(y=1.28*Rgb1_5, color = 'red')
plot.axhline(y=0.72*Rgb1_5, color = 'red')
plot.title('Green/Blue Ratio, 1.5V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,3)
plot.plot(Tspace,b1_5/r1_5)
plot.axhline(y=Rbr1_5)
plot.axhline(y=1.28*Rbr1_5, color = 'red')
plot.axhline(y=0.72*Rbr1_5, color = 'red')
plot.title('Blue/Red Ratio, 1.5V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,4)
plot.plot(Tspace,r2_85/g2_85)
plot.axhline(y=Rrg2_85)
plot.axhline(y=1.28*Rrg2_85, color = 'red')
plot.axhline(y=0.72*Rrg2_85, color = 'red')
plot.title('Red/Green Ratio, 2.85V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,5)
plot.plot(Tspace,g2_85/b2_85)
plot.axhline(y=Rgb2_85)
plot.axhline(y=1.28*Rgb2_85, color = 'red')
plot.axhline(y=0.72*Rgb2_85, color = 'red')
plot.title('Green/Blue Ratio, 2.85V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,6)
plot.plot(Tspace,b2_85/r2_85)
plot.axhline(y=Rbr2_85)
plot.axhline(y=1.28*Rbr2_85, color = 'red')
plot.axhline(y=0.72*Rbr2_85, color = 'red')
plot.title('Blue/Red Ratio, 2.85V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,7)
plot.plot(Tspace,r3/g3)
plot.axhline(y=Rrg3)
plot.axhline(y=1.28*Rrg3, color = 'red')
plot.axhline(y=0.72*Rrg3, color = 'red')
plot.title('Red/Green Ratio, 3V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,8)
plot.plot(Tspace,g3/b3)
plot.axhline(y=Rgb3)
plot.axhline(y=1.28*Rgb3, color = 'red')
plot.axhline(y=0.72*Rgb3, color = 'red')
plot.title('Green/Blue Ratio, 3V')
plot.xlabel('T,K')
plot.ylabel('Ratios')

plot.subplot(3,3,9)
plot.plot(Tspace,b3/r3)
plot.axhline(y=Rbr3)
plot.axhline(y=1.28*Rbr3, color = 'red')
plot.axhline(y=0.72*Rbr3, color = 'red')
plot.title('Blue/Red Ratio, 3V')
plot.xlabel('T,K')
plot.ylabel('Ratios')


# In[137]:


temphigh = []
x = []
for i in range(r1_5.size):
    x.append(abs((r1_5/g1_5)[i] - 1.28*Rrg1_5))
for i in range(r1_5.size):
    if x[i] == min(x):
        temphigh.append(i+1000)
        
x = []
for i in range(r1_5.size):
    x.append(abs((g1_5/b1_5)[i] - 1.28*Rgb1_5))
for i in range(r1_5.size):
    if x[i] == min(x):
        temphigh.append(i+1000)
        
x = []
for i in range(r1_5.size):
    x.append(abs((b1_5/r1_5)[i] - 1.28*Rbr1_5))
for i in range(r1_5.size):
    if x[i] == min(x):
        temphigh.append(i+1000)

x = []
for i in range(r2_85.size):
    x.append(abs((r2_85/g2_85)[i] - 1.28*Rrg2_85))
for i in range(r2_85.size):
    if x[i] == min(x):
        temphigh.append(i+1000)
        
x = []
for i in range(r2_85.size):
    x.append(abs((g2_85/b2_85)[i] - 1.28*Rgb2_85))
for i in range(r2_85.size):
    if x[i] == min(x):
        temphigh.append(i+1000)
        
x = []
for i in range(r2_85.size):
    x.append(abs((b2_85/r2_85)[i] - 1.28*Rbr2_85))
for i in range(r2_85.size):
    if x[i] == min(x):
        temphigh.append(i+1000)
        
x = []
for i in range(r3.size):
    x.append(abs((r3/g3)[i] - 1.28*Rrg3))
for i in range(r3.size):
    if x[i] == min(x):
        temphigh.append(i+1000)
        
x = []
for i in range(r3.size):
    x.append(abs((g3/b3)[i] - 1.28*Rgb3))
for i in range(r3.size):
    if x[i] == min(x):
        temphigh.append(i+1000)
        
x = []
for i in range(r3.size):
    x.append(abs((b3/r3)[i] - 1.28*Rbr3))
for i in range(r3.size):
    if x[i] == min(x):
        temphigh.append(i+1000)
        
templow = []
x = []
for i in range(r1_5.size):
    x.append(abs((r1_5/g1_5)[i] - 0.72*Rrg1_5))
for i in range(r1_5.size):
    if x[i] == min(x):
        templow.append(i+1000)
        
x = []
for i in range(r1_5.size):
    x.append(abs((g1_5/b1_5)[i] - 0.72*Rgb1_5))
for i in range(r1_5.size):
    if x[i] == min(x):
        templow.append(i+1000)
        
x = []
for i in range(r1_5.size):
    x.append(abs((b1_5/r1_5)[i] - 0.72*Rbr1_5))
for i in range(r1_5.size):
    if x[i] == min(x):
        templow.append(i+1000)

x = []
for i in range(r2_85.size):
    x.append(abs((r2_85/g2_85)[i] - 0.72*Rrg2_85))
for i in range(r2_85.size):
    if x[i] == min(x):
        templow.append(i+1000)
        
x = []
for i in range(r2_85.size):
    x.append(abs((g2_85/b2_85)[i] - 0.72*Rgb2_85))
for i in range(r2_85.size):
    if x[i] == min(x):
        templow.append(i+1000)
        
x = []
for i in range(r2_85.size):
    x.append(abs((b2_85/r2_85)[i] - 0.72*Rbr2_85))
for i in range(r2_85.size):
    if x[i] == min(x):
        templow.append(i+1000)
        
x = []
for i in range(r3.size):
    x.append(abs((r3/g3)[i] - 0.72*Rrg3))
for i in range(r3.size):
    if x[i] == min(x):
        templow.append(i+1000)
        
x = []
for i in range(r3.size):
    x.append(abs((g3/b3)[i] - 0.72*Rgb3))
for i in range(r3.size):
    if x[i] == min(x):
        templow.append(i+1000)
        
x = []
for i in range(r3.size):
    x.append(abs((b3/r3)[i] - 0.72*Rbr3))
for i in range(r3.size):
    if x[i] == min(x):
        templow.append(i+1000)


# In[ ]:




