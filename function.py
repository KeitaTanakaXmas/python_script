import numpy as np

def linear(x,a,b):
    return a*x+b

def calculate_linear(x1,y1,x2,y2):
    a = (y1 - y2)/(x1 - x2)
    b = y1 - a * x1
    return a,b

def linear_return_x(a,b,y):
    return (y-b)/a