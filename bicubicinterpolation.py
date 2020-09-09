'''
    Takes in a numpy array and returns a similar one with larger dimensions, by interpolating values from the smaller array

    © 2020 Moses Odhiambo
'''
import numpy as np
from matplotlib import pyplot as plt

def upscale(image, factor, plot_result=False):
    #image is a numpy array of shape n,n,3
    shape = image.shape
    old_res_x, old_res_y = shape[-3], shape[-2]
    new_res_x, new_res_y = (old_res_x - 1) * factor + 1 , (old_res_y - 1) * factor + 1
    depth = shape[-1]
    result = np.zeros((new_res_x, new_res_y, depth), dtype=np.float32)

    for channel in range(depth):
        for X in range(old_res_x-1):  #large X strides
            for Y in range(old_res_y-1):  #large Y strides
                x0,y0 = X*factor, Y*factor
                x1,y1 = (X+1)*factor, (Y+1)*factor

                result[x0, y0, channel] = image[X, Y, channel]      #top left corner
                result[x1, y0, channel] = image[X+1, Y, channel]    #top right corner
                result[x0, y1, channel] = image[X, Y+1, channel]    #bottom left corner
                result[x1, y1, channel] = image[X+1, Y+1, channel]  #bottom right corner
                
                points_map = np.zeros((4, 4))
                for x_map in range(4):
                    if X+(x_map-1) < 0 or X+(x_map-1) >= old_res_x:
                        continue
                    for y_map in range(4):
                        if Y+(y_map-1) < 0 or Y+(y_map-1) >= old_res_y:
                            continue
                        points_map[x_map, y_map] = image[X+(x_map-1), Y+(y_map-1), channel]

                for x in range(factor+1):  #small x strides
                    for y in range(factor+1):  #small y strides
                        if (x,y) == (0,0) or (x,y) == (factor,0) or (x,y) == (0,factor) or (x,y) == (factor,factor):
                            continue
                        norm_x, norm_y = x/factor, y/factor
                        point_x, point_y = X*factor+x, Y*factor+y
                        result[point_x, point_y, channel] = bicubic_interpolation(points_map, norm_x, norm_y)

    if plot_result:
        heights = np.mean(result, axis=-1)
        x, y = range(heights.shape[0]), range(heights.shape[1])
        hf=plt.figure()
        ha=hf.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y)
        ha.plot_surface(X, Y, heights, cmap='viridis')
        ha.set_xlabel("X")
        ha.set_ylabel("Y")
        plt.show()
    return result

def cubic_interpolation(p, x):
	#constants
    a = 1.5*(p[1] - p[2]) + 0.5*(p[3] - p[0])
    b = p[0] - 2.5*p[1] + 2*p[2] - 0.5*p[3]
    c = 0.5*(p[2] - p[0])
    d = p[1]
    return a*pow(x, 3) + b*pow(x, 2) + c*x + d  #third order polynomial

def bicubic_interpolation(p, x, y):
    arr = []
    arr.append(cubic_interpolation(p[0], y))
    arr.append(cubic_interpolation(p[1], y))
    arr.append(cubic_interpolation(p[2], y))
    arr.append(cubic_interpolation(p[3], y))
    return cubic_interpolation(arr, x)
