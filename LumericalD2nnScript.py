import numpy as np
import os, time, sys
import importlib as imp
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process
# print("Python version: " + sys.version)
lumapi = imp.machinery.SourceFileLoader("lumepi","D:\\Program Files\\Lumerical\\v202\\api\\python\\lumapi.py").load_module()


size = 2                                                         # size of the network
num_layer = 5                                                    # number of the layers

x_span = 400e-6                                                  # pixel size: 400 um (400e-6*200 = 0.08)
y_span = 400e-6
z = 3e-2                                                         # distance of propagation(the distance bewteen two layers)
dist = 1e-2                                                      # the distance bewteen the last layer and the detector plane

height_map = np.load('./height_map.npy')                         # load height_map
filter_height_map = np.load('./filter_height_map.npy')

hide = True                                                      # hide the GUI will dramatically cut down the construction time

material = "<Object defined dielectric>"                         # VeroBlackPlus RGD875, refractive index: 1.7227
refractive_index = 1.7227


# construction progress bar
def construction_progress_bar(iteration, total= 1*size*size,prefix = 'Progress', suffix='Complete', decimals = 2, fill='>'):
    # percent format
    length = 50
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
    filledlength = int(length * iteration // total)
    bar = fill*filledlength + '-'*(length- filledlength)
    print(f'\r{prefix}|{bar}|{percent}% {suffix}',end = '\r')
    if iteration == total:
        print()

def addfilter():
    # add a filter behind the source
    global counter
    fdtd.addstructuregroup()
    fdtd.set("name","filter")
    fdtd.adduserprop("material",5,material)                     # set the material prop to the structure group
    fdtd.groupscope("filter")
    for j in range(0,size):                                     # i:x-axis, j: y-axis
            for i in range(0,size):
                fdtd.addrect()                                  # create and select the rect
                fdtd.set("name","filter"+str(i)+"_"+str(j))       # set the name of the little rect
                fdtd.set("x",x_span/2 + i*x_span)               # set the x center position of the rect
                fdtd.set("x span",x_span)
                fdtd.set("y",-y_span/2 - j*y_span)              # set the y center position of the rect
                fdtd.set("y span",y_span)
                fdtd.set("z min",-1*z)                          # set the z center position of the rect
                fdtd.set("z max",-1*z + filter_height_map[j][i])
                fdtd.set("index",refractive_index)
                counter+=1
                construction_progress_bar(iteration=counter,prefix = 'Progress(filter)')
    fdtd.groupscope("::model")
    return None

def addlayer(num_layer):
    def override_layer_mesh(layer_name, num_layer, layer_z_position):
        fdtd.groupscope("::model")
        fdtd.addmesh()
        fdtd.set("name", layer_name + "mesh")
        # set dimension
        fdtd.set("x", size * x_span / 2)
        fdtd.set('x span', size * x_span)
        fdtd.set("y", -1 * size * y_span / 2)
        fdtd.set("y span", size * x_span)
        fdtd.set("z max", layer_z_position)
        fdtd.set("z min", layer_z_position - np.max(height_map[num_layer]))  # 这个值是height_map里面最厚的那个值
        # enable in X,Y,Z direction
        # 1:enable, 0:disable
        fdtd.set("override x mesh", 1)
        fdtd.set("override y mesh", 1)
        fdtd.set("override z mesh", 1)
        # restrict mesh by defining maximum step size
        fdtd.set("set maximum mesh step", 1)
        fdtd.set("dx", 100e-6)
        fdtd.set("dy", 100e-6)
        fdtd.set("dz", 100e-6)
        return None

    # add one diffraction layer
    fdtd = lumapi.FDTD(filename="./D2NN_Simulation.fsp", hide=hide)
    global counter, size, z, x_span, y_span, material, refractive_index
    counter = 0
    l = num_layer  # l is the layer number

    fdtd.addstructuregroup()
    groupname = "diffraction_layer_" + str(l)  # one group represents one layer
    fdtd.set("name", groupname)

    fdtd.groupscope(groupname)
    layer_z_position = 0 + l * z
    for j in range(0, size):  # i:x-axis, j: y-axis
        for i in range(0, size):
            fdtd.addrect()  # create and select the rect
            fdtd.set("name", "rect" + str(i) + "_" + str(j))  # set the name of the little rect
            fdtd.set("x", x_span / 2 + i * x_span)  # set the x center position of the rect
            fdtd.set("x span", x_span)
            fdtd.set("y", -y_span / 2 - j * y_span)  # set the y center position of the rect
            fdtd.set("y span", y_span)
            fdtd.set("z max", 0 + layer_z_position)  # set the z center position of the rect
            fdtd.set("z min", 0 + layer_z_position - height_map[l][j][i])
            fdtd.set("material", material)
            fdtd.set("index", refractive_index)
            counter += 1
            # fdtd.addtogroup(groupname)
            # display the progress of layer construction
            construction_progress_bar(iteration=counter,prefix = 'Progress'+'('+"temp_layer_" + str(num_layer) +')')
    override_layer_mesh(layer_name=groupname, num_layer=l, layer_z_position=layer_z_position)
    fname = "temp_layer_" + str(num_layer) + ".fsp"
    fdtd.save(fname)
    fdtd.close()
    # print("temp_layer_" + str(num_layer) + " has completed.")
    return None

def addsource():
    # add a plane wave source
    fdtd.addplane()
    fdtd.set("injection axis","z")
    fdtd.set("direction","forward") # propagate in the positive z direction
    fdtd.set("x",size*x_span/2)
    fdtd.set("x span",size*x_span)
    fdtd.set("y",-1*size*y_span/2)
    fdtd.set("y span",size*x_span)
    fdtd.set("z",-1*z)
    fdtd.set("wavelength start",750e-6)
    fdtd.set("wavelength stop",750e-6)
    return None

def addDetectorPlane():
    # add monitor, the detector plane
    fdtd.addpower()
    fdtd.set('name','detector_plane_monitor')
    fdtd.set("x",size*x_span/2)
    fdtd.set('x span',size*x_span)
    fdtd.set("y",-1*size*y_span/2)
    fdtd.set("y span",size*x_span)
    fdtd.set("z",(num_layer-1)*z + dist)
    return None

def addCrossSectionalMonitor():
    # add monitor, cross-sectional monitor
    fdtd.addprofile()                          # Frequency-domain field profile monitor
    fdtd.set('name','cross_sectional_monitor')
    fdtd.set("monitor type","2D Y-normal")     # x-z plane monitor
    fdtd.set("x",size*x_span/2)
    fdtd.set('x span',size*x_span)
    fdtd.set("y",-1*size*y_span/2)
    fdtd.set("z min",0-np.max(height_map[0]))
    fdtd.set("z max",(num_layer-1)*z + dist)
    return None

def addSimulationRegion():
    # add a simulation region
    fdtd.groupscope("::model")
    fdtd.addfdtd()
    fdtd.set('dimension',2)                 # 1 = 2D, 2 = 3D, set a 3D simulation region
    fdtd.set("x",size*x_span/2)
    fdtd.set('x span',size*x_span)          # set the x span of the simulation region
    fdtd.set("y",-1*size*y_span/2)          # we are using the negative part of the y-axis
    fdtd.set("y span",size*x_span)
    fdtd.set("z max",(num_layer-1)*z + dist)
    fdtd.set("z min",-1*z)
    return None



if __name__ == '__main__':
    print("Initializing...")
    fdtd = lumapi.FDTD(hide=hide)  # hide the GUI of the software

    global counter
    counter = 0  # counter: count the number of constructed rects

    addsource()
    addDetectorPlane()
    addCrossSectionalMonitor()
    addSimulationRegion()
    addfilter()

    fdtd.save("D2NN_Simulation.fsp")
    fdtd.close()

    # add layers
    processes = []
    for l in range(0, num_layer):
        p = multiprocessing.Process(target=addlayer, args=[l])
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    print("Construction Finished")
















