#import sys
#import pickle
import numpy as np
import os
import sys
from opengl_viewer.opengl_viewer import OpenGlViewer

if __name__ == '__main__':
    if len(sys.argv) == 0:
	print("No arguments provided")
	sys.exit()

    p = sys.argv[1]
    q = sys.argv[2]
    r = sys.argv[3]
    s = sys.argv[4]
#    print p
#    print q
#    print r
    filename = p.split(".")

    lPosition_prev=np.loadtxt(p, usecols=[0])
    lDirection_prev=np.loadtxt(p, usecols=[1])

    lPosition=np.loadtxt(q, usecols=[0])
    lDirection=np.loadtxt(q, usecols=[1]) + np.random.normal(0,0.01,3)
    print lDirection 

    # interpolate between this and the previous saved pose and change camera view
    dist = lPosition - lPosition_prev
    distnorm = np.linalg.norm(dist)
    OpenGlViewer().collect_data( lPosition_prev + float(r)*dist/distnorm , lDirection , 'training_'+filename[0]+'_'+r+'_'+s)
 

