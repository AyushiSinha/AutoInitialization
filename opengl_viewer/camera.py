import numpy as np

p = np.array([0.0, -35.0, 0.0])
f = np.array([0.0, 1.0, 0.0])
u = np.array([0.0, 0.0, 1.0])


class Camera:
    def __init__(self):
	self.position = p.copy()
	self.forward = f.copy()
	self.up = u.copy()
	self.right = self.setRight()

    def setRight( self ):
	tmp = np.cross(self.forward, self.up)
	return tmp/np.linalg.norm(tmp)

    def rotateUp( self, angle ):
	self.rotatePoint(self.up, angle)

    def rotateRight( self, angle ):
	self.rotatePoint(self.right, angle)

    def rotateForward( self, angle ):
	self.rotatePoint(self.forward, angle)

    def rotatePoint( self, axis, angle ):
	x,y,z = self.forward
	self.forward = self.forward * np.cos(angle) - axis * np.sin(angle)
	self.forward /= np.linalg.norm(self.forward)
	tmp = np.cross(self.forward, axis)
	axis = np.cross(self.forward, tmp)
    
    def translate( self, t ):
	self.position += t


