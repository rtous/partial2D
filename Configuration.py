import BodyModelOPENPOSE15
import BodyModelOPENPOSE25

class Configuration(object):
  def __init__(self): 
  	self.bodyModel = None
  	self.norm = 0

  def set_BODY_MODEL(self, BODY_MODEL):
  	if BODY_MODEL == "OPENPOSE_15":
  		self.bodyModel = BodyModelOPENPOSE15
  	elif BODY_MODEL == "OPENPOSE_25":
  		self.bodyModel = BodyModelOPENPOSE25



