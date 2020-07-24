"""
Created on 2019-12-20
@author: Ricardo Padrino - github.com/rpadrino - IMDEA Networks
"""
#imports

#from mod_vars import *
#import mod_vars as gvars
from mod_local_classification import *
from mod_conf import *

import datetime

print("")
print(" <<< CREATE MODELS >>>")
#t_now = datetime.datetime.now()
#t_now_str = "[ %s ]" % t_now.strftime("%Y/%m/%d - %H:%M:%S")
#print( t_now_str )
print( "[ %s ]" % ( datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S") ) )
print("")
print("")

# CREATE MODELS
cc_models = ClassifModels()

# LOAD MODELS
loadClassificationModels(cc_models)

print("")
print(" <<< RUN CLASSIFICATION >>>")
print( "[ %s ]" % ( datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S") ) )
print("")
print("")


# RUN CLASSIFICATION
localClassification(models=cc_models, input_path='./crops/', debug=False)
#alternatives
#loadClassificationModels(cc_models)
#loadClassificationModels(models=cc_models, input_path='./crops/')
#etc.


print("")
print(" <<< END >>>")
print( "[ %s ]" % ( datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S") ) )
print("")


sys.exit(0)


#######################################################
#######################################################

# delete models allocated in memory, create object for model and load
del cc_models
cc_models = ClassifModels()
loadClassificationModels(cc_models)
