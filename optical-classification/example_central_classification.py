"""
Created on 2019-12-20
@author: Ricardo Padrino - github.com/rpadrino - IMDEA Networks
"""
#imports

#from mod_vars import *
#import mod_vars as gvars
from mod_central_classification import *
###from mod_conf import *

import datetime

print("")
#t_now = datetime.datetime.now()
#t_now_str = "[ %s ]" % t_now.strftime("%Y/%m/%d - %H:%M:%S")
#print( t_now_str )
print( "[ %s ]" % ( datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S") ) )
print("")
print("")



print("")
print(" <<< RUN CLASSIFICATION >>>")
print( "[ %s ]" % ( datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S") ) )
print("")
print("")


# RUN CLASSIFICATION
centralClassification(cameras_element='guess', input_path='./classification/', debug=False)
#alternatives
#centralClassification(cameras_element='bottom')
#centralClassification(input_path='./classification/')
#etc.

print("")
print(" <<< END >>>")
print( "[ %s ]" % ( datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S") ) )
print("")


sys.exit(0)
