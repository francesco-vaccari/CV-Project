import utilsBenchmark as utils
import batchesBenchmark as batch

mode = 0 #allows for future implementation of variants.
# Current Modes:
# 0 -

if mode == 0:
    name = "MOG2"
    ct, log = utils.initialize_timestamp(name)


    utils.cleanup(ct, log)
