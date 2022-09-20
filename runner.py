#!/usr/bin/env python3
"""Basic python processing driver example."""
import sys
import time
import datetime
import logging
import argparse
import os

def main():
    """The main driver routine."""
    parser = argparse.ArgumentParser(usage="%(prog)s [options]", \
                                description='runs a single task group')
    parser.add_argument("--log", type=str, help="A logfile, full path.", default=None)
    parser.add_argument("work", type=str, help="The work set, i.e. file1, file2", nargs=2)
    args = parser.parse_args()

    logging.basicConfig(filename=args.log, level=logging.DEBUG)

    fls_log_msg = " ".join([args.work[0], args.work[1]])
    t_start = datetime.datetime.now()
    logging.info("workset \"%s\", start %s", fls_log_msg, t_start.strftime("%Y%j%H.%M.%S") )

    #
    # run the three step process
    #
    logging.debug("running step 1 %s  ...", args.work[0])
    os.system('/nex/modules/m/hydrolight/HE60/backend/./EcoLight6 < batch/{}'.format(args.work[0]))
    
    logging.debug("running step 2 %s  ...", args.work[1])
    os.system('/nex/modules/m/hydrolight/HE60/backend/./EcoLight6 < batch/{}'.format(args.work[1]))
    
    time.sleep(2)

    # 
    # run the three step process
    #
    logging.info("performing post file cleanup...")

    # report on exec time
    t_delt = datetime.datetime.now() - t_start
    logging.info("runtime \"%0.1f\" minutes, seconds %d", float(t_delt.seconds)/60.0, t_delt.seconds) 

    return 0
#main

if __name__ == '__main__':
    sys.exit(main())
