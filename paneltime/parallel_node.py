

import parallel
import parallel_slave
import sys

parallel_slave.run(parallel.Transact(sys.stdin,sys.stdout))