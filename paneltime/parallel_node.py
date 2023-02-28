import sys

import parallel
import parallel_slave

parallel_slave.run(parallel.Transact(sys.stdin,sys.stdout), True)

