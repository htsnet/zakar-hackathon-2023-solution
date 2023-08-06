#variables
DAY_BASE = 2305

# max alerts: 2342
# max sensors: 2549

import math
def day_senoidal(day):
    return 365 * math.sin(day * math.pi / 180)