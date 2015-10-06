__author__ = 'kkboy'

import urllib

a = range(21, 41)
for i in a:
    urllib.urlretrieve("http://zf.ltuj.com/2015/91/18065/935648%02d.jpg" % i, "images/%02d.jpg" % i)