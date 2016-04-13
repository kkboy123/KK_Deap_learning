__author__ = 'kkboy'

import urllib

a = range(801, 843)
for i in a:
    urllib.urlretrieve("http://zf.ltuj.com/2014/69/13745/83323%03d.jpg" % i, "images/2014-69/%03d.jpg" % i)

'''http://ltuj.com/search.html?key=%E5%98%89%E5%AE%9D%E8%B4%9D'''