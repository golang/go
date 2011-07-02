# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import hmac

# local imports
import key

def auth(req):
    k = req.get('key')
    return k == hmac.new(key.accessKey, req.get('builder')).hexdigest() or k == key.accessKey

