# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This is a Django custom template filter to work around the
# fact that GAE's urlencode filter doesn't handle unicode strings.

from google.appengine.ext import webapp

register = webapp.template.create_template_register()

@register.filter
def toutf8(value):
    return value.encode("utf-8")
