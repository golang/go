# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This is the server part of the package dashboard.
# It must be run by App Engine.

from google.appengine.api import memcache
from google.appengine.runtime import DeadlineExceededError
from google.appengine.ext import db
from google.appengine.ext import webapp
from google.appengine.ext.webapp import template
from google.appengine.ext.webapp.util import run_wsgi_app
import binascii
import datetime
import hashlib
import hmac
import logging
import os
import re
import struct
import time
import urllib2

# Storage model for package info recorded on server.
# Just path, count, and time of last install.
class Package(db.Model):
    path = db.StringProperty()
    web_url = db.StringProperty()  # derived from path
    count = db.IntegerProperty()
    last_install = db.DateTimeProperty()

re_bitbucket = re.compile(r'^bitbucket\.org/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+$')
re_googlecode = re.compile(r'^[a-z0-9\-]+\.googlecode\.com/(svn|hg)$')
re_github = re.compile(r'^github\.com/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+$')

MaxPathLength = 100

class PackagePage(webapp.RequestHandler):
    def get(self):
        if self.request.get('fmt') == 'json':
            return self.json()

        q = Package.all()
        q.order('-last_install')
        by_time = q.fetch(100)

        q = Package.all()
        q.order('-count')
        by_count = q.fetch(100)

        self.response.headers['Content-Type'] = 'text/html; charset=utf-8'
        path = os.path.join(os.path.dirname(__file__), 'package.html')
        self.response.out.write(template.render(path, {"by_time": by_time, "by_count": by_count}))

    def json(self):
        self.response.set_status(200)
        self.response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        q = Package.all()
        s = '{"packages": ['
        sep = ''
        for r in q.fetch(1000):
            s += '%s\n\t{"path": "%s", "last_install": "%s", "count": "%s"}' % (sep, r.path, r.last_install, r.count)
            sep = ','
        s += '\n]}\n'
        self.response.out.write(s)

    def can_get_url(self, url):
        try:
            req = urllib2.Request(url)
            response = urllib2.urlopen(req)
            return True
        except:
            return False

    def is_valid_package_path(self, path):
        return (re_bitbucket.match(path) or
            re_googlecode.match(path) or
            re_github.match(path))

    def record_pkg(self, path):
        # sanity check string
        if not path or len(path) > MaxPathLength or not self.is_valid_package_path(path):
            return False

        # look in datastore
        key = 'pkg-' + path
        p = Package.get_by_key_name(key)
        if p is None:
            # not in datastore - verify URL before creating
            if re_bitbucket.match(path):
                check_url = 'http://' + path + '/?cmd=heads'
                web = 'http://' + path + '/'
            elif re_github.match(path):
                # github doesn't let you fetch the .git directory anymore.
                # fetch .git/info/refs instead, like git clone would.
                check_url = 'http://'+path+'.git/info/refs'
                web = 'http://' + path
            elif re_googlecode.match(path):
                check_url = 'http://'+path
                web = 'http://code.google.com/p/' + path[:path.index('.')]
            else:
                logging.error('unrecognized path: %s', path)
                return False
            if not self.can_get_url(check_url):
                logging.error('cannot get %s', check_url)
                return False
            p = Package(key_name = key, path = path, count = 0, web_url = web)

        # update package object
        p.count += 1
        p.last_install = datetime.datetime.utcnow()
        p.put()
        return True

    def post(self):
        path = self.request.get('path')
        ok = self.record_pkg(path)
        if ok:
            self.response.set_status(200)
            self.response.out.write('ok')
        else:
            logging.error('invalid path in post: %s', path)
            self.response.set_status(500)
            self.response.out.write('not ok')

def main():
    app = webapp.WSGIApplication([('/package', PackagePage)], debug=True)
    run_wsgi_app(app)

if __name__ == '__main__':
    main()
