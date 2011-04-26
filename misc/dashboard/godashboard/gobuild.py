# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This is the server part of the continuous build system for Go. It must be run
# by AppEngine.

from google.appengine.api import mail
from google.appengine.api import memcache
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
import bz2

# local imports
import key
import const

# The majority of our state are commit objects. One of these exists for each of
# the commits known to the build system. Their key names are of the form
# <commit number (%08x)> "-" <hg hash>. This means that a sorting by the key
# name is sufficient to order the commits.
#
# The commit numbers are purely local. They need not match up to the commit
# numbers in an hg repo. When inserting a new commit, the parent commit must be
# given and this is used to generate the new commit number. In order to create
# the first Commit object, a special command (/init) is used.
class Commit(db.Model):
    num = db.IntegerProperty() # internal, monotonic counter.
    node = db.StringProperty() # Hg hash
    parentnode = db.StringProperty() # Hg hash
    user = db.StringProperty()
    date = db.DateTimeProperty()
    desc = db.BlobProperty()

    # This is the list of builds. Each element is a string of the form <builder
    # name> '`' <log hash>. If the log hash is empty, then the build was
    # successful.
    builds = db.StringListProperty()

    fail_notification_sent = db.BooleanProperty()

class Cache(db.Model):
    data = db.BlobProperty()
    expire = db.IntegerProperty()

# A CompressedLog contains the textual build log of a failed build. 
# The key name is the hex digest of the SHA256 hash of the contents.
# The contents is bz2 compressed.
class CompressedLog(db.Model):
    log = db.BlobProperty()

N = 30

def cache_get(key):
    c = Cache.get_by_key_name(key)
    if c is None or c.expire < time.time():
        return None
    return c.data

def cache_set(key, val, timeout):
    c = Cache(key_name = key)
    c.data = val
    c.expire = int(time.time() + timeout)
    c.put()

def cache_del(key):
    c = Cache.get_by_key_name(key)
    if c is not None:
        c.delete()

def builderInfo(b):
    f = b.split('-', 3)
    goos = f[0]
    goarch = f[1]
    note = ""
    if len(f) > 2:
        note = f[2]
    return {'name': b, 'goos': goos, 'goarch': goarch, 'note': note}

def builderset():
    q = Commit.all()
    q.order('-__key__')
    results = q.fetch(N)
    builders = set()
    for c in results:
        builders.update(set(parseBuild(build)['builder'] for build in c.builds))
    return builders
    
class MainPage(webapp.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/html; charset=utf-8'

        try:
            page = int(self.request.get('p', 1))
            if not page > 0:
                raise
        except:
            page = 1

        try:
            num = int(self.request.get('n', N))
            if num <= 0 or num > 200:
                raise
        except:
            num = N

        offset = (page-1) * num

        q = Commit.all()
        q.order('-__key__')
        results = q.fetch(num, offset)

        revs = [toRev(r) for r in results]
        builders = {}

        for r in revs:
            for b in r['builds']:
                builders[b['builder']] = builderInfo(b['builder'])

        for r in revs:
            have = set(x['builder'] for x in r['builds'])
            need = set(builders.keys()).difference(have)
            for n in need:
                r['builds'].append({'builder': n, 'log':'', 'ok': False})
            r['builds'].sort(cmp = byBuilder)

        builders = list(builders.items())
        builders.sort()
        values = {"revs": revs, "builders": [v for k,v in builders]}

        values['num'] = num
        values['prev'] = page - 1
        if len(results) == num:
            values['next'] = page + 1

        path = os.path.join(os.path.dirname(__file__), 'main.html')
        self.response.out.write(template.render(path, values))

class GetHighwater(webapp.RequestHandler):
    def get(self):
        builder = self.request.get('builder')
        key = 'todo-%s' % builder
        response = memcache.get(key)
        if response is None:
            # Fell out of memcache.  Rebuild from datastore results.
            # We walk the commit list looking for nodes that have not
            # been built by this builder and record the *parents* of those
            # nodes, because each builder builds the revision *after* the
            # one return (because we might not know about the latest
            # revision).
            q = Commit.all()
            q.order('-__key__')
            todo = []
            need = False
            first = None
            for c in q.fetch(N+1):
                if first is None:
                    first = c
                if need:
                    todo.append(c.node)
                need = not built(c, builder)
            if not todo:
                todo.append(first.node)
            response = ' '.join(todo)
            memcache.set(key, response, 3600)
        self.response.set_status(200)
        if self.request.get('all') != 'yes':
            response = response.split()[0]
        self.response.out.write(response)

def built(c, builder):
    for b in c.builds:
        if b.startswith(builder+'`'):
            return True
    return False

def auth(req):
    k = req.get('key')
    return k == hmac.new(key.accessKey, req.get('builder')).hexdigest() or k == key.accessKey
    
class SetHighwater(webapp.RequestHandler):
    def post(self):
        if not auth(self.request):
            self.response.set_status(403)
            return

        # Allow for old builders.
        # This is a no-op now: we figure out what to build based
        # on the current dashboard status.
        return

class LogHandler(webapp.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        hash = self.request.path[5:]
        l = CompressedLog.get_by_key_name(hash)
        if l is None:
            self.response.set_status(404)
            return
        log = bz2.decompress(l.log)
        self.response.set_status(200)
        self.response.out.write(log)

# Init creates the commit with id 0. Since this commit doesn't have a parent,
# it cannot be created by Build.
class Init(webapp.RequestHandler):
    def post(self):
        if not auth(self.request):
            self.response.set_status(403)
            return

        date = parseDate(self.request.get('date'))
        node = self.request.get('node')
        if not validNode(node) or date is None:
            logging.error("Not valid node ('%s') or bad date (%s %s)", node, date, self.request.get('date'))
            self.response.set_status(500)
            return

        commit = Commit(key_name = '00000000-%s' % node)
        commit.num = 0
        commit.node = node
        commit.parentnode = ''
        commit.user = self.request.get('user').encode('utf8')
        commit.date = date
        commit.desc = self.request.get('desc').encode('utf8')

        commit.put()

        self.response.set_status(200)

# Build is the main command: it records the result of a new build.
class Build(webapp.RequestHandler):
    def post(self):
        if not auth(self.request):
            self.response.set_status(403)
            return

        builder = self.request.get('builder')
        log = self.request.get('log').encode('utf-8')

        loghash = ''
        if len(log) > 0:
            loghash = hashlib.sha256(log).hexdigest()
            l = CompressedLog(key_name=loghash)
            l.log = bz2.compress(log)
            l.put()

        date = parseDate(self.request.get('date'))
        user = self.request.get('user').encode('utf8')
        desc = self.request.get('desc').encode('utf8')
        node = self.request.get('node')
        parenthash = self.request.get('parent')
        if not validNode(node) or not validNode(parenthash) or date is None:
            logging.error("Not valid node ('%s') or bad date (%s %s)", node, date, self.request.get('date'))
            self.response.set_status(500)
            return

        q = Commit.all()
        q.filter('node =', parenthash)
        parent = q.get()
        if parent is None:
            logging.error('Cannot find parent %s of node %s' % (parenthash, node))
            self.response.set_status(404)
            return
        parentnum, _ = parent.key().name().split('-', 1)
        nodenum = int(parentnum, 16) + 1

        key_name = '%08x-%s' % (nodenum, node)

        def add_build():
            n = Commit.get_by_key_name(key_name)
            if n is None:
                n = Commit(key_name = key_name)
                n.num = nodenum
                n.node = node
                n.parentnode = parenthash
                n.user = user
                n.date = date
                n.desc = desc
            s = '%s`%s' % (builder, loghash)
            for i, b in enumerate(n.builds):
                if b.split('`', 1)[0] == builder:
                    n.builds[i] = s
                    break
            else:
                n.builds.append(s)
            n.put()

        db.run_in_transaction(add_build)

        key = 'todo-%s' % builder
        memcache.delete(key)

        def mark_sent():
            n = Commit.get_by_key_name(key_name)
            n.fail_notification_sent = True
            n.put()

        n = Commit.get_by_key_name(key_name)
        if loghash and not failed(parent, builder) and not n.fail_notification_sent:
            subject = const.mail_fail_subject % (builder, desc.split("\n")[0])
            path = os.path.join(os.path.dirname(__file__), 'fail-notify.txt')
            body = template.render(path, {
                "builder": builder,
                "node": node[:12],
                "user": user,
                "desc": desc, 
                "loghash": loghash
            })
            mail.send_mail(
                sender=const.mail_from,
                reply_to=const.mail_fail_reply_to,
                to=const.mail_fail_to,
                subject=subject,
                body=body
            )
            db.run_in_transaction(mark_sent)

        self.response.set_status(200)

def failed(c, builder):
    for i, b in enumerate(c.builds):
        p = b.split('`', 1)
        if p[0] == builder:
            return len(p[1]) > 0
    return False

def node(num):
    q = Commit.all()
    q.filter('num =', num)
    n = q.get()
    return n

class FixedOffset(datetime.tzinfo):
    """Fixed offset in minutes east from UTC."""

    def __init__(self, offset):
        self.__offset = datetime.timedelta(seconds = offset)

    def utcoffset(self, dt):
        return self.__offset

    def tzname(self, dt):
        return None

    def dst(self, dt):
        return datetime.timedelta(0)

def validNode(node):
    if len(node) != 40:
        return False
    for x in node:
        o = ord(x)
        if (o < ord('0') or o > ord('9')) and (o < ord('a') or o > ord('f')):
            return False
    return True

def parseDate(date):
    if '-' in date:
        (a, offset) = date.split('-', 1)
        try:
            return datetime.datetime.fromtimestamp(float(a), FixedOffset(0-int(offset)))
        except ValueError:
            return None
    if '+' in date:
        (a, offset) = date.split('+', 1)
        try:
            return datetime.datetime.fromtimestamp(float(a), FixedOffset(int(offset)))
        except ValueError:
            return None
    try:
        return datetime.datetime.utcfromtimestamp(float(date))
    except ValueError:
        return None

email_re = re.compile('^[^<]+<([^>]*)>$')

def toUsername(user):
    r = email_re.match(user)
    if r is None:
        return user
    email = r.groups()[0]
    return email.replace('@golang.org', '')

def dateToShortStr(d):
    return d.strftime('%a %b %d %H:%M')

def parseBuild(build):
    [builder, logblob] = build.split('`')
    return {'builder': builder, 'log': logblob, 'ok': len(logblob) == 0}

def nodeInfo(c):
    return {
        "node": c.node,
        "user": toUsername(c.user),
        "date": dateToShortStr(c.date),
        "desc": c.desc,
        "shortdesc": c.desc.split('\n', 2)[0]
    }

def toRev(c):
    b = nodeInfo(c)
    b['builds'] = [parseBuild(build) for build in c.builds]
    return b

def byBuilder(x, y):
    return cmp(x['builder'], y['builder'])

# This is the URL map for the server. The first three entries are public, the
# rest are only used by the builders.
application = webapp.WSGIApplication(
                                     [('/', MainPage),
                                      ('/log/.*', LogHandler),
                                      ('/hw-get', GetHighwater),
                                      ('/hw-set', SetHighwater),

                                      ('/init', Init),
                                      ('/build', Build),
                                     ], debug=True)

def main():
    run_wsgi_app(application)

if __name__ == "__main__":
    main()

