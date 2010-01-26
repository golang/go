# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This is the server part of the continuous build system for Go. It must be run
# by AppEngine.

from google.appengine.ext import db
from google.appengine.ext import webapp
from google.appengine.ext.webapp import template
from google.appengine.ext.webapp.util import run_wsgi_app
import binascii
import datetime
import hashlib
import logging
import os
import re
import struct

import key

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
    # name> "`" <log hash>. If the log hash is empty, then the build was
    # successful.
    builds = db.StringListProperty()

class Benchmark(db.Model):
    name = db.StringProperty()

class BenchmarkResult(db.Model):
    num = db.IntegerProperty()
    builder = db.StringProperty()
    iterations = db.IntegerProperty()
    nsperop = db.IntegerProperty()

# A Log contains the textual build log of a failed build. The key name is the
# hex digest of the SHA256 hash of the contents.
class Log(db.Model):
    log = db.BlobProperty()

# For each builder, we store the last revision that it built. So, if it
# crashes, it knows where to start up from. The key names for these objects are
# "hw-" <builder name>
class Highwater(db.Model):
    commit = db.StringProperty()

class MainPage(webapp.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/html; charset=utf-8'

        q = Commit.all()
        q.order('-__key__')
        results = q.fetch(30)

        revs = [toRev(r) for r in results]
        allbuilders = set()

        for r in revs:
            for b in r['builds']:
                allbuilders.add(b['builder'])
        for r in revs:
            have = set(x['builder'] for x in r['builds'])
            need = allbuilders.difference(have)
            for n in need:
                r['builds'].append({'builder': n, 'log':'', 'ok': False})
            r['builds'].sort(cmp = byBuilder)

        builders = list(allbuilders)
        builders.sort()
        values = {"revs": revs, "builders": builders}

        path = os.path.join(os.path.dirname(__file__), 'main.html')
        self.response.out.write(template.render(path, values))

class GetHighwater(webapp.RequestHandler):
    def get(self):
        builder = self.request.get('builder')

        hw = Highwater.get_by_key_name('hw-%s' % builder)
        if hw is None:
            # If no highwater has been recorded for this builder, we find the
            # initial commit and return that.
            q = Commit.all()
            q.filter('num =', 0)
            commitzero = q.get()
            self.response.set_status(200)
            self.response.out.write(commitzero.node)
            return

        self.response.set_status(200)
        self.response.out.write(hw.commit)

class SetHighwater(webapp.RequestHandler):
    def post(self):
        if self.request.get('key') != key.accessKey:
            self.response.set_status(403)
            return

        builder = self.request.get('builder')
        newhw = self.request.get('hw')
        q = Commit.all()
        q.filter('node =', newhw)
        c = q.get()
        if c is None:
            self.response.set_status(404)
            return

        hw = Highwater(key_name = 'hw-%s' % builder)
        hw.commit = c.node
        hw.put()

class LogHandler(webapp.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        hash = self.request.path[5:]
        l = Log.get_by_key_name(hash)
        if l is None:
            self.response.set_status(404)
            return
        self.response.set_status(200)
        self.response.out.write(l.log)

# Init creates the commit with id 0. Since this commit doesn't have a parent,
# it cannot be created by Build.
class Init(webapp.RequestHandler):
    def post(self):
        if self.request.get('key') != key.accessKey:
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
        commit.user = self.request.get('user')
        commit.date = date
        commit.desc = self.request.get('desc').encode('utf8')

        commit.put()

        self.response.set_status(200)

# Build is the main command: it records the result of a new build.
class Build(webapp.RequestHandler):
    def post(self):
        if self.request.get('key') != key.accessKey:
            self.response.set_status(403)
            return

        builder = self.request.get('builder')
        log = self.request.get('log').encode('utf-8')

        loghash = ''
        if len(log) > 0:
            loghash = hashlib.sha256(log).hexdigest()
            l = Log(key_name = loghash)
            l.log = log
            l.put()

        date = parseDate(self.request.get('date'))
        node = self.request.get('node')
        parent = self.request.get('parent')
        if not validNode(node) or not validNode(parent) or date is None:
            logging.error("Not valid node ('%s') or bad date (%s %s)", node, date, self.request.get('date'))
            self.response.set_status(500)
            return

        q = Commit.all()
        q.filter('node =', parent)
        p = q.get()
        if p is None:
            logging.error('Cannot find parent %s of node %s' % (parent, node))
            self.response.set_status(404)
            return
        parentnum, _ = p.key().name().split('-', 1)
        nodenum = int(parentnum, 16) + 1

        def add_build():
            key_name = '%08x-%s' % (nodenum, node)
            n = Commit.get_by_key_name(key_name)
            if n is None:
                n = Commit(key_name = key_name)
                n.num = nodenum
                n.node = node
                n.parentnode = parent
                n.user = self.request.get('user')
                n.date = date
                n.desc = self.request.get('desc').encode('utf8')
            s = '%s`%s' % (builder, loghash)
            for i, b in enumerate(n.builds):
                if b.split('`', 1)[0] == builder:
                    n.builds[i] = s
                    break
            else:
                n.builds.append(s)
            n.put()

        db.run_in_transaction(add_build)

        hw = Highwater.get_by_key_name('hw-%s' % builder)
        if hw is None:
            hw = Highwater(key_name = 'hw-%s' % builder)
        hw.commit = node
        hw.put()

        self.response.set_status(200)

class Benchmarks(webapp.RequestHandler):
    def get(self):
        q = Benchmark.all()
        bs = q.fetch(10000)

        self.response.set_status(200)
        self.response.headers['Content-Type'] = 'application/json; charset=utf-8'
        self.response.out.write('{"benchmarks": [\n')

        first = True
        for b in bs:
            if not first:
                self.response.out.write(',"' + b.name + '"\n')
            else:
                self.response.out.write('"' + b.name + '"\n')
                first = False
        self.response.out.write(']}\n')

    def post(self):
        if self.request.get('key') != key.accessKey:
            self.response.set_status(403)
            return

        builder = self.request.get('builder')
        node = self.request.get('node')
        if not validNode(node):
            logging.error("Not valid node ('%s')", node)
            self.response.set_status(500)
            return

        benchmarkdata = self.request.get('benchmarkdata')
        benchmarkdata = binascii.a2b_base64(benchmarkdata)

        def get_string(i):
            l, = struct.unpack('>H', i[:2])
            s = i[2:2+l]
            if len(s) != l:
                return None, None
            return s, i[2+l:]

        benchmarks = {}
        while len(benchmarkdata) > 0:
            name, benchmarkdata = get_string(benchmarkdata)
            iterations_str, benchmarkdata = get_string(benchmarkdata)
            time_str, benchmarkdata = get_string(benchmarkdata)
            iterations = int(iterations_str)
            time = int(time_str)

            benchmarks[name] = (iterations, time)

        q = Commit.all()
        q.filter('node =', node)
        n = q.get()
        if n is None:
            logging.error('Client asked for unknown commit while uploading benchmarks')
            self.response.set_status(404)
            return

        for (benchmark, (iterations, time)) in benchmarks.items():
            b = Benchmark.get_or_insert(benchmark.encode('base64'), name = benchmark)
            r = BenchmarkResult(key_name = '%08x/builder' % n.num, parent = b, num = n.num, iterations = iterations, nsperop = time, builder = builder)
            r.put()

        self.response.set_status(200)

class GetBenchmarks(webapp.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'application/json; charset=utf-8'
        benchmark = self.request.path[12:].decode('hex').encode('base64')

        b = Benchmark.get_by_key_name(benchmark)
        if b is None:
            self.response.set_status(404)
            return

        q = BenchmarkResult.all()
        q.ancestor(b)
        q.order('-__key__')
        results = q.fetch(10000)

        if len(results) == 0:
            self.response.set_status(404)
            return

        max = -1
        min = 2000000000
        builders = set()
        for r in results:
            if max < r.num:
                max = r.num
            if min > r.num:
                min = r.num
            builders.add(r.builder)

        res = {}
        for b in builders:
            res[b] = [[-1] * ((max - min) + 1), [-1] * ((max - min) + 1)]

        for r in results:
            res[r.builder][0][r.num - min] = r.iterations
            res[r.builder][1][r.num - min] = r.nsperop

        self.response.out.write(str(res))

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

def toRev(c):
        b = { "node": c.node,
              "user": toUsername(c.user),
              "date": dateToShortStr(c.date),
              "desc": c.desc}
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
                                      ('/benchmarks', Benchmarks),
                                      ('/benchmarks/.*', GetBenchmarks),
                                     ])

def main():
    run_wsgi_app(application)

if __name__ == "__main__":
    main()
