# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This is the server part of the continuous build system for Go. It must be run
# by AppEngine.

from django.utils import simplejson
from google.appengine.api import mail
from google.appengine.api import memcache
from google.appengine.ext import db
from google.appengine.ext import webapp
from google.appengine.ext.webapp import template
from google.appengine.ext.webapp.util import run_wsgi_app
import datetime
import hashlib
import logging
import os
import re
import bz2

# local imports
from auth import auth
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
#
# N.B. user is a StringProperty, so it must be type 'unicode'.
# desc is a BlobProperty, so it must be type 'string'.  [sic]
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

# A CompressedLog contains the textual build log of a failed build. 
# The key name is the hex digest of the SHA256 hash of the contents.
# The contents is bz2 compressed.
class CompressedLog(db.Model):
    log = db.BlobProperty()

N = 30

def builderInfo(b):
    f = b.split('-', 3)
    if len(f) < 2:
      f.append(None)
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
                if b['builder'] in builders:
                    continue
                bi = builderInfo(b['builder'])
                builders[b['builder']] = bi
        bad_builders = [key for key in builders if not builders[key]['goarch']]
        for key in bad_builders:
            del builders[key]
        for r in revs:
            r['builds'] = [b for b in r['builds'] if b['builder'] not in bad_builders]

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

        values['bad'] = bad_builders
        path = os.path.join(os.path.dirname(__file__), 'main.html')
        self.response.out.write(template.render(path, values))

# A DashboardHandler is a webapp.RequestHandler but provides
#    authenticated_post - called by post after authenticating
#    json - writes object in json format to response output
class DashboardHandler(webapp.RequestHandler):
    def post(self):
        if not auth(self.request):
            self.response.set_status(403)
            return
        self.authenticated_post()

    def authenticated_post(self):
        return
    
    def json(self, obj):
        self.response.set_status(200)
        simplejson.dump(obj, self.response.out)
        return

# Todo serves /todo.  It tells the builder which commits need to be built.
class Todo(DashboardHandler):
    def get(self):
        builder = self.request.get('builder')
        key = 'todo-%s' % builder
        response = memcache.get(key)
        if response is None:
            # Fell out of memcache.  Rebuild from datastore results.
            # We walk the commit list looking for nodes that have not
            # been built by this builder.
            q = Commit.all()
            q.order('-__key__')
            todo = []
            first = None
            for c in q.fetch(N+1):
                if first is None:
                    first = c
                if not built(c, builder):
                    todo.append({'Hash': c.node})
            response = simplejson.dumps(todo)
            memcache.set(key, response, 3600)
        self.response.set_status(200)
        self.response.out.write(response)

def built(c, builder):
    for b in c.builds:
        if b.startswith(builder+'`'):
            return True
    return False

# Log serves /log/.  It retrieves log data by content hash.
class LogHandler(DashboardHandler):
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
class Init(DashboardHandler):
    def authenticated_post(self):
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

# The last commit when we switched to using entity groups.
# This is the root of the new commit entity group.
RootCommitKeyName = '00000f26-f32c6f1038207c55d5780231f7484f311020747e'

# CommitHandler serves /commit.
# A GET of /commit retrieves information about the specified commit.
# A POST of /commit creates a node for the given commit.
# If the commit already exists, the POST silently succeeds (like mkdir -p).
class CommitHandler(DashboardHandler):
    def get(self):
        node = self.request.get('node')
        if not validNode(node):
            return self.json({'Status': 'FAIL', 'Error': 'malformed node hash'})
        n = nodeByHash(node)
        if n is None:
            return self.json({'Status': 'FAIL', 'Error': 'unknown revision'})
        return self.json({'Status': 'OK', 'Node': nodeObj(n)})

    def authenticated_post(self):
        # Require auth with the master key, not a per-builder key.
        if self.request.get('builder'):
            self.response.set_status(403)
            return

        node = self.request.get('node')
        date = parseDate(self.request.get('date'))
        user = self.request.get('user')
        desc = self.request.get('desc').encode('utf8')
        parenthash = self.request.get('parent')

        if not validNode(node) or not validNode(parenthash) or date is None:
            return self.json({'Status': 'FAIL', 'Error': 'malformed node, parent, or date'})

        n = nodeByHash(node)
        if n is None:
            p = nodeByHash(parenthash)
            if p is None:
                return self.json({'Status': 'FAIL', 'Error': 'unknown parent'})

            # Want to create new node in a transaction so that multiple
            # requests creating it do not collide and so that multiple requests
            # creating different nodes get different sequence numbers.
            # All queries within a transaction must include an ancestor,
            # but the original datastore objects we used for the dashboard
            # have no common ancestor.  Instead, we use a well-known
            # root node - the last one before we switched to entity groups -
            # as the as the common ancestor.
            root = Commit.get_by_key_name(RootCommitKeyName)

            def add_commit():
                if nodeByHash(node, ancestor=root) is not None:
                    return

                # Determine number for this commit.
                # Once we have created one new entry it will be lastRooted.num+1,
                # but the very first commit created in this scheme will have to use
                # last.num's number instead (last is likely not rooted).
                q = Commit.all()
                q.order('-__key__')
                q.ancestor(root)
                last = q.fetch(1)[0]
                num = last.num+1

                n = Commit(key_name = '%08x-%s' % (num, node), parent = root)
                n.num = num
                n.node = node
                n.parentnode = parenthash
                n.user = user
                n.date = date
                n.desc = desc
                n.put()
            db.run_in_transaction(add_commit)
            n = nodeByHash(node)
            if n is None:
                return self.json({'Status': 'FAIL', 'Error': 'failed to create commit node'})

        return self.json({'Status': 'OK', 'Node': nodeObj(n)})

# Build serves /build.
# A POST to /build records a new build result.
class Build(webapp.RequestHandler):
    def post(self):
        if not auth(self.request):
            self.response.set_status(403)
            return

        builder = self.request.get('builder')
        log = self.request.get('log').encode('utf8')

        loghash = ''
        if len(log) > 0:
            loghash = hashlib.sha256(log).hexdigest()
            l = CompressedLog(key_name=loghash)
            l.log = bz2.compress(log)
            l.put()

        node = self.request.get('node')
        if not validNode(node):
            logging.error('Invalid node %s' % (node))
            self.response.set_status(500)
            return

        n = nodeByHash(node)
        if n is None:
            logging.error('Cannot find node %s' % (node))
            self.response.set_status(404)
            return
        nn = n

        def add_build():
            n = nodeByHash(node, ancestor=nn)
            if n is None:
                logging.error('Cannot find hash in add_build: %s %s' % (builder, node))
                return

            s = '%s`%s' % (builder, loghash)
            for i, b in enumerate(n.builds):
                if b.split('`', 1)[0] == builder:
                    # logging.error('Found result for %s %s already' % (builder, node))
                    n.builds[i] = s
                    break
            else:
                # logging.error('Added result for %s %s' % (builder, node))
                n.builds.append(s)
            n.put()

        db.run_in_transaction(add_build)

        key = 'todo-%s' % builder
        memcache.delete(key)

        c = getBrokenCommit(node, builder)
        if c is not None and not c.fail_notification_sent:
            notifyBroken(c, builder, log)

        self.response.set_status(200)


def getBrokenCommit(node, builder):
    """
    getBrokenCommit returns a Commit that breaks the build.
    The Commit will be either the one specified by node or the one after.
    """

    # Squelch mail if already fixed.
    head = firstResult(builder)
    if broken(head, builder) == False:
        return

    # Get current node and node before, after.
    cur = nodeByHash(node)
    if cur is None:
        return
    before = nodeBefore(cur)
    after = nodeAfter(cur)

    if broken(before, builder) == False and broken(cur, builder):
        return cur
    if broken(cur, builder) == False and broken(after, builder):
        return after

    return

def firstResult(builder):
    q = Commit.all().order('-__key__')
    for c in q.fetch(20):
        for i, b in enumerate(c.builds):
            p = b.split('`', 1)
            if p[0] == builder:
                return c
    return None

def nodeBefore(c):
    return nodeByHash(c.parentnode)

def nodeAfter(c):
    return Commit.all().filter('parenthash', c.node).get()

def notifyBroken(c, builder, log):
    def send():
        n = Commit.get(c.key())
        if n is None:
            logging.error("couldn't retrieve Commit '%s'" % c.key())
            return False
        if n.fail_notification_sent:
            return False
        n.fail_notification_sent = True
        return n.put()
    if not db.run_in_transaction(send):
        return

    # get last 100 lines of the build log
    log = '\n'.join(log.split('\n')[-100:])

    subject = const.mail_fail_subject % (builder, c.desc.split('\n')[0])
    path = os.path.join(os.path.dirname(__file__), 'fail-notify.txt')
    body = template.render(path, {
        "builder": builder,
        "node": c.node,
        "user": c.user,
        "desc": c.desc,
        "loghash": logHash(c, builder),
        "log": log,
    })
    mail.send_mail(
        sender=const.mail_from,
        to=const.mail_fail_to,
        subject=subject,
        body=body
    )

def logHash(c, builder):
    for i, b in enumerate(c.builds):
        p = b.split('`', 1)
        if p[0] == builder:
            return p[1]
    return ""

def broken(c, builder):
    """
    broken returns True if commit c breaks the build for the specified builder,
    False if it is a good build, and None if no results exist for this builder.
    """
    if c is None:
        return None
    for i, b in enumerate(c.builds):
        p = b.split('`', 1)
        if p[0] == builder:
            return len(p[1]) > 0
    return None

def node(num):
    q = Commit.all()
    q.filter('num =', num)
    n = q.get()
    return n

def nodeByHash(hash, ancestor=None):
    q = Commit.all()
    q.filter('node =', hash)
    if ancestor is not None:
      q.ancestor(ancestor)
    n = q.get()
    return n

# nodeObj returns a JSON object (ready to be passed to simplejson.dump) describing node.
def nodeObj(n):
    return {
        'Hash': n.node,
        'ParentHash': n.parentnode,
        'User': n.user,
        'Date': n.date.strftime('%Y-%m-%d %H:%M %z'),
        'Desc': n.desc,
    }

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

# Give old builders work; otherwise they pound on the web site.
class Hwget(DashboardHandler):
    def get(self):
        self.response.out.write("8000\n")

# This is the URL map for the server. The first three entries are public, the
# rest are only used by the builders.
application = webapp.WSGIApplication(
                                     [('/', MainPage),
                                      ('/hw-get', Hwget),
                                      ('/log/.*', LogHandler),
                                      ('/commit', CommitHandler),
                                      ('/init', Init),
                                      ('/todo', Todo),
                                      ('/build', Build),
                                     ], debug=True)

def main():
    run_wsgi_app(application)

if __name__ == "__main__":
    main()

