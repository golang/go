# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This is the server part of the package dashboard.
# It must be run by App Engine.

from google.appengine.api import mail
from google.appengine.api import memcache
from google.appengine.api import taskqueue
from google.appengine.api import urlfetch
from google.appengine.api import users
from google.appengine.ext import db
from google.appengine.ext import webapp
from google.appengine.ext.webapp import template
from google.appengine.ext.webapp.util import run_wsgi_app
import datetime
import logging
import os
import re
import sets
import urllib2

# local imports
from auth import auth
import toutf8
import const

template.register_template_library('toutf8')

# Storage model for package info recorded on server.
class Package(db.Model):
    path = db.StringProperty()
    web_url = db.StringProperty()           # derived from path
    count = db.IntegerProperty()            # grand total
    week_count = db.IntegerProperty()       # rolling weekly count
    day_count = db.TextProperty(default='') # daily count
    last_install = db.DateTimeProperty()

    # data contributed by gobuilder
    info = db.StringProperty()  
    ok = db.BooleanProperty()
    last_ok = db.DateTimeProperty()

    def get_day_count(self):
        counts = {}
        if not self.day_count:
            return counts
        for d in str(self.day_count).split('\n'):
            date, count = d.split(' ')
            counts[date] = int(count)
        return counts

    def set_day_count(self, count):
        days = []
        for day, count in count.items():
            days.append('%s %d' % (day, count))
        days.sort(reverse=True)
        days = days[:28]
        self.day_count = '\n'.join(days)

    def inc(self):
        count = self.get_day_count()
        today = str(datetime.date.today())
        count[today] = count.get(today, 0) + 1
        self.set_day_count(count)
        self.update_week_count(count)
        self.count += 1

    def update_week_count(self, count=None):
        if count is None:
            count = self.get_day_count()
        total = 0
        today = datetime.date.today()
        for i in range(7):
            day = str(today - datetime.timedelta(days=i))
            if day in count:
                total += count[day]
        self.week_count = total


# PackageDaily kicks off the daily package maintenance cron job
# and serves the associated task queue.
class PackageDaily(webapp.RequestHandler):

    def get(self):
        # queue a task to update each package with a week_count > 0
        keys = Package.all(keys_only=True).filter('week_count >', 0)
        for key in keys:
            taskqueue.add(url='/package/daily', params={'key': key.name()})

    def post(self):
        # update a single package (in a task queue)
        def update(key):
            p = Package.get_by_key_name(key)
            if not p:
                return
            p.update_week_count()
            p.put()
        key = self.request.get('key')
        if not key:
            return
        db.run_in_transaction(update, key)
 

class Project(db.Model):
    name = db.StringProperty(indexed=True)
    descr = db.StringProperty()
    web_url = db.StringProperty()
    package = db.ReferenceProperty(Package)
    category = db.StringProperty(indexed=True)
    tags = db.ListProperty(str)
    approved = db.BooleanProperty(indexed=True)


re_bitbucket = re.compile(r'^(bitbucket\.org/[a-z0-9A-Z_.\-]+/[a-zA-Z0-9_.\-]+)(/[a-z0-9A-Z_.\-/]+)?$')
re_googlecode = re.compile(r'^[a-z0-9\-]+\.googlecode\.com/(svn|hg)(/[a-z0-9A-Z_.\-/]+)?$')
re_github = re.compile(r'^github\.com/[a-z0-9A-Z_.\-]+(/[a-z0-9A-Z_.\-]+)+$')
re_launchpad = re.compile(r'^launchpad\.net/([a-z0-9A-Z_.\-]+(/[a-z0-9A-Z_.\-]+)?|~[a-z0-9A-Z_.\-]+/(\+junk|[a-z0-9A-Z_.\-]+)/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]+)?$')

def vc_to_web(path):
    if re_bitbucket.match(path):
        m = re_bitbucket.match(path)
        check_url = 'http://' + m.group(1) + '/?cmd=heads'
        web = 'http://' + m.group(1) + '/'
    elif re_github.match(path):
        m = re_github_web.match(path)
        check_url = 'https://raw.github.com/' + m.group(1) + '/' + m.group(2) + '/master/'
        web = 'http://github.com/' + m.group(1) + '/' + m.group(2) + '/'
    elif re_googlecode.match(path):
        m = re_googlecode.match(path)
        check_url = 'http://'+path
        if not m.group(2):  # append / after bare '/hg'
            check_url += '/'
        web = 'http://code.google.com/p/' + path[:path.index('.')]
    elif re_launchpad.match(path):
        check_url = web = 'https://'+path
    else:
        return False, False
    return web, check_url

re_bitbucket_web = re.compile(r'bitbucket\.org/([a-z0-9A-Z_.\-]+)/([a-z0-9A-Z_.\-]+)')
re_googlecode_web = re.compile(r'code.google.com/p/([a-z0-9\-]+)')
re_github_web = re.compile(r'github\.com/([a-z0-9A-Z_.\-]+)/([a-z0-9A-Z_.\-]+)')
re_launchpad_web = re.compile(r'launchpad\.net/([a-z0-9A-Z_.\-]+(/[a-z0-9A-Z_.\-]+)?|~[a-z0-9A-Z_.\-]+/(\+junk|[a-z0-9A-Z_.\-]+)/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]+)?')
re_striphttp = re.compile(r'https?://(www\.)?')

def web_to_vc(url):
    url = re_striphttp.sub('', url)
    m = re_bitbucket_web.match(url)
    if m:
        return 'bitbucket.org/'+m.group(1)+'/'+m.group(2)
    m = re_github_web.match(url)
    if m:
        return 'github.com/'+m.group(1)+'/'+m.group(2)
    m = re_googlecode_web.match(url)
    if m:
        path = m.group(1)+'.googlecode.com/'
        # perform http request to path/hg to check if they're using mercurial
        vcs = 'svn'
        try:
            response = urlfetch.fetch('http://'+path+'hg', deadline=1)
            if response.status_code == 200:
                vcs = 'hg'
        except: pass
        return path + vcs
    m = re_launchpad_web.match(url)
    if m:
        return m.group(0)
    return False

MaxPathLength = 100
CacheTimeout = 3600

class PackagePage(webapp.RequestHandler):
    def get(self):
        if self.request.get('fmt') == 'json':
            return self.json()

        html = memcache.get('view-package')
        if not html:
            tdata = {}

            q = Package.all().filter('week_count >', 0)
            q.order('-week_count')
            tdata['by_week_count'] = q.fetch(50)

            q = Package.all()
            q.order('-last_install')
            tdata['by_time'] = q.fetch(20)

            q = Package.all()
            q.order('-count')
            tdata['by_count'] = q.fetch(100)

            path = os.path.join(os.path.dirname(__file__), 'package.html')
            html = template.render(path, tdata)
            memcache.set('view-package', html, time=CacheTimeout)

        self.response.headers['Content-Type'] = 'text/html; charset=utf-8'
        self.response.out.write(html)

    def json(self):
        json = memcache.get('view-package-json')
        if not json:
            q = Package.all()
            s = '{"packages": ['
            sep = ''
            for r in q.fetch(1000):
                s += '%s\n\t{"path": "%s", "last_install": "%s", "count": "%s"}' % (sep, r.path, r.last_install, r.count)
                sep = ','
            s += '\n]}\n'
            json = s
            memcache.set('view-package-json', json, time=CacheTimeout)
        self.response.set_status(200)
        self.response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        self.response.out.write(json)

    def can_get_url(self, url):
        try:
            urllib2.urlopen(urllib2.Request(url))
            return True
        except:
            return False

    def is_valid_package_path(self, path):
        return (re_bitbucket.match(path) or
            re_googlecode.match(path) or
            re_github.match(path) or
            re_launchpad.match(path))

    def record_pkg(self, path):
        # sanity check string
        if not path or len(path) > MaxPathLength or not self.is_valid_package_path(path):
            return False

        # look in datastore
        key = 'pkg-' + path
        p = Package.get_by_key_name(key)
        if p is None:
            # not in datastore - verify URL before creating
            web, check_url = vc_to_web(path)
            if not web:
                logging.error('unrecognized path: %s', path)
                return False
            if not self.can_get_url(check_url):
                logging.error('cannot get %s', check_url)
                return False
            p = Package(key_name = key, path = path, count = 0, web_url = web)

        if auth(self.request):
            # builder updating package metadata
            p.info = self.request.get('info')
            p.ok = self.request.get('ok') == "true"
            if p.ok:
                p.last_ok = datetime.datetime.utcnow()
        else:
            # goinstall reporting an install
            p.inc()
            p.last_install = datetime.datetime.utcnow()

        # update package object
        p.put()
        return True

    def post(self):
        path = self.request.get('path')
        ok = db.run_in_transaction(self.record_pkg, path)
        if ok:
            self.response.set_status(200)
            self.response.out.write('ok')
        else:
            logging.error('invalid path in post: %s', path)
            self.response.set_status(500)
            self.response.out.write('not ok')

class ProjectPage(webapp.RequestHandler):

    def get(self):
        admin = users.is_current_user_admin()
        if self.request.path == "/project/login":
            self.redirect(users.create_login_url("/project"))
        elif self.request.path == "/project/logout":
            self.redirect(users.create_logout_url("/project"))
        elif self.request.path == "/project/edit" and admin:
            self.edit()
        elif self.request.path == "/project/assoc" and admin:
            self.assoc()
        else:
            self.list()

    def assoc(self):
        projects = Project.all()
        for p in projects:
            if p.package:
                continue
            path = web_to_vc(p.web_url)
            if not path:
                continue
            pkg = Package.get_by_key_name("pkg-"+path)
            if not pkg:
                self.response.out.write('no: %s %s<br>' % (p.web_url, path))
                continue
            p.package = pkg
            p.put()
            self.response.out.write('yes: %s %s<br>' % (p.web_url, path))

    def post(self):
        if self.request.path == "/project/edit":
            self.edit(True)
        else:
            data = dict(map(lambda x: (x, self.request.get(x)), ["name","descr","web_url"]))
            if reduce(lambda x, y: x or not y, data.values(), False):
                data["submitMsg"] = "You must complete all the fields."
                self.list(data)
                return
            p = Project.get_by_key_name("proj-"+data["name"])
            if p is not None:
                data["submitMsg"] = "A project by this name already exists."
                self.list(data)
                return
            p = Project(key_name="proj-"+data["name"], **data)
            p.put()
		
            path = os.path.join(os.path.dirname(__file__), 'project-notify.txt')
            mail.send_mail(
                sender=const.mail_from,
                to=const.mail_submit_to,
                subject=const.mail_submit_subject,
                body=template.render(path, {'project': p}))

            self.list({"submitMsg": "Your project has been submitted."})

    def list(self, additional_data={}):
        cache_key = 'view-project-data'
        tag = self.request.get('tag', None)
        if tag:
            cache_key += '-'+tag
        data = memcache.get(cache_key)
        admin = users.is_current_user_admin()
        if admin or not data:
            projects = Project.all().order('category').order('name')
            if not admin:
                projects = projects.filter('approved =', True)
            projects = list(projects)

            tags = sets.Set()
            for p in projects:
                for t in p.tags:
                    tags.add(t)

            if tag:
                projects = filter(lambda x: tag in x.tags, projects)

            data = {}
            data['tag'] = tag
            data['tags'] = tags
            data['projects'] = projects 
            data['admin']= admin
            if not admin:
                memcache.set(cache_key, data, time=CacheTimeout)

        for k, v in additional_data.items():
            data[k] = v

        self.response.headers['Content-Type'] = 'text/html; charset=utf-8'
        path = os.path.join(os.path.dirname(__file__), 'project.html')
        self.response.out.write(template.render(path, data))

    def edit(self, save=False):
        if save:
            name = self.request.get("orig_name")
        else:
            name = self.request.get("name")

        p = Project.get_by_key_name("proj-"+name)
        if not p:
            self.response.out.write("Couldn't find that Project.")
            return

        if save:
            if self.request.get("do") == "Delete":
                p.delete()
            else:
                pkg_name = self.request.get("package", None)
                if pkg_name:
                    pkg = Package.get_by_key_name("pkg-"+pkg_name)
                    if pkg:
                        p.package = pkg.key()
                for f in ['name', 'descr', 'web_url', 'category']:
                    setattr(p, f, self.request.get(f, None))
                p.approved = self.request.get("approved") == "1"
                p.tags = filter(lambda x: x, self.request.get("tags", "").split(","))
                p.put()
            memcache.delete('view-project-data')
            self.redirect('/project')
            return

        # get all project categories and tags
        cats, tags = sets.Set(), sets.Set()
        for r in Project.all():
            cats.add(r.category)
            for t in r.tags:
                tags.add(t)

        self.response.headers['Content-Type'] = 'text/html; charset=utf-8'
        path = os.path.join(os.path.dirname(__file__), 'project-edit.html')
        self.response.out.write(template.render(path, { 
            "taglist": tags, "catlist": cats, "p": p, "tags": ",".join(p.tags) }))

    def redirect(self, url):
        self.response.set_status(302)
        self.response.headers.add_header("Location", url)

def main():
    app = webapp.WSGIApplication([
        ('/package', PackagePage),
        ('/package/daily', PackageDaily),
        ('/project.*', ProjectPage),
        ], debug=True)
    run_wsgi_app(app)

if __name__ == '__main__':
    main()
