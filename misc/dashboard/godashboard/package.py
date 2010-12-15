# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This is the server part of the package dashboard.
# It must be run by App Engine.

mail_to      = "adg@golang.org"
mail_from    = "Go Dashboard <adg@golang.org>"
mail_subject = "New Project Submitted"

from google.appengine.api import memcache
from google.appengine.runtime import DeadlineExceededError
from google.appengine.ext import db
from google.appengine.ext import webapp
from google.appengine.ext.webapp import template
from google.appengine.ext.webapp.util import run_wsgi_app
from google.appengine.api import users
from google.appengine.api import mail
from google.appengine.api import urlfetch
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
import sets

# local imports
import toutf8

template.register_template_library('toutf8')

# Storage model for package info recorded on server.
# Just path, count, and time of last install.
class Package(db.Model):
    path = db.StringProperty()
    web_url = db.StringProperty()  # derived from path
    count = db.IntegerProperty()
    last_install = db.DateTimeProperty()

class Project(db.Model):
    name = db.StringProperty(indexed=True)
    descr = db.StringProperty()
    web_url = db.StringProperty()
    package = db.ReferenceProperty(Package)
    category = db.StringProperty(indexed=True)
    tags = db.ListProperty(str)
    approved = db.BooleanProperty(indexed=True)

re_bitbucket = re.compile(r'^bitbucket\.org/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+$')
re_googlecode = re.compile(r'^[a-z0-9\-]+\.googlecode\.com/(svn|hg)$')
re_github = re.compile(r'^github\.com/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+$')

def vc_to_web(path):
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
        return False, False
    return web, check_url

re_bitbucket_web = re.compile(r'bitbucket\.org/([a-z0-9A-Z_.\-]+)/([a-z0-9A-Z_.\-]+)')
re_googlecode_web = re.compile(r'code.google.com/p/([a-z0-9\-]+)')
re_github_web = re.compile(r'github\.com/([a-z0-9A-Z_.\-]+)/([a-z0-9A-Z_.\-]+)')
re_striphttp = re.compile(r'http://(www\.)?')

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
    return False

MaxPathLength = 100
CacheTimeout = 3600

class PackagePage(webapp.RequestHandler):
    def get(self):
        if self.request.get('fmt') == 'json':
            return self.json()

        html = memcache.get('view-package')
        if not html:
            q = Package.all()
            q.order('-last_install')
            by_time = q.fetch(100)

            q = Package.all()
            q.order('-count')
            by_count = q.fetch(100)

            self.response.headers['Content-Type'] = 'text/html; charset=utf-8'
            path = os.path.join(os.path.dirname(__file__), 'package.html')
            html = template.render(
                path, 
                {"by_time": by_time, "by_count": by_count}
            )
            memcache.set('view-package', html, time=CacheTimeout)

        self.response.out.write(html)

    def json(self):
        json = memcache.get('view-package-json')
        if not json:
            self.response.set_status(200)
            self.response.headers['Content-Type'] = 'text/plain; charset=utf-8'
            q = Package.all()
            s = '{"packages": ['
            sep = ''
            for r in q.fetch(1000):
                s += '%s\n\t{"path": "%s", "last_install": "%s", "count": "%s"}' % (sep, r.path, r.last_install, r.count)
                sep = ','
            s += '\n]}\n'
            json = s
            memcache.set('view-package-json', json, time=CacheTimeoout)
        self.response.out.write(json)

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
            web, check_url = vc_to_web(path)
            if not web:
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
                sender=mail_from, to=mail_to, subject=mail_subject,
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
        ('/project.*', ProjectPage),
        ], debug=True)
    run_wsgi_app(app)

if __name__ == '__main__':
    main()
