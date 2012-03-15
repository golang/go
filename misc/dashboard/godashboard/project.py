# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from google.appengine.api import mail
from google.appengine.api import memcache
from google.appengine.api import users
from google.appengine.ext import db
from google.appengine.ext import webapp
from google.appengine.ext.webapp import template
from google.appengine.ext.webapp.util import run_wsgi_app
import os
import sets

# local imports
import toutf8
import const

template.register_template_library('toutf8')

class Project(db.Model):
    name = db.StringProperty(indexed=True)
    descr = db.StringProperty()
    web_url = db.StringProperty()
    package = db.ReferenceProperty(Package)
    category = db.StringProperty(indexed=True)
    tags = db.ListProperty(str)
    approved = db.BooleanProperty(indexed=True)

CacheTimeout = 3600

class ProjectPage(webapp.RequestHandler):

    def get(self):
        admin = users.is_current_user_admin()
        if self.request.path == "/project/login":
            self.redirect(users.create_login_url("/project"))
        elif self.request.path == "/project/edit" and admin:
            self.edit()
        else:
            self.list()

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
        ('/.*', ProjectPage),
        ], debug=True)
    run_wsgi_app(app)

if __name__ == '__main__':
    main()
