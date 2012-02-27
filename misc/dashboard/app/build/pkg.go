// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"net/http"
	"regexp"
	"strings"

	"appengine"
	"appengine/datastore"
	"appengine/delay"
	"appengine/urlfetch"
)

func init() {
	http.HandleFunc("/install", installHandler)
	http.HandleFunc("/install/cron", installCronHandler)
}

// installHandler serves requests from the go tool to increment the install
// count for a given package.
func installHandler(w http.ResponseWriter, r *http.Request) {
	installLater.Call(appengine.NewContext(r), r.FormValue("packagePath"))
}

// installCronHandler starts a task to update the weekly install counts for
// every external package.
func installCronHandler(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)
	q := datastore.NewQuery("Package").Filter("Kind=", "external").KeysOnly()
	for t := q.Run(c); ; {
		key, err := t.Next(nil)
		if err == datastore.Done {
			break
		} else if err != nil {
			c.Errorf("%v", err)
			return
		}
		updateWeeklyLater.Call(c, key)
	}
}

var (
	installLater      = delay.Func("install", install)
	updateWeeklyLater = delay.Func("updateWeekly", updateWeekly)
)

// install validates the provided package path, increments its install count,
// and creates the Package record if it doesn't exist.
func install(c appengine.Context, path string) {
	if !validPath(c, path) {
		return
	}
	tx := func(c appengine.Context) error {
		p := &Package{Path: path, Kind: "external"}
		err := datastore.Get(c, p.Key(c), p)
		if err != nil && err != datastore.ErrNoSuchEntity {
			return err
		}
		p.IncrementInstalls()
		_, err = datastore.Put(c, p.Key(c), p)
		return err
	}
	if err := datastore.RunInTransaction(c, tx, nil); err != nil {
		c.Errorf("install(%q): %v", path, err)
	}
}

// updateWeekly updates the weekly count for the specified Package.
func updateWeekly(c appengine.Context, key *datastore.Key) {
	tx := func(c appengine.Context) error {
		p := new(Package)
		if err := datastore.Get(c, key, p); err != nil {
			return err
		}
		p.UpdateInstallsThisWeek()
		_, err := datastore.Put(c, key, p)
		return err
	}
	if err := datastore.RunInTransaction(c, tx, nil); err != nil {
		c.Errorf("updateWeekly: %v", err)
	}
}

// validPath validates the specified import path by matching it against the
// vcsPath regexen and validating its existence by making an HTTP GET request
// to the remote repository.
func validPath(c appengine.Context, path string) bool {
	for _, p := range vcsPaths {
		if !strings.HasPrefix(path, p.prefix) {
			continue
		}
		m := p.regexp.FindStringSubmatch(path)
		if m == nil {
			continue
		}
		if p.check == nil {
			// no check function, so just say OK
			return true
		}
		match := make(map[string]string)
		for i, name := range p.regexp.SubexpNames() {
			if name != "" {
				match[name] = m[i]
			}
		}
		return p.check(c, match)
	}
	c.Debugf("validPath(%q): matching vcsPath not found", path)
	return false
}

// A vcsPath describes how to convert an import path into a version control
// system and repository name.
//
// This is a cut down and modified version of the data structure from
// $GOROOT/src/cmd/go/vcs.go.
type vcsPath struct {
	prefix string // prefix this description applies to
	re     string // pattern for import path

	// check should perform an HTTP request to validate the import path
	check func(c appengine.Context, match map[string]string) bool

	regexp *regexp.Regexp // cached compiled form of re
}

// vcsPaths lists the known vcs paths.
//
// This is a cut down version of the data from $GOROOT/src/cmd/go/vcs.go.
var vcsPaths = []*vcsPath{
	// Google Code - new syntax
	{
		prefix: "code.google.com/",
		re:     `^(?P<root>code\.google\.com/p/(?P<project>[a-z0-9\-]+)(\.(?P<subrepo>[a-z0-9\-]+))?)(/[A-Za-z0-9_.\-]+)*$`,
		check:  googleCodeVCS,
	},

	// Github
	{
		prefix: "github.com/",
		re:     `^(?P<root>github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)(/[A-Za-z0-9_.\-]+)*$`,
		check:  checkRoot,
	},

	// Bitbucket
	{
		prefix: "bitbucket.org/",
		re:     `^(?P<root>bitbucket\.org/(?P<bitname>[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+))(/[A-Za-z0-9_.\-]+)*$`,
		check:  checkRoot,
	},

	// Launchpad
	{
		prefix: "launchpad.net/",
		re:     `^(?P<root>launchpad\.net/((?P<project>[A-Za-z0-9_.\-]+)(?P<series>/[A-Za-z0-9_.\-]+)?|~[A-Za-z0-9_.\-]+/(\+junk|[A-Za-z0-9_.\-]+)/[A-Za-z0-9_.\-]+))(/[A-Za-z0-9_.\-]+)*$`,
		// TODO(adg): write check function for Launchpad
	},
}

func init() {
	// Compile the regular expressions.
	for i := range vcsPaths {
		vcsPaths[i].regexp = regexp.MustCompile(vcsPaths[i].re)
	}
}

// googleCodeVCS performs an HTTP GET to verify that a Google Code project
// (and, optionally, a sub-repository) exists.
func googleCodeVCS(c appengine.Context, match map[string]string) bool {
	u := "https://code.google.com/p/" + match["project"]
	if match["subrepo"] != "" {
		u += "/source/checkout?repo=" + match["subrepo"]
	}
	return checkURL(c, u)
}

// checkRoot performs an HTTP GET to verify that a specific repository root
// exists (for github and bitbucket both).
func checkRoot(c appengine.Context, match map[string]string) bool {
	return checkURL(c, "https://"+match["root"])
}

// checkURL performs an HTTP GET to the specified URL and returns whether the
// remote server returned a 2xx response.
func checkURL(c appengine.Context, u string) bool {
	client := urlfetch.Client(c)
	resp, err := client.Get(u)
	if err != nil {
		c.Errorf("checkURL(%q): %v", u, err)
		return false
	}
	if resp.StatusCode/100 != 2 {
		c.Debugf("checkURL(%q): HTTP status: %s", u, resp.Status)
		return false
	}
	return true
}
