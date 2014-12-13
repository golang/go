// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"net/http"
	"strings"

	"appengine"
)

func handleFunc(path string, h http.HandlerFunc) {
	for _, d := range dashboards {
		http.HandleFunc(d.Prefix+path, h)
	}
}

// Dashboard describes a unique build dashboard.
type Dashboard struct {
	Name      string     // This dashboard's name (eg, "Go")
	Namespace string     // This dashboard's namespace (eg, "" (default), "Git")
	Prefix    string     // The path prefix (no trailing /)
	Packages  []*Package // The project's packages to build
}

// dashboardForRequest returns the appropriate dashboard for a given URL path.
func dashboardForRequest(r *http.Request) *Dashboard {
	if strings.HasPrefix(r.URL.Path, gccgoDash.Prefix) {
		return gccgoDash
	}
	if strings.HasPrefix(r.URL.Path, hgDash.Prefix) {
		return hgDash
	}
	return goDash
}

// Context returns a namespaced context for this dashboard, or panics if it
// fails to create a new context.
func (d *Dashboard) Context(c appengine.Context) appengine.Context {
	if d.Namespace == "" {
		return c
	}
	n, err := appengine.Namespace(c, d.Namespace)
	if err != nil {
		panic(err)
	}
	return n
}

// the currently known dashboards.
var dashboards = []*Dashboard{goDash, hgDash, gccgoDash}

// hgDash is the dashboard for the old Mercural Go repository.
var hgDash = &Dashboard{
	Name:      "Mercurial",
	Namespace: "", // Used to be the default.
	Prefix:    "/hg",
	Packages:  hgPackages,
}

// hgPackages is a list of all of the packages
// built by the old Mercurial Go repository.
var hgPackages = []*Package{
	{
		Kind: "go",
		Name: "Go",
	},
	{
		Kind: "subrepo",
		Name: "go.blog",
		Path: "code.google.com/p/go.blog",
	},
	{
		Kind: "subrepo",
		Name: "go.codereview",
		Path: "code.google.com/p/go.codereview",
	},
	{
		Kind: "subrepo",
		Name: "go.crypto",
		Path: "code.google.com/p/go.crypto",
	},
	{
		Kind: "subrepo",
		Name: "go.exp",
		Path: "code.google.com/p/go.exp",
	},
	{
		Kind: "subrepo",
		Name: "go.image",
		Path: "code.google.com/p/go.image",
	},
	{
		Kind: "subrepo",
		Name: "go.net",
		Path: "code.google.com/p/go.net",
	},
	{
		Kind: "subrepo",
		Name: "go.sys",
		Path: "code.google.com/p/go.sys",
	},
	{
		Kind: "subrepo",
		Name: "go.talks",
		Path: "code.google.com/p/go.talks",
	},
	{
		Kind: "subrepo",
		Name: "go.tools",
		Path: "code.google.com/p/go.tools",
	},
}

// goDash is the dashboard for the main go repository.
var goDash = &Dashboard{
	Name:      "Go",
	Namespace: "Git",
	Prefix:    "",
	Packages:  goPackages,
}

// goPackages is a list of all of the packages built by the main go repository.
var goPackages = []*Package{
	{
		Kind: "go",
		Name: "Go",
	},
	{
		Kind: "subrepo",
		Name: "blog",
		Path: "golang.org/x/blog",
	},
	{
		Kind: "subrepo",
		Name: "crypto",
		Path: "golang.org/x/crypto",
	},
	{
		Kind: "subrepo",
		Name: "exp",
		Path: "golang.org/x/exp",
	},
	{
		Kind: "subrepo",
		Name: "image",
		Path: "golang.org/x/image",
	},
	{
		Kind: "subrepo",
		Name: "mobile",
		Path: "golang.org/x/mobile",
	},
	{
		Kind: "subrepo",
		Name: "net",
		Path: "golang.org/x/net",
	},
	{
		Kind: "subrepo",
		Name: "review",
		Path: "golang.org/x/review",
	},
	{
		Kind: "subrepo",
		Name: "sys",
		Path: "golang.org/x/sys",
	},
	{
		Kind: "subrepo",
		Name: "talks",
		Path: "golang.org/x/talks",
	},
	{
		Kind: "subrepo",
		Name: "text",
		Path: "golang.org/x/text",
	},
	{
		Kind: "subrepo",
		Name: "tools",
		Path: "golang.org/x/tools",
	},
}

// gccgoDash is the dashboard for gccgo.
var gccgoDash = &Dashboard{
	Name:      "Gccgo",
	Namespace: "Gccgo",
	Prefix:    "/gccgo",
	Packages: []*Package{
		{
			Kind: "gccgo",
			Name: "Gccgo",
		},
	},
}
