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

// Dashboard describes a unique build dashboard.
type Dashboard struct {
	Name     string     // This dashboard's name and namespace
	RelPath  string     // The relative url path
	Packages []*Package // The project's packages to build
}

// dashboardForRequest returns the appropriate dashboard for a given URL path.
func dashboardForRequest(r *http.Request) *Dashboard {
	if strings.HasPrefix(r.URL.Path, gccgoDash.RelPath) {
		return gccgoDash
	}
	return goDash
}

// Context returns a namespaced context for this dashboard, or panics if it
// fails to create a new context.
func (d *Dashboard) Context(c appengine.Context) appengine.Context {
	// No namespace needed for the original Go dashboard.
	if d.Name == "Go" {
		return c
	}
	n, err := appengine.Namespace(c, d.Name)
	if err != nil {
		panic(err)
	}
	return n
}

// the currently known dashboards.
var dashboards = []*Dashboard{goDash, gccgoDash}

// goDash is the dashboard for the main go repository.
var goDash = &Dashboard{
	Name:     "Go",
	RelPath:  "/",
	Packages: goPackages,
}

// goPackages is a list of all of the packages built by the main go repository.
var goPackages = []*Package{
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
		Name: "go.talks",
		Path: "code.google.com/p/go.talks",
	},
	{
		Kind: "subrepo",
		Name: "go.tools",
		Path: "code.google.com/p/go.tools",
	},
}

// gccgoDash is the dashboard for gccgo.
var gccgoDash = &Dashboard{
	Name:    "Gccgo",
	RelPath: "/gccgo/",
	Packages: []*Package{
		{
			Kind: "gccgo",
			Name: "Gccgo",
		},
	},
}
