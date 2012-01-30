// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"fmt"
	"http"

	"appengine"
	"appengine/datastore"
	"cache"
)

// defaultPackages specifies the Package records to be created by initHandler.
var defaultPackages = []*Package{
	&Package{Name: "Go", Kind: "go"},
}

// subRepos specifies the Go project sub-repositories.
var subRepos = []string{
	"codereview",
	"crypto",
	"image",
	"net",
}

// Put subRepos into defaultPackages.
func init() {
	for _, name := range subRepos {
		p := &Package{
			Kind: "subrepo",
			Name: "go." + name,
			Path: "code.google.com/p/go." + name,
		}
		defaultPackages = append(defaultPackages, p)
	}
}

func initHandler(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)
	defer cache.Tick(c)
	for _, p := range defaultPackages {
		if err := datastore.Get(c, p.Key(c), new(Package)); err == nil {
			continue
		} else if err != datastore.ErrNoSuchEntity {
			logErr(w, r, err)
			return
		}
		if _, err := datastore.Put(c, p.Key(c), p); err != nil {
			logErr(w, r, err)
			return
		}
	}
	fmt.Fprint(w, "OK")
}
