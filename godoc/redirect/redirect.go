// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package redirect provides hooks to register HTTP handlers that redirect old
// godoc paths to their new equivalents.
package redirect // import "golang.org/x/tools/godoc/redirect"

import (
	"net/http"
	"regexp"
)

// Register registers HTTP handlers that redirect old godoc paths to their new equivalents.
// If mux is nil it uses http.DefaultServeMux.
func Register(mux *http.ServeMux) {
	if mux == nil {
		mux = http.DefaultServeMux
	}
	// NB: /src/pkg (sans trailing slash) is the index of packages.
	mux.HandleFunc("/src/pkg/", srcPkgHandler)
}

func Handler(target string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		url := target
		if qs := r.URL.RawQuery; qs != "" {
			url += "?" + qs
		}
		http.Redirect(w, r, url, http.StatusMovedPermanently)
	})
}

var validID = regexp.MustCompile(`^[A-Za-z0-9-]*/?$`)

func PrefixHandler(prefix, baseURL string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if p := r.URL.Path; p == prefix {
			// redirect /prefix/ to /prefix
			http.Redirect(w, r, p[:len(p)-1], http.StatusFound)
			return
		}
		id := r.URL.Path[len(prefix):]
		if !validID.MatchString(id) {
			http.Error(w, "Not found", http.StatusNotFound)
			return
		}
		target := baseURL + id
		http.Redirect(w, r, target, http.StatusFound)
	})
}

// Redirect requests from the old "/src/pkg/foo" to the new "/src/foo".
// See http://golang.org/s/go14nopkg
func srcPkgHandler(w http.ResponseWriter, r *http.Request) {
	r.URL.Path = "/src/" + r.URL.Path[len("/src/pkg/"):]
	http.Redirect(w, r, r.URL.String(), http.StatusMovedPermanently)
}
