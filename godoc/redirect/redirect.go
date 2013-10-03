// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package redirect provides hooks to register HTTP handlers that redirect old
// godoc paths to their new equivalents and assist in accessing the issue
// tracker, wiki, code review system, etc.
package redirect

import (
	"net/http"
	"regexp"
)

// Register registers HTTP handlers that redirect old godoc paths to their new
// equivalents and assist in accessing the issue tracker, wiki, code review
// system, etc. If mux is nil it uses http.DefaultServeMux.
func Register(mux *http.ServeMux) {
	if mux == nil {
		mux = http.DefaultServeMux
	}
	handlePathRedirects(mux, pkgRedirects, "/pkg/")
	handlePathRedirects(mux, cmdRedirects, "/cmd/")
	for prefix, redirect := range prefixHelpers {
		p := "/" + prefix + "/"
		mux.Handle(p, PrefixHandler(p, redirect))
	}
	for path, redirect := range redirects {
		mux.Handle(path, Handler(redirect))
	}
}

func handlePathRedirects(mux *http.ServeMux, redirects map[string]string, prefix string) {
	for source, target := range redirects {
		h := Handler(prefix + target + "/")
		p := prefix + source
		mux.Handle(p, h)
		mux.Handle(p+"/", h)
	}
}

// Packages that were renamed between r60 and go1.
var pkgRedirects = map[string]string{
	"asn1":              "encoding/asn1",
	"big":               "math/big",
	"cmath":             "math/cmplx",
	"csv":               "encoding/csv",
	"exec":              "os/exec",
	"exp/template/html": "html/template",
	"gob":               "encoding/gob",
	"http":              "net/http",
	"http/cgi":          "net/http/cgi",
	"http/fcgi":         "net/http/fcgi",
	"http/httptest":     "net/http/httptest",
	"http/pprof":        "net/http/pprof",
	"json":              "encoding/json",
	"mail":              "net/mail",
	"rand":              "math/rand",
	"rpc":               "net/rpc",
	"rpc/jsonrpc":       "net/rpc/jsonrpc",
	"scanner":           "text/scanner",
	"smtp":              "net/smtp",
	"tabwriter":         "text/tabwriter",
	"template":          "text/template",
	"template/parse":    "text/template/parse",
	"url":               "net/url",
	"utf16":             "unicode/utf16",
	"utf8":              "unicode/utf8",
	"xml":               "encoding/xml",
}

// Commands that were renamed between r60 and go1.
var cmdRedirects = map[string]string{
	"gofix":     "fix",
	"goinstall": "go",
	"gopack":    "pack",
	"gotest":    "go",
	"govet":     "vet",
	"goyacc":    "yacc",
}

var redirects = map[string]string{
	"/blog":       "/blog/",
	"/build":      "http://build.golang.org",
	"/change":     "https://code.google.com/p/go/source/list",
	"/cl":         "https://gocodereview.appspot.com/",
	"/cmd/godoc/": "http://godoc.org/code.google.com/p/go.tools/cmd/godoc/",
	"/cmd/vet/":   "http://godoc.org/code.google.com/p/go.tools/cmd/vet/",
	"/issue":      "https://code.google.com/p/go/issues",
	"/issue/new":  "https://code.google.com/p/go/issues/entry",
	"/issues":     "https://code.google.com/p/go/issues",
	"/play":       "http://play.golang.org",

	// In Go 1.2 the references page is part of /doc/.
	"/ref": "/doc/#references",
	// This next rule clobbers /ref/spec and /ref/mem.
	// TODO(adg): figure out what to do here, if anything.
	// "/ref/": "/doc/#references",

	// Be nice to people who are looking in the wrong place.
	"/doc/mem":  "/ref/mem",
	"/doc/spec": "/ref/spec",

	"/talks": "http://talks.golang.org",
	"/tour":  "http://tour.golang.org",
	"/wiki":  "https://code.google.com/p/go-wiki/w/list",

	"/doc/articles/c_go_cgo.html":                    "/blog/c-go-cgo",
	"/doc/articles/concurrency_patterns.html":        "/blog/go-concurrency-patterns-timing-out-and",
	"/doc/articles/defer_panic_recover.html":         "/blog/defer-panic-and-recover",
	"/doc/articles/error_handling.html":              "/blog/error-handling-and-go",
	"/doc/articles/gobs_of_data.html":                "/blog/gobs-of-data",
	"/doc/articles/godoc_documenting_go_code.html":   "/blog/godoc-documenting-go-code",
	"/doc/articles/gos_declaration_syntax.html":      "/blog/gos-declaration-syntax",
	"/doc/articles/image_draw.html":                  "/blog/go-imagedraw-package",
	"/doc/articles/image_package.html":               "/blog/go-image-package",
	"/doc/articles/json_and_go.html":                 "/blog/json-and-go",
	"/doc/articles/json_rpc_tale_of_interfaces.html": "/blog/json-rpc-tale-of-interfaces",
	"/doc/articles/laws_of_reflection.html":          "/blog/laws-of-reflection",
	"/doc/articles/race_detector.html":               "/blog/race-detector",
	"/doc/articles/slices_usage_and_internals.html":  "/blog/go-slices-usage-and-internals",
	"/doc/go_for_cpp_programmers.html":               "https://code.google.com/p/go-wiki/wiki/GoForCPPProgrammers",
	"/doc/go_tutorial.html":                          "http://tour.golang.org/",
}

var prefixHelpers = map[string]string{
	"change": "https://code.google.com/p/go/source/detail?r=",
	"cl":     "https://codereview.appspot.com/",
	"issue":  "https://code.google.com/p/go/issues/detail?id=",
	"play":   "http://play.golang.org/",
	"talks":  "http://talks.golang.org/",
	"wiki":   "https://code.google.com/p/go-wiki/wiki/",
}

func Handler(target string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target, http.StatusMovedPermanently)
	})
}

var validId = regexp.MustCompile(`^[A-Za-z0-9-]*$`)

func PrefixHandler(prefix, baseURL string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if p := r.URL.Path; p == prefix {
			// redirect /prefix/ to /prefix
			http.Redirect(w, r, p[:len(p)-1], http.StatusFound)
			return
		}
		id := r.URL.Path[len(prefix):]
		if !validId.MatchString(id) {
			http.Error(w, "Not found", http.StatusNotFound)
			return
		}
		target := baseURL + id
		http.Redirect(w, r, target, http.StatusFound)
	})
}
