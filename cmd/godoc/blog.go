// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/build"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"code.google.com/p/go.tools/blog"
	"code.google.com/p/go.tools/godoc/redirect"
)

const (
	blogRepo = "code.google.com/p/go.blog"
	blogURL  = "http://blog.golang.org/"
	blogPath = "/blog/"
)

var (
	blogServer   http.Handler // set by blogInit
	blogInitOnce sync.Once
	playEnabled  bool
)

func init() {
	// Initialize blog only when first accessed.
	http.HandleFunc(blogPath, func(w http.ResponseWriter, r *http.Request) {
		blogInitOnce.Do(blogInit)
		blogServer.ServeHTTP(w, r)
	})
}

func blogInit() {
	// Binary distributions will include the blog content in "/blog".
	root := filepath.Join(runtime.GOROOT(), "blog")

	// Prefer content from go.blog repository if present.
	if pkg, err := build.Import(blogRepo, "", build.FindOnly); err == nil {
		root = pkg.Dir
	}

	// If content is not available fall back to redirect.
	if fi, err := os.Stat(root); err != nil || !fi.IsDir() {
		fmt.Fprintf(os.Stderr, "Blog content not available locally. "+
			"To install, run \n\tgo get %v\n", blogRepo)
		blogServer = http.HandlerFunc(blogRedirectHandler)
		return
	}

	s, err := blog.NewServer(blog.Config{
		BaseURL:      blogPath,
		BasePath:     strings.TrimSuffix(blogPath, "/"),
		ContentPath:  filepath.Join(root, "content"),
		TemplatePath: filepath.Join(root, "template"),
		HomeArticles: 5,
		PlayEnabled:  playEnabled,
	})
	if err != nil {
		log.Fatal(err)
	}
	blogServer = s
}

func blogRedirectHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == blogPath {
		http.Redirect(w, r, blogURL, http.StatusFound)
		return
	}
	blogPrefixHandler.ServeHTTP(w, r)
}

var blogPrefixHandler = redirect.PrefixHandler(blogPath, blogURL)
