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
	"sync"

	"code.google.com/p/go.tools/blog"
)

const (
	blogRedirect = false
	blogRepo     = "code.google.com/p/go.blog"
	blogURL      = "http://blog.golang.org/"
)

var (
	blogServer   http.Handler // set by blogInit
	blogInitOnce sync.Once
)

func init() {
	// Initialize blog only when first accessed.
	http.HandleFunc("/blog/", func(w http.ResponseWriter, r *http.Request) {
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
		blogServer = makePrefixRedirectHandler("/blog/", blogURL)
		return
	}

	s, err := blog.NewServer(blog.Config{
		BaseURL:      "/blog/",
		BasePath:     "/blog",
		ContentPath:  filepath.Join(root, "content"),
		TemplatePath: filepath.Join(root, "template"),
		HomeArticles: 5,
		PlayEnabled:  *showPlayground,
	})
	if err != nil {
		log.Fatal(err)
	}
	blogServer = s
}
