// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package main

// This file replaces main.go when running godoc under app-engine.
// See README.godoc-app for details.

import (
	"archive/zip"
	"log"
	"net/http"
	"path"
)

func serveError(w http.ResponseWriter, r *http.Request, relpath string, err error) {
	w.WriteHeader(http.StatusNotFound)
	servePage(w, Page{
		Title:    "File " + relpath,
		Subtitle: relpath,
		Body:     applyTemplate(errorHTML, "errorHTML", err), // err may contain an absolute path!
	})
}

func init() {
	log.Println("initializing godoc ...")
	log.Printf(".zip file   = %s", zipFilename)
	log.Printf(".zip GOROOT = %s", zipGoroot)
	log.Printf("index files = %s", indexFilenames)

	// initialize flags for app engine
	*goroot = path.Join("/", zipGoroot) // fsHttp paths are relative to '/'
	*indexEnabled = true
	*indexFiles = indexFilenames
	*maxResults = 100    // reduce latency by limiting the number of fulltext search results
	*indexThrottle = 0.3 // in case *indexFiles is empty (and thus the indexer is run)
	*showPlayground = true

	// read .zip file and set up file systems
	const zipfile = zipFilename
	rc, err := zip.OpenReader(zipfile)
	if err != nil {
		log.Fatalf("%s: %s\n", zipfile, err)
	}
	// rc is never closed (app running forever)
	fs.Bind("/", NewZipFS(rc, zipFilename), *goroot, bindReplace)

	// initialize http handlers
	readTemplates()
	initHandlers()
	registerPublicHandlers(http.DefaultServeMux)
	registerPlaygroundHandlers(http.DefaultServeMux)

	// initialize default directory tree with corresponding timestamp.
	initFSTree()

	// Immediately update metadata.
	updateMetadata()

	// initialize search index
	if *indexEnabled {
		go indexer()
	}

	log.Println("godoc initialization complete")
}
