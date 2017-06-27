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
	"os"
	"path"
	"regexp"
	"runtime"

	"golang.org/x/tools/godoc"
	"golang.org/x/tools/godoc/dl"
	"golang.org/x/tools/godoc/proxy"
	"golang.org/x/tools/godoc/short"
	"golang.org/x/tools/godoc/static"
	"golang.org/x/tools/godoc/vfs"
	"golang.org/x/tools/godoc/vfs/gatefs"
	"golang.org/x/tools/godoc/vfs/mapfs"
	"golang.org/x/tools/godoc/vfs/zipfs"
	"google.golang.org/appengine"
)

func init() {
	var (
		// .zip filename
		zipFilename = os.Getenv("GODOC_ZIP")

		// goroot directory in .zip file
		zipGoroot = os.Getenv("GODOC_ZIP_PREFIX")

		// glob pattern describing search index files
		// (if empty, the index is built at run-time)
		indexFilenames = os.Getenv("GODOC_INDEX_GLOB")
	)

	enforceHosts = !appengine.IsDevAppServer()
	playEnabled = true

	log.Println("initializing godoc ...")
	log.Printf(".zip file   = %s", zipFilename)
	log.Printf(".zip GOROOT = %s", zipGoroot)
	log.Printf("index files = %s", indexFilenames)

	if zipFilename != "" {
		goroot := path.Join("/", zipGoroot) // fsHttp paths are relative to '/'
		// read .zip file and set up file systems
		rc, err := zip.OpenReader(zipFilename)
		if err != nil {
			log.Fatalf("%s: %s\n", zipFilename, err)
		}
		// rc is never closed (app running forever)
		fs.Bind("/", zipfs.New(rc, zipFilename), goroot, vfs.BindReplace)
	} else {
		rootfs := gatefs.New(vfs.OS(runtime.GOROOT()), make(chan bool, 20))
		fs.Bind("/", rootfs, "/", vfs.BindReplace)
	}

	fs.Bind("/lib/godoc", mapfs.New(static.Files), "/", vfs.BindReplace)

	corpus := godoc.NewCorpus(fs)
	corpus.Verbose = false
	corpus.MaxResults = 10000 // matches flag default in main.go
	corpus.IndexEnabled = true
	corpus.IndexFiles = indexFilenames
	if err := corpus.Init(); err != nil {
		log.Fatal(err)
	}
	corpus.IndexDirectory = indexDirectoryDefault
	go corpus.RunIndexer()

	pres = godoc.NewPresentation(corpus)
	pres.TabWidth = 8
	pres.ShowPlayground = true
	pres.ShowExamples = true
	pres.DeclLinks = true
	pres.NotesRx = regexp.MustCompile("BUG")

	readTemplates(pres, true)

	mux := registerHandlers(pres)
	dl.RegisterHandlers(mux)
	short.RegisterHandlers(mux)

	// Register /compile and /share handlers against the default serve mux
	// so that other app modules can make plain HTTP requests to those
	// hosts. (For reasons, HTTPS communication between modules is broken.)
	proxy.RegisterHandlers(http.DefaultServeMux)

	log.Println("godoc initialization complete")
}
