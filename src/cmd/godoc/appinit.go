// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// To run godoc under app engine, substitute main.go with
// this file (appinit.go), provide a .zip file containing
// the file system to serve, and adjust the configuration
// parameters in appconfig.go accordingly.
//
// The current app engine SDK may be based on an older Go
// release version. To correct for version skew, copy newer
// packages into the alt directory (e.g. alt/strings) and
// adjust the imports in the godoc source files (e.g. from
// `import "strings"` to `import "alt/strings"`). Both old
// and new packages may be used simultaneously as long as
// there is no package global state that needs to be shared.
//
// The directory structure should look as follows:
//
// godoc			// directory containing the app engine app
//      alt			// alternative packages directory to
//				//	correct for version skew
//		strings		// never version of the strings package
//		...		//
//	app.yaml		// app engine control file
//	go.zip			// zip file containing the file system to serve
//	godoc			// contains godoc sources
//		appinit.go	// this file instead of godoc/main.go
//		appconfig.go	// godoc for app engine configuration
//		...		//
//
// To run app the engine emulator locally:
//
//	dev_appserver.py -a 0 godoc
//
// godoc is the top-level "goroot" directory.
// The godoc home page is served at: <hostname>:8080 and localhost:8080.

package main

import (
	"alt/archive/zip"
	"http"
	"log"
	"os"
)

func serveError(w http.ResponseWriter, r *http.Request, relpath string, err os.Error) {
	contents := applyTemplate(errorHTML, "errorHTML", err) // err may contain an absolute path!
	w.WriteHeader(http.StatusNotFound)
	servePage(w, "File "+relpath, "", "", contents)
}

func init() {
	log.Println("initializing godoc ...")
	*goroot = path.Join("/", zipGoroot) // fsHttp paths are relative to '/'

	// read .zip file and set up file systems
	const zipfile = zipFilename
	rc, err := zip.OpenReader(zipfile)
	if err != nil {
		log.Fatalf("%s: %s\n", zipfile, err)
	}
	fs = NewZipFS(rc)
	fsHttp = NewHttpZipFS(rc, *goroot)

	// initialize http handlers
	initHandlers()
	readTemplates()
	registerPublicHandlers(http.DefaultServeMux)

	// initialize default directory tree with corresponding timestamp.
	initFSTree()

	// initialize directory trees for user-defined file systems (-path flag).
	initDirTrees()

	// create search index
	// TODO(gri) Disabled for now as it takes too long. Find a solution for this.
	/*
		*indexEnabled = true
		go indexer()
	*/

	log.Println("godoc initialization complete")
}
