// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file replaces main.go when running godoc under the app engine emulator.
// See the README file for instructions.

package main

import (
	"http"
	"log"
	"os"
	"path/filepath"
)

func serveError(w http.ResponseWriter, r *http.Request, relpath string, err os.Error) {
	contents := applyTemplate(errorHTML, "errorHTML", err) // err may contain an absolute path!
	w.WriteHeader(http.StatusNotFound)
	servePage(w, "File "+relpath, "", "", contents)
}

func init() {
	// set goroot
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("cwd: %s", err)
	}
	log.Printf("cwd = %s", cwd)
	*goroot = filepath.Clean(cwd)

	initHandlers()
	readTemplates()
	registerPublicHandlers(http.DefaultServeMux)
}
