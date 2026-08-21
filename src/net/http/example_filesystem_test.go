// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"io/fs"
	"log"
	"net/http"
	"os"
	"slices"
	"strings"
)

// containsDotFile reports whether name contains a path element starting with a period.
// The name is delimited by forward slashes, as guaranteed by the fs.FS interface.
func containsDotFile(name string) bool {
	if name == "." {
		return false // allow the root directory, ".".
	}
	for part := range strings.SplitSeq(name, "/") {
		if strings.HasPrefix(part, ".") {
			return true
		}
	}
	return false
}

// dotFileHidingFile is the fs.File use in dotFileHidingFileSystem.
// It is used to wrap the Readdir method of fs.ReadDirFile so that we can
// remove files and directories that start with a period from its output.
type dotFileHidingFile struct {
	fs.ReadDirFile
}

// Readdir is a wrapper around the Readdir method of the embedded File
// that filters out all files that start with a period in their name.
func (f dotFileHidingFile) ReadDir(n int) ([]fs.DirEntry, error) {
	ents, err := f.ReadDirFile.ReadDir(n)
	ents = slices.DeleteFunc(ents, func(ent fs.DirEntry) bool {
		return strings.HasPrefix(ent.Name(), ".")
	})
	return ents, err
}

// dotFileHidingFileSystem is an http.FileSystem that hides
// hidden "dot files" from being served.
type dotFileHidingFileSystem struct {
	fs.FS
}

// Open is a wrapper around the Open method of the embedded FileSystem
// that serves a 403 permission error when name has a file or directory
// with whose name starts with a period in its path.
func (fsys dotFileHidingFileSystem) Open(name string) (fs.File, error) {
	if containsDotFile(name) { // If dot file, return 403 response
		return nil, fs.ErrPermission
	}
	file, err := fsys.FS.Open(name)
	if rdf, ok := file.(fs.ReadDirFile); ok {
		file = dotFileHidingFile{rdf}
	}
	return file, err
}

// FileServerFS will serve files starting with a dot, which can expose sensitive
// directories such as .git or sensitive files such as .htpassword.
//
// This example demonstrates hiding dot files by wrapping the fs.FS.
func ExampleFileServerFS_dotFileHiding() {
	root, err := os.OpenRoot("doc")
	if err != nil {
		log.Fatal(err)
	}
	fsys := dotFileHidingFileSystem{root.FS()}
	handler := http.FileServerFS(fsys)
	log.Fatal(http.ListenAndServe(":8080", handler))
}
