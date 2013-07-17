// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines types for abstract file system access and
// provides an implementation accessing the file system of the
// underlying OS.

package main

import (
	"io/ioutil"

	"code.google.com/p/go.tools/godoc/vfs"
)

// fs is the file system that godoc reads from and serves.
// It is a virtual file system that operates on slash-separated paths,
// and its root corresponds to the Go distribution root: /src/pkg
// holds the source tree, and so on.  This means that the URLs served by
// the godoc server are the same as the paths in the virtual file
// system, which helps keep things simple.
//
// New file trees - implementations of FileSystem - can be added to
// the virtual file system using nameSpace's Bind method.
// The usual setup is to bind OS(runtime.GOROOT) to the root
// of the name space and then bind any GOPATH/src directories
// on top of /src/pkg, so that all sources are in /src/pkg.
//
// For more about name spaces, see the NameSpace type's
// documentation in code.google.com/p/go.tools/godoc/vfs.
//
// The use of this virtual file system means that most code processing
// paths can assume they are slash-separated and should be using
// package path (often imported as pathpkg) to manipulate them,
// even on Windows.
//
var fs = vfs.NameSpace{} // the underlying file system for godoc

// ReadFile reads the file named by path from fs and returns the contents.
func ReadFile(fs vfs.FileSystem, path string) ([]byte, error) {
	rc, err := fs.Open(path)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return ioutil.ReadAll(rc)
}
