// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package buildutil provides utilities related to the go/build
// package in the standard library.
//
// All I/O is done via the build.Context file system interface, which must
// be concurrency-safe.
package buildutil

import (
	"go/build"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// AllPackages returns the import path of each Go package in any source
// directory of the specified build context (e.g. $GOROOT or an element
// of $GOPATH).  Errors are ignored.  The results are sorted.
//
// The result may include import paths for directories that contain no
// *.go files, such as "archive" (in $GOROOT/src).
//
// All I/O is done via the build.Context file system interface,
// which must be concurrency-safe.
//
func AllPackages(ctxt *build.Context) []string {
	var list []string
	var mu sync.Mutex
	ForEachPackage(ctxt, func(pkg string, _ error) {
		mu.Lock()
		list = append(list, pkg)
		mu.Unlock()
	})
	sort.Strings(list)
	return list
}

// ForEachPackage calls the found function with the import path of
// each Go package it finds in any source directory of the specified
// build context (e.g. $GOROOT or an element of $GOPATH).
//
// If the package directory exists but could not be read, the second
// argument to the found function provides the error.
//
// The found function and the build.Context file system interface
// accessors must be concurrency safe.
//
func ForEachPackage(ctxt *build.Context, found func(importPath string, err error)) {
	// We use a counting semaphore to limit
	// the number of parallel calls to ReadDir.
	sema := make(chan bool, 20)

	var wg sync.WaitGroup
	for _, root := range ctxt.SrcDirs() {
		root := root
		wg.Add(1)
		go func() {
			allPackages(ctxt, sema, root, found)
			wg.Done()
		}()
	}
	wg.Wait()
}

func allPackages(ctxt *build.Context, sema chan bool, root string, found func(string, error)) {
	root = filepath.Clean(root) + string(os.PathSeparator)

	var wg sync.WaitGroup

	var walkDir func(dir string)
	walkDir = func(dir string) {
		// Avoid .foo, _foo, and testdata directory trees.
		base := filepath.Base(dir)
		if base == "" || base[0] == '.' || base[0] == '_' || base == "testdata" {
			return
		}

		pkg := filepath.ToSlash(strings.TrimPrefix(dir, root))

		// Prune search if we encounter any of these import paths.
		switch pkg {
		case "builtin":
			return
		}

		sema <- true
		files, err := ReadDir(ctxt, dir)
		<-sema
		if pkg != "" || err != nil {
			found(pkg, err)
		}
		for _, fi := range files {
			fi := fi
			if fi.IsDir() {
				wg.Add(1)
				go func() {
					walkDir(filepath.Join(dir, fi.Name()))
					wg.Done()
				}()
			}
		}
	}

	walkDir(root)
	wg.Wait()
}
