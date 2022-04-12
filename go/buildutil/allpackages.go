// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package buildutil provides utilities related to the go/build
// package in the standard library.
//
// All I/O is done via the build.Context file system interface, which must
// be concurrency-safe.
package buildutil // import "golang.org/x/tools/go/buildutil"

import (
	"go/build"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// AllPackages returns the package path of each Go package in any source
// directory of the specified build context (e.g. $GOROOT or an element
// of $GOPATH).  Errors are ignored.  The results are sorted.
// All package paths are canonical, and thus may contain "/vendor/".
//
// The result may include import paths for directories that contain no
// *.go files, such as "archive" (in $GOROOT/src).
//
// All I/O is done via the build.Context file system interface,
// which must be concurrency-safe.
func AllPackages(ctxt *build.Context) []string {
	var list []string
	ForEachPackage(ctxt, func(pkg string, _ error) {
		list = append(list, pkg)
	})
	sort.Strings(list)
	return list
}

// ForEachPackage calls the found function with the package path of
// each Go package it finds in any source directory of the specified
// build context (e.g. $GOROOT or an element of $GOPATH).
// All package paths are canonical, and thus may contain "/vendor/".
//
// If the package directory exists but could not be read, the second
// argument to the found function provides the error.
//
// All I/O is done via the build.Context file system interface,
// which must be concurrency-safe.
func ForEachPackage(ctxt *build.Context, found func(importPath string, err error)) {
	ch := make(chan item)

	var wg sync.WaitGroup
	for _, root := range ctxt.SrcDirs() {
		root := root
		wg.Add(1)
		go func() {
			allPackages(ctxt, root, ch)
			wg.Done()
		}()
	}
	go func() {
		wg.Wait()
		close(ch)
	}()

	// All calls to found occur in the caller's goroutine.
	for i := range ch {
		found(i.importPath, i.err)
	}
}

type item struct {
	importPath string
	err        error // (optional)
}

// We use a process-wide counting semaphore to limit
// the number of parallel calls to ReadDir.
var ioLimit = make(chan bool, 20)

func allPackages(ctxt *build.Context, root string, ch chan<- item) {
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

		ioLimit <- true
		files, err := ReadDir(ctxt, dir)
		<-ioLimit
		if pkg != "" || err != nil {
			ch <- item{pkg, err}
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

// ExpandPatterns returns the set of packages matched by patterns,
// which may have the following forms:
//
//	golang.org/x/tools/cmd/guru     # a single package
//	golang.org/x/tools/...          # all packages beneath dir
//	...                             # the entire workspace.
//
// Order is significant: a pattern preceded by '-' removes matching
// packages from the set.  For example, these patterns match all encoding
// packages except encoding/xml:
//
//	encoding/... -encoding/xml
//
// A trailing slash in a pattern is ignored.  (Path components of Go
// package names are separated by slash, not the platform's path separator.)
func ExpandPatterns(ctxt *build.Context, patterns []string) map[string]bool {
	// TODO(adonovan): support other features of 'go list':
	// - "std"/"cmd"/"all" meta-packages
	// - "..." not at the end of a pattern
	// - relative patterns using "./" or "../" prefix

	pkgs := make(map[string]bool)
	doPkg := func(pkg string, neg bool) {
		if neg {
			delete(pkgs, pkg)
		} else {
			pkgs[pkg] = true
		}
	}

	// Scan entire workspace if wildcards are present.
	// TODO(adonovan): opt: scan only the necessary subtrees of the workspace.
	var all []string
	for _, arg := range patterns {
		if strings.HasSuffix(arg, "...") {
			all = AllPackages(ctxt)
			break
		}
	}

	for _, arg := range patterns {
		if arg == "" {
			continue
		}

		neg := arg[0] == '-'
		if neg {
			arg = arg[1:]
		}

		if arg == "..." {
			// ... matches all packages
			for _, pkg := range all {
				doPkg(pkg, neg)
			}
		} else if dir := strings.TrimSuffix(arg, "/..."); dir != arg {
			// dir/... matches all packages beneath dir
			for _, pkg := range all {
				if strings.HasPrefix(pkg, dir) &&
					(len(pkg) == len(dir) || pkg[len(dir)] == '/') {
					doPkg(pkg, neg)
				}
			}
		} else {
			// single package
			doPkg(strings.TrimSuffix(arg, "/"), neg)
		}
	}

	return pkgs
}
