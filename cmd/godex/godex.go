// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"code.google.com/p/go.tools/go/types"
)

var (
	source  = flag.String("s", "", "only consider packages from this source")
	verbose = flag.Bool("v", false, "verbose mode")
)

var (
	sources      []string         // sources of export data corresponding to importers
	importers    []types.Importer // importers for corresponding sources
	importFailed = errors.New("import failed")
	packages     = make(map[string]*types.Package)
)

func usage() {
	fmt.Fprintln(os.Stderr, "usage: godex [flags] {path|qualifiedIdent}")
	flag.PrintDefaults()
	os.Exit(2)
}

func report(msg string) {
	fmt.Fprintln(os.Stderr, "error: "+msg)
	os.Exit(2)
}

func main() {
	flag.Usage = usage
	flag.Parse()

	if flag.NArg() == 0 {
		report("no package name, path, or file provided")
	}

	imp := tryImport
	if *source != "" {
		imp = lookup(*source)
		if imp == nil {
			report("source (-s argument) must be one of: " + strings.Join(sources, ", "))
		}
	}

	for _, arg := range flag.Args() {
		path, name := splitPathIdent(arg)
		if *verbose {
			fmt.Fprintf(os.Stderr, "(processing %q: path = %q, name = %s)\n", arg, path, name)
		}

		// import package
		pkg, err := imp(packages, path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ignoring %q: %s\n", path, err)
			continue
		}

		// filter objects if needed
		filter := types.Object.Exported
		if name != "" {
			f := filter
			filter = func(obj types.Object) bool {
				// TODO(gri) perhaps use regular expression matching here?
				return f(obj) && obj.Name() == name
			}
		}

		// print contents
		print(os.Stdout, pkg, filter)
	}
}

// protect protects an importer imp from panics and returns the protected importer.
func protect(imp types.Importer) types.Importer {
	return func(packages map[string]*types.Package, path string) (pkg *types.Package, err error) {
		defer func() {
			if recover() != nil {
				pkg = nil
				err = importFailed
			}
		}()
		return imp(packages, path)
	}
}

func tryImport(packages map[string]*types.Package, path string) (pkg *types.Package, err error) {
	for i, imp := range importers {
		if *verbose {
			fmt.Fprintf(os.Stderr, "(trying source: %s)\n", sources[i])
		}
		pkg, err = imp(packages, path)
		if err == nil {
			break
		}
	}
	return
}

// splitPathIdent splits a path.name argument into its components.
// All but the last path element may contain dots.
// TODO(gri) document this also in doc.go.
func splitPathIdent(arg string) (path, name string) {
	const sep = string(filepath.Separator)
	if i := strings.LastIndex(arg, "."); i >= 0 {
		if j := strings.LastIndex(arg, sep); j < i {
			// '.' is not part of path
			path = arg[:i]
			name = arg[i+1:]
			return
		}
	}
	path = arg
	return
}

func register(src string, imp types.Importer) {
	if lookup(src) != nil {
		panic(src + " importer already registered")
	}
	sources = append(sources, src)
	importers = append(importers, imp)
}

func lookup(src string) types.Importer {
	for i, s := range sources {
		if s == src {
			return importers[i]
		}
	}
	return nil
}
