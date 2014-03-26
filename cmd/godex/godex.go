// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"code.google.com/p/go.tools/go/types"
)

// BUG(gri) cannot specify package paths with dots (code.google.com/p/go.tools/cmd/ssadump)

var (
	source  = flag.String("s", "", "only consider packages from this source")
	verbose = flag.Bool("v", false, "verbose mode")
)

var (
	importFailed = errors.New("import failed")
	importers    = make(map[string]types.Importer)
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
		imp = importers[*source]
		if imp == nil {
			report("source must be one of: " + importersList())
		}
	}

	for _, arg := range flag.Args() {
		if *verbose {
			fmt.Fprintf(os.Stderr, "(processing %s)\n", arg)
		}

		// determine import path, object name
		var path, name string
		elems := strings.Split(arg, ".")
		switch len(elems) {
		case 2:
			name = elems[1]
			fallthrough
		case 1:
			path = elems[0]
		default:
			fmt.Fprintf(os.Stderr, "ignoring %q: invalid path or (qualified) identifier\n", arg)
			continue
		}

		// import package
		pkg, err := imp(packages, path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ignoring %q: %s\n", path, err)
			continue
		}

		// filter objects if needed
		filter := exportFilter
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
	for source, imp := range importers {
		if *verbose {
			fmt.Fprintf(os.Stderr, "(trying as %s)\n", source)
		}
		pkg, err = imp(packages, path)
		if err == nil {
			break
		}
	}
	return
}

func register(source string, imp types.Importer) {
	if _, ok := importers[source]; ok {
		panic(source + " importer already registered")
	}
	importers[source] = imp
}

func importersList() string {
	var s string
	for n := range importers {
		if len(s) == 0 {
			s = n
		} else {
			s = s + ", " + n
		}
	}
	return s
}

func exportFilter(obj types.Object) bool {
	return obj.Exported()
}
