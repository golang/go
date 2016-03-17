// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package main

import (
	"errors"
	"flag"
	"fmt"
	"go/build"
	"go/types"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

var (
	source  = flag.String("s", "", "only consider packages from src, where src is one of the supported compilers")
	verbose = flag.Bool("v", false, "verbose mode")
)

// lists of registered sources and corresponding importers
var (
	sources      []string
	importers    []types.Importer
	importFailed = errors.New("import failed")
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

	var imp types.Importer = new(tryImporters)
	if *source != "" {
		imp = lookup(*source)
		if imp == nil {
			report("source (-s argument) must be one of: " + strings.Join(sources, ", "))
		}
	}

	for _, arg := range flag.Args() {
		path, name := splitPathIdent(arg)
		logf("\tprocessing %q: path = %q, name = %s\n", arg, path, name)

		// generate possible package path prefixes
		// (at the moment we do this for each argument - should probably cache the generated prefixes)
		prefixes := make(chan string)
		go genPrefixes(prefixes, !filepath.IsAbs(path) && !build.IsLocalImport(path))

		// import package
		pkg, err := tryPrefixes(prefixes, path, imp)
		if err != nil {
			logf("\t=> ignoring %q: %s\n", path, err)
			continue
		}

		// filter objects if needed
		var filter func(types.Object) bool
		if name != "" {
			filter = func(obj types.Object) bool {
				// TODO(gri) perhaps use regular expression matching here?
				return obj.Name() == name
			}
		}

		// print contents
		print(os.Stdout, pkg, filter)
	}
}

func logf(format string, args ...interface{}) {
	if *verbose {
		fmt.Fprintf(os.Stderr, format, args...)
	}
}

// splitPathIdent splits a path.name argument into its components.
// All but the last path element may contain dots.
func splitPathIdent(arg string) (path, name string) {
	if i := strings.LastIndex(arg, "."); i >= 0 {
		if j := strings.LastIndex(arg, "/"); j < i {
			// '.' is not part of path
			path = arg[:i]
			name = arg[i+1:]
			return
		}
	}
	path = arg
	return
}

// tryPrefixes tries to import the package given by (the possibly partial) path using the given importer imp
// by prepending all possible prefixes to path. It returns with the first package that it could import, or
// with an error.
func tryPrefixes(prefixes chan string, path string, imp types.Importer) (pkg *types.Package, err error) {
	for prefix := range prefixes {
		actual := path
		if prefix == "" {
			// don't use filepath.Join as it will sanitize the path and remove
			// a leading dot and then the path is not recognized as a relative
			// package path by the importers anymore
			logf("\ttrying no prefix\n")
		} else {
			actual = filepath.Join(prefix, path)
			logf("\ttrying prefix %q\n", prefix)
		}
		pkg, err = imp.Import(actual)
		if err == nil {
			break
		}
		logf("\t=> importing %q failed: %s\n", actual, err)
	}
	return
}

// tryImporters is an importer that tries all registered importers
// successively until one of them succeeds or all of them failed.
type tryImporters struct{}

func (t *tryImporters) Import(path string) (pkg *types.Package, err error) {
	for i, imp := range importers {
		logf("\t\ttrying %s import\n", sources[i])
		pkg, err = imp.Import(path)
		if err == nil {
			break
		}
		logf("\t\t=> %s import failed: %s\n", sources[i], err)
	}
	return
}

type protector struct {
	imp types.Importer
}

func (p *protector) Import(path string) (pkg *types.Package, err error) {
	defer func() {
		if recover() != nil {
			pkg = nil
			err = importFailed
		}
	}()
	return p.imp.Import(path)
}

// protect protects an importer imp from panics and returns the protected importer.
func protect(imp types.Importer) types.Importer {
	return &protector{imp}
}

// register registers an importer imp for a given source src.
func register(src string, imp types.Importer) {
	if lookup(src) != nil {
		panic(src + " importer already registered")
	}
	sources = append(sources, src)
	importers = append(importers, protect(imp))
}

// lookup returns the importer imp for a given source src.
func lookup(src string) types.Importer {
	for i, s := range sources {
		if s == src {
			return importers[i]
		}
	}
	return nil
}

func genPrefixes(out chan string, all bool) {
	out <- ""
	if all {
		platform := build.Default.GOOS + "_" + build.Default.GOARCH
		dirnames := append([]string{build.Default.GOROOT}, filepath.SplitList(build.Default.GOPATH)...)
		for _, dirname := range dirnames {
			walkDir(filepath.Join(dirname, "pkg", platform), "", out)
		}
	}
	close(out)
}

func walkDir(dirname, prefix string, out chan string) {
	fiList, err := ioutil.ReadDir(dirname)
	if err != nil {
		return
	}
	for _, fi := range fiList {
		if fi.IsDir() && !strings.HasPrefix(fi.Name(), ".") {
			prefix := filepath.Join(prefix, fi.Name())
			out <- prefix
			walkDir(filepath.Join(dirname, fi.Name()), prefix, out)
		}
	}
}
