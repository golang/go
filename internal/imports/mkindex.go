//go:build ignore
// +build ignore

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command mkindex creates the file "pkgindex.go" containing an index of the Go
// standard library. The file is intended to be built as part of the imports
// package, so that the package may be used in environments where a GOROOT is
// not available (such as App Engine).
package imports

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/format"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"
)

var (
	pkgIndex = make(map[string][]pkg)
	exports  = make(map[string]map[string]bool)
)

func main() {
	// Don't use GOPATH.
	ctx := build.Default
	ctx.GOPATH = ""

	// Populate pkgIndex global from GOROOT.
	for _, path := range ctx.SrcDirs() {
		f, err := os.Open(path)
		if err != nil {
			log.Print(err)
			continue
		}
		children, err := f.Readdir(-1)
		f.Close()
		if err != nil {
			log.Print(err)
			continue
		}
		for _, child := range children {
			if child.IsDir() {
				loadPkg(path, child.Name())
			}
		}
	}
	// Populate exports global.
	for _, ps := range pkgIndex {
		for _, p := range ps {
			e := loadExports(p.dir)
			if e != nil {
				exports[p.dir] = e
			}
		}
	}

	// Construct source file.
	var buf bytes.Buffer
	fmt.Fprint(&buf, pkgIndexHead)
	fmt.Fprintf(&buf, "var pkgIndexMaster = %#v\n", pkgIndex)
	fmt.Fprintf(&buf, "var exportsMaster = %#v\n", exports)
	src := buf.Bytes()

	// Replace main.pkg type name with pkg.
	src = bytes.Replace(src, []byte("main.pkg"), []byte("pkg"), -1)
	// Replace actual GOROOT with "/go".
	src = bytes.Replace(src, []byte(ctx.GOROOT), []byte("/go"), -1)
	// Add some line wrapping.
	src = bytes.Replace(src, []byte("}, "), []byte("},\n"), -1)
	src = bytes.Replace(src, []byte("true, "), []byte("true,\n"), -1)

	var err error
	src, err = format.Source(src)
	if err != nil {
		log.Fatal(err)
	}

	// Write out source file.
	err = os.WriteFile("pkgindex.go", src, 0644)
	if err != nil {
		log.Fatal(err)
	}
}

const pkgIndexHead = `package imports

func init() {
	pkgIndexOnce.Do(func() {
		pkgIndex.m = pkgIndexMaster
	})
	loadExports = func(dir string) map[string]bool {
		return exportsMaster[dir]
	}
}
`

type pkg struct {
	importpath string // full pkg import path, e.g. "net/http"
	dir        string // absolute file path to pkg directory e.g. "/usr/lib/go/src/fmt"
}

var fset = token.NewFileSet()

func loadPkg(root, importpath string) {
	shortName := path.Base(importpath)
	if shortName == "testdata" {
		return
	}

	dir := filepath.Join(root, importpath)
	pkgIndex[shortName] = append(pkgIndex[shortName], pkg{
		importpath: importpath,
		dir:        dir,
	})

	pkgDir, err := os.Open(dir)
	if err != nil {
		return
	}
	children, err := pkgDir.Readdir(-1)
	pkgDir.Close()
	if err != nil {
		return
	}
	for _, child := range children {
		name := child.Name()
		if name == "" {
			continue
		}
		if c := name[0]; c == '.' || ('0' <= c && c <= '9') {
			continue
		}
		if child.IsDir() {
			loadPkg(root, filepath.Join(importpath, name))
		}
	}
}

func loadExports(dir string) map[string]bool {
	exports := make(map[string]bool)
	buildPkg, err := build.ImportDir(dir, 0)
	if err != nil {
		if strings.Contains(err.Error(), "no buildable Go source files in") {
			return nil
		}
		log.Printf("could not import %q: %v", dir, err)
		return nil
	}
	for _, file := range buildPkg.GoFiles {
		f, err := parser.ParseFile(fset, filepath.Join(dir, file), nil, 0)
		if err != nil {
			log.Printf("could not parse %q: %v", file, err)
			continue
		}
		for name := range f.Scope.Objects {
			if ast.IsExported(name) {
				exports[name] = true
			}
		}
	}
	return exports
}
