// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is a reduced copy of $GOROOT/src/go/internal/gcimporter/gcimporter.go.

// Package gcimporter provides various functions for reading
// gc-generated object files that can be used to implement the
// Importer interface defined by the Go 1.5 standard library package.
//
// The encoding is deterministic: if the encoder is applied twice to
// the same types.Package data structure, both encodings are equal.
// This property may be important to avoid spurious changes in
// applications such as build systems.
//
// However, the encoder is not necessarily idempotent. Importing an
// exported package may yield a types.Package that, while it
// represents the same set of Go types as the original, may differ in
// the details of its internal representation. Because of these
// differences, re-encoding the imported package may yield a
// different, but equally valid, encoding of the package.
package gcimporter // import "golang.org/x/tools/internal/gcimporter"

import (
	"bufio"
	"bytes"
	"fmt"
	"go/build"
	"go/token"
	"go/types"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

const (
	// Enable debug during development: it adds some additional checks, and
	// prevents errors from being recovered.
	debug = false

	// If trace is set, debugging output is printed to std out.
	trace = false
)

var exportMap sync.Map // package dir → func() (string, bool)

// lookupGorootExport returns the location of the export data
// (normally found in the build cache, but located in GOROOT/pkg
// in prior Go releases) for the package located in pkgDir.
//
// (We use the package's directory instead of its import path
// mainly to simplify handling of the packages in src/vendor
// and cmd/vendor.)
func lookupGorootExport(pkgDir string) (string, bool) {
	f, ok := exportMap.Load(pkgDir)
	if !ok {
		var (
			listOnce   sync.Once
			exportPath string
		)
		f, _ = exportMap.LoadOrStore(pkgDir, func() (string, bool) {
			listOnce.Do(func() {
				cmd := exec.Command("go", "list", "-export", "-f", "{{.Export}}", pkgDir)
				cmd.Dir = build.Default.GOROOT
				var output []byte
				output, err := cmd.Output()
				if err != nil {
					return
				}

				exports := strings.Split(string(bytes.TrimSpace(output)), "\n")
				if len(exports) != 1 {
					return
				}

				exportPath = exports[0]
			})

			return exportPath, exportPath != ""
		})
	}

	return f.(func() (string, bool))()
}

var pkgExts = [...]string{".a", ".o"}

// FindPkg returns the filename and unique package id for an import
// path based on package information provided by build.Import (using
// the build.Default build.Context). A relative srcDir is interpreted
// relative to the current working directory.
// If no file was found, an empty filename is returned.
func FindPkg(path, srcDir string) (filename, id string) {
	if path == "" {
		return
	}

	var noext string
	switch {
	default:
		// "x" -> "$GOPATH/pkg/$GOOS_$GOARCH/x.ext", "x"
		// Don't require the source files to be present.
		if abs, err := filepath.Abs(srcDir); err == nil { // see issue 14282
			srcDir = abs
		}
		bp, _ := build.Import(path, srcDir, build.FindOnly|build.AllowBinary)
		if bp.PkgObj == "" {
			var ok bool
			if bp.Goroot && bp.Dir != "" {
				filename, ok = lookupGorootExport(bp.Dir)
			}
			if !ok {
				id = path // make sure we have an id to print in error message
				return
			}
		} else {
			noext = strings.TrimSuffix(bp.PkgObj, ".a")
			id = bp.ImportPath
		}

	case build.IsLocalImport(path):
		// "./x" -> "/this/directory/x.ext", "/this/directory/x"
		noext = filepath.Join(srcDir, path)
		id = noext

	case filepath.IsAbs(path):
		// for completeness only - go/build.Import
		// does not support absolute imports
		// "/x" -> "/x.ext", "/x"
		noext = path
		id = path
	}

	if false { // for debugging
		if path != id {
			fmt.Printf("%s -> %s\n", path, id)
		}
	}

	if filename != "" {
		if f, err := os.Stat(filename); err == nil && !f.IsDir() {
			return
		}
	}

	// try extensions
	for _, ext := range pkgExts {
		filename = noext + ext
		if f, err := os.Stat(filename); err == nil && !f.IsDir() {
			return
		}
	}

	filename = "" // not found
	return
}

// Import imports a gc-generated package given its import path and srcDir, adds
// the corresponding package object to the packages map, and returns the object.
// The packages map must contain all packages already imported.
func Import(packages map[string]*types.Package, path, srcDir string, lookup func(path string) (io.ReadCloser, error)) (pkg *types.Package, err error) {
	var rc io.ReadCloser
	var filename, id string
	if lookup != nil {
		// With custom lookup specified, assume that caller has
		// converted path to a canonical import path for use in the map.
		if path == "unsafe" {
			return types.Unsafe, nil
		}
		id = path

		// No need to re-import if the package was imported completely before.
		if pkg = packages[id]; pkg != nil && pkg.Complete() {
			return
		}
		f, err := lookup(path)
		if err != nil {
			return nil, err
		}
		rc = f
	} else {
		filename, id = FindPkg(path, srcDir)
		if filename == "" {
			if path == "unsafe" {
				return types.Unsafe, nil
			}
			return nil, fmt.Errorf("can't find import: %q", id)
		}

		// no need to re-import if the package was imported completely before
		if pkg = packages[id]; pkg != nil && pkg.Complete() {
			return
		}

		// open file
		f, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		defer func() {
			if err != nil {
				// add file name to error
				err = fmt.Errorf("%s: %v", filename, err)
			}
		}()
		rc = f
	}
	defer rc.Close()

	var hdr string
	var size int64
	buf := bufio.NewReader(rc)
	if hdr, size, err = FindExportData(buf); err != nil {
		return
	}

	switch hdr {
	case "$$B\n":
		var data []byte
		data, err = ioutil.ReadAll(buf)
		if err != nil {
			break
		}

		// TODO(gri): allow clients of go/importer to provide a FileSet.
		// Or, define a new standard go/types/gcexportdata package.
		fset := token.NewFileSet()

		// Select appropriate importer.
		if len(data) > 0 {
			switch data[0] {
			case 'v', 'c', 'd': // binary, till go1.10
				return nil, fmt.Errorf("binary (%c) import format is no longer supported", data[0])

			case 'i': // indexed, till go1.19
				_, pkg, err := IImportData(fset, packages, data[1:], id)
				return pkg, err

			case 'u': // unified, from go1.20
				_, pkg, err := UImportData(fset, packages, data[1:size], id)
				return pkg, err

			default:
				l := len(data)
				if l > 10 {
					l = 10
				}
				return nil, fmt.Errorf("unexpected export data with prefix %q for path %s", string(data[:l]), id)
			}
		}

	default:
		err = fmt.Errorf("unknown export data header: %q", hdr)
	}

	return
}

func deref(typ types.Type) types.Type {
	if p, _ := typ.(*types.Pointer); p != nil {
		return p.Elem()
	}
	return typ
}

type byPath []*types.Package

func (a byPath) Len() int           { return len(a) }
func (a byPath) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byPath) Less(i, j int) bool { return a[i].Path() < a[j].Path() }
