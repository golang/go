// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package importer implements Import for gc-generated object files.
package importer

import (
	"bufio"
	"cmd/compile/internal/types2"
	"fmt"
	"go/build"
	"internal/pkgbits"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// debugging/development support
const debug = false

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
			id = path // make sure we have an id to print in error message
			return
		}
		noext = strings.TrimSuffix(bp.PkgObj, ".a")
		id = bp.ImportPath

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
func Import(packages map[string]*types2.Package, path, srcDir string, lookup func(path string) (io.ReadCloser, error)) (pkg *types2.Package, err error) {
	var rc io.ReadCloser
	var id string
	if lookup != nil {
		// With custom lookup specified, assume that caller has
		// converted path to a canonical import path for use in the map.
		if path == "unsafe" {
			return types2.Unsafe, nil
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
		var filename string
		filename, id = FindPkg(path, srcDir)
		if filename == "" {
			if path == "unsafe" {
				return types2.Unsafe, nil
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

	buf := bufio.NewReader(rc)
	hdr, size, err := FindExportData(buf)
	if err != nil {
		return
	}

	switch hdr {
	case "$$\n":
		err = fmt.Errorf("import %q: old textual export format no longer supported (recompile library)", path)

	case "$$B\n":
		var data []byte
		var r io.Reader = buf
		if size >= 0 {
			r = io.LimitReader(r, int64(size))
		}
		data, err = ioutil.ReadAll(r)
		if err != nil {
			break
		}

		if len(data) == 0 {
			err = fmt.Errorf("import %q: missing export data", path)
			break
		}
		exportFormat := data[0]
		s := string(data[1:])

		// The indexed export format starts with an 'i'; the older
		// binary export format starts with a 'c', 'd', or 'v'
		// (from "version"). Select appropriate importer.
		switch exportFormat {
		case 'u':
			s = s[:strings.Index(s, "\n$$\n")]
			input := pkgbits.NewPkgDecoder(id, s)
			pkg = ReadPackage(nil, packages, input)
		case 'i':
			pkg, err = ImportData(packages, s, id)
		default:
			err = fmt.Errorf("import %q: old binary export format no longer supported (recompile library)", path)
		}

	default:
		err = fmt.Errorf("import %q: unknown export data header: %q", path, hdr)
	}

	return
}

type byPath []*types2.Package

func (a byPath) Len() int           { return len(a) }
func (a byPath) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byPath) Less(i, j int) bool { return a[i].Path() < a[j].Path() }
