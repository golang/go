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
	"fmt"
	"go/token"
	"go/types"
	"io"
	"os"
)

const (
	// Enable debug during development: it adds some additional checks, and
	// prevents errors from being recovered.
	debug = false

	// If trace is set, debugging output is printed to std out.
	trace = false
)

// Import imports a gc-generated package given its import path and srcDir, adds
// the corresponding package object to the packages map, and returns the object.
// The packages map must contain all packages already imported.
//
// Import is only used in tests.
func Import(fset *token.FileSet, packages map[string]*types.Package, path, srcDir string, lookup func(path string) (io.ReadCloser, error)) (pkg *types.Package, err error) {
	var rc io.ReadCloser
	var id string
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
		var filename string
		filename, id, err = FindPkg(path, srcDir)
		if filename == "" {
			if path == "unsafe" {
				return types.Unsafe, nil
			}
			return nil, err
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
	data, err := ReadUnified(buf)
	if err != nil {
		err = fmt.Errorf("import %q: %v", path, err)
		return
	}

	// unified: emitted by cmd/compile since go1.20.
	_, pkg, err = UImportData(fset, packages, data, id)

	return
}
