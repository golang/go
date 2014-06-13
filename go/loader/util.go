// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"sync"
)

// parseFiles parses the Go source files within directory dir and
// returns the ASTs of the ones that could be at least partially parsed,
// along with a list of I/O and parse errors encountered.
//
// I/O is done via ctxt, which may specify a virtual file system.
// displayPath is used to transform the filenames attached to the ASTs.
//
func parseFiles(fset *token.FileSet, ctxt *build.Context, displayPath func(string) string, dir string, files []string, mode parser.Mode) ([]*ast.File, []error) {
	if displayPath == nil {
		displayPath = func(path string) string { return path }
	}
	isAbs := filepath.IsAbs
	if ctxt.IsAbsPath != nil {
		isAbs = ctxt.IsAbsPath
	}
	joinPath := filepath.Join
	if ctxt.JoinPath != nil {
		joinPath = ctxt.JoinPath
	}
	var wg sync.WaitGroup
	n := len(files)
	parsed := make([]*ast.File, n)
	errors := make([]error, n)
	for i, file := range files {
		if !isAbs(file) {
			file = joinPath(dir, file)
		}
		wg.Add(1)
		go func(i int, file string) {
			defer wg.Done()
			var rd io.ReadCloser
			var err error
			if ctxt.OpenFile != nil {
				rd, err = ctxt.OpenFile(file)
			} else {
				rd, err = os.Open(file)
			}
			if err != nil {
				errors[i] = err // open failed
				return
			}

			// ParseFile may return both an AST and an error.
			parsed[i], errors[i] = parser.ParseFile(fset, displayPath(file), rd, mode)
			rd.Close()
		}(i, file)
	}
	wg.Wait()

	// Eliminate nils, preserving order.
	var o int
	for _, f := range parsed {
		if f != nil {
			parsed[o] = f
			o++
		}
	}
	parsed = parsed[:o]

	o = 0
	for _, err := range errors {
		if err != nil {
			errors[o] = err
			o++
		}
	}
	errors = errors[:o]

	return parsed, errors
}

// ---------- Internal helpers ----------

// TODO(adonovan): make this a method: func (*token.File) Contains(token.Pos)
func tokenFileContainsPos(f *token.File, pos token.Pos) bool {
	p := int(pos)
	base := f.Base()
	return base <= p && p < base+f.Size()
}
