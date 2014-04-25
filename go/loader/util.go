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

// parseFiles parses the Go source files files within directory dir
// and returns their ASTs, or the first parse error if any.
//
// I/O is done via ctxt, which may specify a virtual file system.
// displayPath is used to transform the filenames attached to the ASTs.
//
func parseFiles(fset *token.FileSet, ctxt *build.Context, displayPath func(string) string, dir string, files []string, mode parser.Mode) ([]*ast.File, error) {
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
			defer rd.Close()
			if err != nil {
				errors[i] = err
				return
			}
			parsed[i], errors[i] = parser.ParseFile(fset, displayPath(file), rd, mode)
		}(i, file)
	}
	wg.Wait()

	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}
	return parsed, nil
}

// ---------- Internal helpers ----------

// TODO(adonovan): make this a method: func (*token.File) Contains(token.Pos)
func tokenFileContainsPos(f *token.File, pos token.Pos) bool {
	p := int(pos)
	base := f.Base()
	return base <= p && p < base+f.Size()
}
