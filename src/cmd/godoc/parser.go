// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains support functions for parsing .go files.
// Similar functionality is found in package go/parser but the
// functions here operate using godoc's file system fs instead
// of calling the OS's file operations directly.

package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
)

func parseFile(fset *token.FileSet, filename string, mode parser.Mode) (*ast.File, error) {
	src, err := ReadFile(fs, filename)
	if err != nil {
		return nil, err
	}
	return parser.ParseFile(fset, filename, src, mode)
}

func parseFiles(fset *token.FileSet, filenames []string) (pkgs map[string]*ast.Package, first error) {
	pkgs = make(map[string]*ast.Package)
	for _, filename := range filenames {
		file, err := parseFile(fset, filename, parser.ParseComments)
		if err != nil {
			if first == nil {
				first = err
			}
			continue
		}

		name := file.Name.Name
		pkg, found := pkgs[name]
		if !found {
			// TODO(gri) Use NewPackage here; reconsider ParseFiles API.
			pkg = &ast.Package{name, nil, nil, make(map[string]*ast.File)}
			pkgs[name] = pkg
		}
		pkg.Files[filename] = file
	}
	return
}

func parseDir(fset *token.FileSet, path string, filter func(os.FileInfo) bool) (map[string]*ast.Package, error) {
	list, err := fs.ReadDir(path)
	if err != nil {
		return nil, err
	}

	filenames := make([]string, len(list))
	i := 0
	for _, d := range list {
		if filter == nil || filter(d) {
			filenames[i] = filepath.Join(path, d.Name())
			i++
		}
	}
	filenames = filenames[0:i]

	return parseFiles(fset, filenames)
}
