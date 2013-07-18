// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains support functions for parsing .go files
// accessed via godoc's file system fs.

package godoc

import (
	"go/ast"
	"go/parser"
	"go/token"
	pathpkg "path"

	"code.google.com/p/go.tools/godoc/vfs"
)

func (c *Corpus) parseFile(fset *token.FileSet, filename string, mode parser.Mode) (*ast.File, error) {
	src, err := vfs.ReadFile(c.fs, filename)
	if err != nil {
		return nil, err
	}
	return parser.ParseFile(fset, filename, src, mode)
}

func (c *Corpus) parseFiles(fset *token.FileSet, abspath string, localnames []string) (map[string]*ast.File, error) {
	files := make(map[string]*ast.File)
	for _, f := range localnames {
		absname := pathpkg.Join(abspath, f)
		file, err := c.parseFile(fset, absname, parser.ParseComments)
		if err != nil {
			return nil, err
		}
		files[absname] = file
	}

	return files, nil
}
