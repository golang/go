// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains support functions for parsing .go files
// accessed via godoc's file system fs.

package godoc

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	pathpkg "path"

	"golang.org/x/tools/godoc/vfs"
)

var linePrefix = []byte("//line ")

// This function replaces source lines starting with "//line " with a blank line.
// It does this irrespective of whether the line is truly a line comment or not;
// e.g., the line may be inside a string, or a /*-style comment; however that is
// rather unlikely (proper testing would require a full Go scan which we want to
// avoid for performance).
func replaceLinePrefixCommentsWithBlankLine(src []byte) {
	for {
		i := bytes.Index(src, linePrefix)
		if i < 0 {
			break // we're done
		}
		// 0 <= i && i+len(linePrefix) <= len(src)
		if i == 0 || src[i-1] == '\n' {
			// at beginning of line: blank out line
			for i < len(src) && src[i] != '\n' {
				src[i] = ' '
				i++
			}
		} else {
			// not at beginning of line: skip over prefix
			i += len(linePrefix)
		}
		// i <= len(src)
		src = src[i:]
	}
}

func (c *Corpus) parseFile(fset *token.FileSet, filename string, mode parser.Mode) (*ast.File, error) {
	src, err := vfs.ReadFile(c.fs, filename)
	if err != nil {
		return nil, err
	}

	// Temporary ad-hoc fix for issue 5247.
	// TODO(gri,dmitshur) Remove this in favor of a better fix, eventually (see issue 32092).
	replaceLinePrefixCommentsWithBlankLine(src)

	return parser.ParseFile(fset, filename, src, mode)
}

func (c *Corpus) parseFiles(fset *token.FileSet, relpath string, abspath string, localnames []string) (map[string]*ast.File, error) {
	files := make(map[string]*ast.File)
	for _, f := range localnames {
		absname := pathpkg.Join(abspath, f)
		file, err := c.parseFile(fset, absname, parser.ParseComments)
		if err != nil {
			return nil, err
		}
		files[pathpkg.Join(relpath, f)] = file
	}

	return files, nil
}
