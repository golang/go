// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package format implements standard formatting of Go source.
package format

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io"
	"strings"
)

var config = printer.Config{Mode: printer.UseSpaces | printer.TabIndent, Tabwidth: 8}

// Node formats node in canonical gofmt style and writes the result to dst.
//
// The node type must be *ast.File, *printer.CommentedNode, []ast.Decl,
// []ast.Stmt, or assignment-compatible to ast.Expr, ast.Decl, ast.Spec,
// or ast.Stmt. Node does not modify node. Imports are not sorted for
// nodes representing partial source files (i.e., if the node is not an
// *ast.File or a *printer.CommentedNode not wrapping an *ast.File).
//
// The function may return early (before the entire result is written)
// and return a formatting error, for instance due to an incorrect AST.
//
func Node(dst io.Writer, fset *token.FileSet, node interface{}) error {
	// Determine if we have a complete source file (file != nil).
	var file *ast.File
	var cnode *printer.CommentedNode
	switch n := node.(type) {
	case *ast.File:
		file = n
	case *printer.CommentedNode:
		if f, ok := n.Node.(*ast.File); ok {
			file = f
			cnode = n
		}
	}

	// Sort imports if necessary.
	if file != nil && hasUnsortedImports(file) {
		// Make a copy of the AST because ast.SortImports is destructive.
		// TODO(gri) Do this more efficiently.
		var buf bytes.Buffer
		err := config.Fprint(&buf, fset, file)
		if err != nil {
			return err
		}
		file, err = parser.ParseFile(fset, "", buf.Bytes(), parser.ParseComments)
		if err != nil {
			// We should never get here. If we do, provide good diagnostic.
			return fmt.Errorf("format.Node internal error (%s)", err)
		}
		ast.SortImports(fset, file)

		// Use new file with sorted imports.
		node = file
		if cnode != nil {
			node = &printer.CommentedNode{Node: file, Comments: cnode.Comments}
		}
	}

	return config.Fprint(dst, fset, node)
}

// Source formats src in canonical gofmt style and returns the result
// or an (I/O or syntax) error. src is expected to be a syntactically
// correct Go source file, or a list of Go declarations or statements.
//
// If src is a partial source file, the leading and trailing space of src
// is applied to the result (such that it has the same leading and trailing
// space as src), and the result is indented by the same amount as the first
// line of src containing code. Imports are not sorted for partial source files.
//
func Source(src []byte) ([]byte, error) {
	fset := token.NewFileSet()
	node, err := parse(fset, src)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	if file, ok := node.(*ast.File); ok {
		// Complete source file.
		ast.SortImports(fset, file)
		err := config.Fprint(&buf, fset, file)
		if err != nil {
			return nil, err
		}

	} else {
		// Partial source file.
		// Determine and prepend leading space.
		i, j := 0, 0
		for j < len(src) && isSpace(src[j]) {
			if src[j] == '\n' {
				i = j + 1 // index of last line in leading space
			}
			j++
		}
		buf.Write(src[:i])

		// Determine indentation of first code line.
		// Spaces are ignored unless there are no tabs,
		// in which case spaces count as one tab.
		indent := 0
		hasSpace := false
		for _, b := range src[i:j] {
			switch b {
			case ' ':
				hasSpace = true
			case '\t':
				indent++
			}
		}
		if indent == 0 && hasSpace {
			indent = 1
		}

		// Format the source.
		cfg := config
		cfg.Indent = indent
		err := cfg.Fprint(&buf, fset, node)
		if err != nil {
			return nil, err
		}

		// Determine and append trailing space.
		i = len(src)
		for i > 0 && isSpace(src[i-1]) {
			i--
		}
		buf.Write(src[i:])
	}

	return buf.Bytes(), nil
}

func hasUnsortedImports(file *ast.File) bool {
	for _, d := range file.Decls {
		d, ok := d.(*ast.GenDecl)
		if !ok || d.Tok != token.IMPORT {
			// Not an import declaration, so we're done.
			// Imports are always first.
			return false
		}
		if d.Lparen.IsValid() {
			// For now assume all grouped imports are unsorted.
			// TODO(gri) Should check if they are sorted already.
			return true
		}
		// Ungrouped imports are sorted by default.
	}
	return false
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func parse(fset *token.FileSet, src []byte) (interface{}, error) {
	// Try as a complete source file.
	file, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err == nil {
		return file, nil
	}
	// If the source is missing a package clause, try as a source fragment; otherwise fail.
	if !strings.Contains(err.Error(), "expected 'package'") {
		return nil, err
	}

	// Try as a declaration list by prepending a package clause in front of src.
	// Use ';' not '\n' to keep line numbers intact.
	psrc := append([]byte("package p;"), src...)
	file, err = parser.ParseFile(fset, "", psrc, parser.ParseComments)
	if err == nil {
		return file.Decls, nil
	}
	// If the source is missing a declaration, try as a statement list; otherwise fail.
	if !strings.Contains(err.Error(), "expected declaration") {
		return nil, err
	}

	// Try as statement list by wrapping a function around src.
	fsrc := append(append([]byte("package p; func _() {"), src...), '}')
	file, err = parser.ParseFile(fset, "", fsrc, parser.ParseComments)
	if err == nil {
		return file.Decls[0].(*ast.FuncDecl).Body.List, nil
	}

	// Failed, and out of options.
	return nil, err
}
