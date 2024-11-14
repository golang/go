// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package analysisutil defines various helper functions
// used by two or more packages beneath go/analysis.
package analysisutil

import (
	"bytes"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"os"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/aliases"
	"golang.org/x/tools/internal/analysisinternal"
)

// Format returns a string representation of the expression.
func Format(fset *token.FileSet, x ast.Expr) string {
	var b bytes.Buffer
	printer.Fprint(&b, fset, x)
	return b.String()
}

// HasSideEffects reports whether evaluation of e has side effects.
func HasSideEffects(info *types.Info, e ast.Expr) bool {
	safe := true
	ast.Inspect(e, func { node ->
		switch n := node.(type) {
		case *ast.CallExpr:
			typVal := info.Types[n.Fun]
			switch {
			case typVal.IsType():
				// Type conversion, which is safe.
			case typVal.IsBuiltin():
				// Builtin func, conservatively assumed to not
				// be safe for now.
				safe = false
				return false
			default:
				// A non-builtin func or method call.
				// Conservatively assume that all of them have
				// side effects for now.
				safe = false
				return false
			}
		case *ast.UnaryExpr:
			if n.Op == token.ARROW {
				safe = false
				return false
			}
		}
		return true
	})
	return !safe
}

// ReadFile reads a file and adds it to the FileSet
// so that we can report errors against it using lineStart.
func ReadFile(pass *analysis.Pass, filename string) ([]byte, *token.File, error) {
	readFile := pass.ReadFile
	if readFile == nil {
		readFile = os.ReadFile
	}
	content, err := readFile(filename)
	if err != nil {
		return nil, nil, err
	}
	tf := pass.Fset.AddFile(filename, -1, len(content))
	tf.SetLinesForContent(content)
	return content, tf, nil
}

// LineStart returns the position of the start of the specified line
// within file f, or NoPos if there is no line of that number.
func LineStart(f *token.File, line int) token.Pos {
	// Use binary search to find the start offset of this line.
	//
	// TODO(adonovan): eventually replace this function with the
	// simpler and more efficient (*go/token.File).LineStart, added
	// in go1.12.

	min := 0        // inclusive
	max := f.Size() // exclusive
	for {
		offset := (min + max) / 2
		pos := f.Pos(offset)
		posn := f.Position(pos)
		if posn.Line == line {
			return pos - (token.Pos(posn.Column) - 1)
		}

		if min+1 >= max {
			return token.NoPos
		}

		if posn.Line < line {
			min = offset
		} else {
			max = offset
		}
	}
}

// Imports returns true if path is imported by pkg.
func Imports(pkg *types.Package, path string) bool {
	for _, imp := range pkg.Imports() {
		if imp.Path() == path {
			return true
		}
	}
	return false
}

// IsNamedType reports whether t is the named type with the given package path
// and one of the given names.
// This function avoids allocating the concatenation of "pkg.Name",
// which is important for the performance of syntax matching.
func IsNamedType(t types.Type, pkgPath string, names ...string) bool {
	n, ok := aliases.Unalias(t).(*types.Named)
	if !ok {
		return false
	}
	obj := n.Obj()
	if obj == nil || obj.Pkg() == nil || obj.Pkg().Path() != pkgPath {
		return false
	}
	name := obj.Name()
	for _, n := range names {
		if name == n {
			return true
		}
	}
	return false
}

// IsFunctionNamed reports whether f is a top-level function defined in the
// given package and has one of the given names.
// It returns false if f is nil or a method.
func IsFunctionNamed(f *types.Func, pkgPath string, names ...string) bool {
	if f == nil {
		return false
	}
	if f.Pkg() == nil || f.Pkg().Path() != pkgPath {
		return false
	}
	if f.Type().(*types.Signature).Recv() != nil {
		return false
	}
	for _, n := range names {
		if f.Name() == n {
			return true
		}
	}
	return false
}

var MustExtractDoc = analysisinternal.MustExtractDoc
