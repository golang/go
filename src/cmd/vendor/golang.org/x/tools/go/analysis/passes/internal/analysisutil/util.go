// Package analysisutil defines various helper functions
// used by two or more packages beneath go/analysis.
package analysisutil

import (
	"bytes"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"io/ioutil"
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
	ast.Inspect(e, func(node ast.Node) bool {
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

// Unparen returns e with any enclosing parentheses stripped.
func Unparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}

// ReadFile reads a file and adds it to the FileSet
// so that we can report errors against it using lineStart.
func ReadFile(fset *token.FileSet, filename string) ([]byte, *token.File, error) {
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, err
	}
	tf := fset.AddFile(filename, -1, len(content))
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
