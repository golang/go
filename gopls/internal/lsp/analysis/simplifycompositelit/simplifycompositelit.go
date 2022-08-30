// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package simplifycompositelit defines an Analyzer that simplifies composite literals.
// https://github.com/golang/go/blob/master/src/cmd/gofmt/simplify.go
// https://golang.org/cmd/gofmt/#hdr-The_simplify_command
package simplifycompositelit

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"reflect"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `check for composite literal simplifications

An array, slice, or map composite literal of the form:
	[]T{T{}, T{}}
will be simplified to:
	[]T{{}, {}}

This is one of the simplifications that "gofmt -s" applies.`

var Analyzer = &analysis.Analyzer{
	Name:     "simplifycompositelit",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{(*ast.CompositeLit)(nil)}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		expr := n.(*ast.CompositeLit)

		outer := expr
		var keyType, eltType ast.Expr
		switch typ := outer.Type.(type) {
		case *ast.ArrayType:
			eltType = typ.Elt
		case *ast.MapType:
			keyType = typ.Key
			eltType = typ.Value
		}

		if eltType == nil {
			return
		}
		var ktyp reflect.Value
		if keyType != nil {
			ktyp = reflect.ValueOf(keyType)
		}
		typ := reflect.ValueOf(eltType)
		for _, x := range outer.Elts {
			// look at value of indexed/named elements
			if t, ok := x.(*ast.KeyValueExpr); ok {
				if keyType != nil {
					simplifyLiteral(pass, ktyp, keyType, t.Key)
				}
				x = t.Value
			}
			simplifyLiteral(pass, typ, eltType, x)
		}
	})
	return nil, nil
}

func simplifyLiteral(pass *analysis.Pass, typ reflect.Value, astType, x ast.Expr) {
	// if the element is a composite literal and its literal type
	// matches the outer literal's element type exactly, the inner
	// literal type may be omitted
	if inner, ok := x.(*ast.CompositeLit); ok && match(typ, reflect.ValueOf(inner.Type)) {
		var b bytes.Buffer
		printer.Fprint(&b, pass.Fset, inner.Type)
		createDiagnostic(pass, inner.Type.Pos(), inner.Type.End(), b.String())
	}
	// if the outer literal's element type is a pointer type *T
	// and the element is & of a composite literal of type T,
	// the inner &T may be omitted.
	if ptr, ok := astType.(*ast.StarExpr); ok {
		if addr, ok := x.(*ast.UnaryExpr); ok && addr.Op == token.AND {
			if inner, ok := addr.X.(*ast.CompositeLit); ok {
				if match(reflect.ValueOf(ptr.X), reflect.ValueOf(inner.Type)) {
					var b bytes.Buffer
					printer.Fprint(&b, pass.Fset, inner.Type)
					// Account for the & by subtracting 1 from typ.Pos().
					createDiagnostic(pass, inner.Type.Pos()-1, inner.Type.End(), "&"+b.String())
				}
			}
		}
	}
}

func createDiagnostic(pass *analysis.Pass, start, end token.Pos, typ string) {
	pass.Report(analysis.Diagnostic{
		Pos:     start,
		End:     end,
		Message: "redundant type from array, slice, or map composite literal",
		SuggestedFixes: []analysis.SuggestedFix{{
			Message: fmt.Sprintf("Remove '%s'", typ),
			TextEdits: []analysis.TextEdit{{
				Pos:     start,
				End:     end,
				NewText: []byte{},
			}},
		}},
	})
}

// match reports whether pattern matches val,
// recording wildcard submatches in m.
// If m == nil, match checks whether pattern == val.
// from https://github.com/golang/go/blob/26154f31ad6c801d8bad5ef58df1e9263c6beec7/src/cmd/gofmt/rewrite.go#L160
func match(pattern, val reflect.Value) bool {
	// Otherwise, pattern and val must match recursively.
	if !pattern.IsValid() || !val.IsValid() {
		return !pattern.IsValid() && !val.IsValid()
	}
	if pattern.Type() != val.Type() {
		return false
	}

	// Special cases.
	switch pattern.Type() {
	case identType:
		// For identifiers, only the names need to match
		// (and none of the other *ast.Object information).
		// This is a common case, handle it all here instead
		// of recursing down any further via reflection.
		p := pattern.Interface().(*ast.Ident)
		v := val.Interface().(*ast.Ident)
		return p == nil && v == nil || p != nil && v != nil && p.Name == v.Name
	case objectPtrType, positionType:
		// object pointers and token positions always match
		return true
	case callExprType:
		// For calls, the Ellipsis fields (token.Position) must
		// match since that is how f(x) and f(x...) are different.
		// Check them here but fall through for the remaining fields.
		p := pattern.Interface().(*ast.CallExpr)
		v := val.Interface().(*ast.CallExpr)
		if p.Ellipsis.IsValid() != v.Ellipsis.IsValid() {
			return false
		}
	}

	p := reflect.Indirect(pattern)
	v := reflect.Indirect(val)
	if !p.IsValid() || !v.IsValid() {
		return !p.IsValid() && !v.IsValid()
	}

	switch p.Kind() {
	case reflect.Slice:
		if p.Len() != v.Len() {
			return false
		}
		for i := 0; i < p.Len(); i++ {
			if !match(p.Index(i), v.Index(i)) {
				return false
			}
		}
		return true

	case reflect.Struct:
		for i := 0; i < p.NumField(); i++ {
			if !match(p.Field(i), v.Field(i)) {
				return false
			}
		}
		return true

	case reflect.Interface:
		return match(p.Elem(), v.Elem())
	}

	// Handle token integers, etc.
	return p.Interface() == v.Interface()
}

// Values/types for special cases.
var (
	identType     = reflect.TypeOf((*ast.Ident)(nil))
	objectPtrType = reflect.TypeOf((*ast.Object)(nil))
	positionType  = reflect.TypeOf(token.NoPos)
	callExprType  = reflect.TypeOf((*ast.CallExpr)(nil))
)
