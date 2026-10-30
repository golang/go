// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gofixdirective searches for and validates go:fix directives. The
// go/analysis/passes/inline package uses findgofix to perform inlining.
// The go/analysis/passes/gofix package uses findgofix to check for problems
// with go:fix directives.
//
// gofixdirective is separate from gofix to avoid depending on refactor/inline,
// which is large.
package gofixdirective

// This package is tested by go/analysis/passes/inline.

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/inspector"
	internalastutil "golang.org/x/tools/internal/astutil"
)

// A Handler handles language entities with go:fix directives.
type Handler interface {
	HandleFunc(*ast.FuncDecl)
	HandleAlias(*ast.TypeSpec)
	HandleConst(name, rhs *ast.Ident)
}

// Find finds functions and constants annotated with an appropriate "//go:fix"
// comment (the syntax proposed by #32816), and calls handler methods for each one.
// h may be nil.
func Find(pass *analysis.Pass, root inspector.Cursor, h Handler) {
	for cur := range root.Preorder((*ast.FuncDecl)(nil), (*ast.GenDecl)(nil)) {
		switch decl := cur.Node().(type) {
		case *ast.FuncDecl:
			findFunc(decl, h)

		case *ast.GenDecl:
			if decl.Tok != token.CONST && decl.Tok != token.TYPE {
				continue
			}
			declInline := hasFixInline(decl.Doc)
			// Accept inline directives on the entire decl as well as individual specs.
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec: // Tok == TYPE
					findAlias(pass, spec, declInline, h)

				case *ast.ValueSpec: // Tok == CONST
					findConst(pass, spec, declInline, h)
				}
			}
		}
	}
}

func findFunc(decl *ast.FuncDecl, h Handler) {
	if !hasFixInline(decl.Doc) {
		return
	}
	if h != nil {
		h.HandleFunc(decl)
	}
}

func findAlias(pass *analysis.Pass, spec *ast.TypeSpec, declInline bool, h Handler) {
	if !declInline && !hasFixInline(spec.Doc) {
		return
	}
	if !spec.Assign.IsValid() {
		pass.Reportf(spec.Pos(), "invalid //go:fix inline directive: not a type alias")
		return
	}

	// Disallow inlines of type expressions containing array types.
	// Given an array type like [N]int where N is a named constant, go/types provides
	// only the value of the constant as an int64. So inlining A in this code:
	//
	//    const N = 5
	//    type A = [N]int
	//
	// would result in [5]int, breaking the connection with N.
	for n := range ast.Preorder(spec.Type) {
		if ar, ok := n.(*ast.ArrayType); ok && ar.Len != nil {
			// Make an exception when the array length is a literal int.
			if lit, ok := ast.Unparen(ar.Len).(*ast.BasicLit); ok && lit.Kind == token.INT {
				continue
			}
			pass.Reportf(spec.Pos(), "invalid //go:fix inline directive: array types not supported")
			return
		}
	}
	if h != nil {
		h.HandleAlias(spec)
	}
}

func findConst(pass *analysis.Pass, spec *ast.ValueSpec, declInline bool, h Handler) {
	specInline := hasFixInline(spec.Doc)
	if declInline || specInline {
		for i, nameIdent := range spec.Names {
			if i >= len(spec.Values) {
				// Possible following an iota.
				break
			}
			var rhsIdent *ast.Ident
			switch val := spec.Values[i].(type) {
			case *ast.Ident:
				// Constants defined with the predeclared iota cannot be inlined.
				if pass.TypesInfo.Uses[val] == builtinIota {
					pass.Reportf(val.Pos(), "invalid //go:fix inline directive: const value is iota")
					return
				}
				rhsIdent = val
			case *ast.SelectorExpr:
				rhsIdent = val.Sel
			default:
				pass.Reportf(val.Pos(), "invalid //go:fix inline directive: const value is not the name of another constant")
				return
			}
			if h != nil {
				h.HandleConst(nameIdent, rhsIdent)
			}
		}
	}
}

// hasFixInline reports the presence of a "//go:fix inline" directive
// in the comments.
func hasFixInline(cg *ast.CommentGroup) bool {
	for _, d := range internalastutil.Directives(cg) {
		if d.Tool == "go" && d.Name == "fix" && d.Args == "inline" {
			return true
		}
	}
	return false
}

var builtinIota = types.Universe.Lookup("iota")
