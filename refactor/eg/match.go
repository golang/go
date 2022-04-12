// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eg

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"log"
	"os"
	"reflect"

	"golang.org/x/tools/go/ast/astutil"
)

// matchExpr reports whether pattern x matches y.
//
// If tr.allowWildcards, Idents in x that refer to parameters are
// treated as wildcards, and match any y that is assignable to the
// parameter type; matchExpr records this correspondence in tr.env.
// Otherwise, matchExpr simply reports whether the two trees are
// equivalent.
//
// A wildcard appearing more than once in the pattern must
// consistently match the same tree.
func (tr *Transformer) matchExpr(x, y ast.Expr) bool {
	if x == nil && y == nil {
		return true
	}
	if x == nil || y == nil {
		return false
	}
	x = unparen(x)
	y = unparen(y)

	// Is x a wildcard?  (a reference to a 'before' parameter)
	if xobj, ok := tr.wildcardObj(x); ok {
		return tr.matchWildcard(xobj, y)
	}

	// Object identifiers (including pkg-qualified ones)
	// are handled semantically, not syntactically.
	xobj := isRef(x, tr.info)
	yobj := isRef(y, tr.info)
	if xobj != nil {
		return xobj == yobj
	}
	if yobj != nil {
		return false
	}

	// TODO(adonovan): audit: we cannot assume these ast.Exprs
	// contain non-nil pointers.  e.g. ImportSpec.Name may be a
	// nil *ast.Ident.

	if reflect.TypeOf(x) != reflect.TypeOf(y) {
		return false
	}
	switch x := x.(type) {
	case *ast.Ident:
		log.Fatalf("unexpected Ident: %s", astString(tr.fset, x))

	case *ast.BasicLit:
		y := y.(*ast.BasicLit)
		xval := constant.MakeFromLiteral(x.Value, x.Kind, 0)
		yval := constant.MakeFromLiteral(y.Value, y.Kind, 0)
		return constant.Compare(xval, token.EQL, yval)

	case *ast.FuncLit:
		// func literals (and thus statement syntax) never match.
		return false

	case *ast.CompositeLit:
		y := y.(*ast.CompositeLit)
		return (x.Type == nil) == (y.Type == nil) &&
			(x.Type == nil || tr.matchType(x.Type, y.Type)) &&
			tr.matchExprs(x.Elts, y.Elts)

	case *ast.SelectorExpr:
		y := y.(*ast.SelectorExpr)
		return tr.matchSelectorExpr(x, y) &&
			tr.info.Selections[x].Obj() == tr.info.Selections[y].Obj()

	case *ast.IndexExpr:
		y := y.(*ast.IndexExpr)
		return tr.matchExpr(x.X, y.X) &&
			tr.matchExpr(x.Index, y.Index)

	case *ast.SliceExpr:
		y := y.(*ast.SliceExpr)
		return tr.matchExpr(x.X, y.X) &&
			tr.matchExpr(x.Low, y.Low) &&
			tr.matchExpr(x.High, y.High) &&
			tr.matchExpr(x.Max, y.Max) &&
			x.Slice3 == y.Slice3

	case *ast.TypeAssertExpr:
		y := y.(*ast.TypeAssertExpr)
		return tr.matchExpr(x.X, y.X) &&
			tr.matchType(x.Type, y.Type)

	case *ast.CallExpr:
		y := y.(*ast.CallExpr)
		match := tr.matchExpr // function call
		if tr.info.Types[x.Fun].IsType() {
			match = tr.matchType // type conversion
		}
		return x.Ellipsis.IsValid() == y.Ellipsis.IsValid() &&
			match(x.Fun, y.Fun) &&
			tr.matchExprs(x.Args, y.Args)

	case *ast.StarExpr:
		y := y.(*ast.StarExpr)
		return tr.matchExpr(x.X, y.X)

	case *ast.UnaryExpr:
		y := y.(*ast.UnaryExpr)
		return x.Op == y.Op &&
			tr.matchExpr(x.X, y.X)

	case *ast.BinaryExpr:
		y := y.(*ast.BinaryExpr)
		return x.Op == y.Op &&
			tr.matchExpr(x.X, y.X) &&
			tr.matchExpr(x.Y, y.Y)

	case *ast.KeyValueExpr:
		y := y.(*ast.KeyValueExpr)
		return tr.matchExpr(x.Key, y.Key) &&
			tr.matchExpr(x.Value, y.Value)
	}

	panic(fmt.Sprintf("unhandled AST node type: %T", x))
}

func (tr *Transformer) matchExprs(xx, yy []ast.Expr) bool {
	if len(xx) != len(yy) {
		return false
	}
	for i := range xx {
		if !tr.matchExpr(xx[i], yy[i]) {
			return false
		}
	}
	return true
}

// matchType reports whether the two type ASTs denote identical types.
func (tr *Transformer) matchType(x, y ast.Expr) bool {
	tx := tr.info.Types[x].Type
	ty := tr.info.Types[y].Type
	return types.Identical(tx, ty)
}

func (tr *Transformer) wildcardObj(x ast.Expr) (*types.Var, bool) {
	if x, ok := x.(*ast.Ident); ok && x != nil && tr.allowWildcards {
		if xobj, ok := tr.info.Uses[x].(*types.Var); ok && tr.wildcards[xobj] {
			return xobj, true
		}
	}
	return nil, false
}

func (tr *Transformer) matchSelectorExpr(x, y *ast.SelectorExpr) bool {
	if xobj, ok := tr.wildcardObj(x.X); ok {
		field := x.Sel.Name
		yt := tr.info.TypeOf(y.X)
		o, _, _ := types.LookupFieldOrMethod(yt, true, tr.currentPkg, field)
		if o != nil {
			tr.env[xobj.Name()] = y.X // record binding
			return true
		}
	}
	return tr.matchExpr(x.X, y.X)
}

func (tr *Transformer) matchWildcard(xobj *types.Var, y ast.Expr) bool {
	name := xobj.Name()

	if tr.verbose {
		fmt.Fprintf(os.Stderr, "%s: wildcard %s -> %s?: ",
			tr.fset.Position(y.Pos()), name, astString(tr.fset, y))
	}

	// Check that y is assignable to the declared type of the param.
	yt := tr.info.TypeOf(y)
	if yt == nil {
		// y has no type.
		// Perhaps it is an *ast.Ellipsis in [...]T{}, or
		// an *ast.KeyValueExpr in T{k: v}.
		// Clearly these pseudo-expressions cannot match a
		// wildcard, but it would nice if we had a way to ignore
		// the difference between T{v} and T{k:v} for structs.
		return false
	}
	if !types.AssignableTo(yt, xobj.Type()) {
		if tr.verbose {
			fmt.Fprintf(os.Stderr, "%s not assignable to %s\n", yt, xobj.Type())
		}
		return false
	}

	// A wildcard matches any expression.
	// If it appears multiple times in the pattern, it must match
	// the same expression each time.
	if old, ok := tr.env[name]; ok {
		// found existing binding
		tr.allowWildcards = false
		r := tr.matchExpr(old, y)
		if tr.verbose {
			fmt.Fprintf(os.Stderr, "%t secondary match, primary was %s\n",
				r, astString(tr.fset, old))
		}
		tr.allowWildcards = true
		return r
	}

	if tr.verbose {
		fmt.Fprintf(os.Stderr, "primary match\n")
	}

	tr.env[name] = y // record binding
	return true
}

// -- utilities --------------------------------------------------------

func unparen(e ast.Expr) ast.Expr { return astutil.Unparen(e) }

// isRef returns the object referred to by this (possibly qualified)
// identifier, or nil if the node is not a referring identifier.
func isRef(n ast.Node, info *types.Info) types.Object {
	switch n := n.(type) {
	case *ast.Ident:
		return info.Uses[n]

	case *ast.SelectorExpr:
		if _, ok := info.Selections[n]; !ok {
			// qualified ident
			return info.Uses[n.Sel]
		}
	}
	return nil
}
