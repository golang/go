// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"strconv"
)

type fix struct {
	name string
	f    func(*ast.File) bool
	desc string
}

// main runs sort.Sort(fixes) after init process is done.
type fixlist []fix

func (f fixlist) Len() int           { return len(f) }
func (f fixlist) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }
func (f fixlist) Less(i, j int) bool { return f[i].name < f[j].name }

var fixes fixlist

func register(f fix) {
	fixes = append(fixes, f)
}

// walk traverses the AST x, calling visit(y) for each node y in the tree but
// also with a pointer to each ast.Expr, ast.Stmt, and *ast.BlockStmt,
// in a bottom-up traversal.
func walk(x interface{}, visit func(interface{})) {
	walkBeforeAfter(x, nop, visit)
}

func nop(interface{}) {}

// walkBeforeAfter is like walk but calls before(x) before traversing
// x's children and after(x) afterward.
func walkBeforeAfter(x interface{}, before, after func(interface{})) {
	before(x)

	switch n := x.(type) {
	default:
		panic(fmt.Errorf("unexpected type %T in walkBeforeAfter", x))

	case nil:

	// pointers to interfaces
	case *ast.Decl:
		walkBeforeAfter(*n, before, after)
	case *ast.Expr:
		walkBeforeAfter(*n, before, after)
	case *ast.Spec:
		walkBeforeAfter(*n, before, after)
	case *ast.Stmt:
		walkBeforeAfter(*n, before, after)

	// pointers to struct pointers
	case **ast.BlockStmt:
		walkBeforeAfter(*n, before, after)
	case **ast.CallExpr:
		walkBeforeAfter(*n, before, after)
	case **ast.FieldList:
		walkBeforeAfter(*n, before, after)
	case **ast.FuncType:
		walkBeforeAfter(*n, before, after)

	// pointers to slices
	case *[]ast.Stmt:
		walkBeforeAfter(*n, before, after)
	case *[]ast.Expr:
		walkBeforeAfter(*n, before, after)
	case *[]ast.Decl:
		walkBeforeAfter(*n, before, after)
	case *[]ast.Spec:
		walkBeforeAfter(*n, before, after)
	case *[]*ast.File:
		walkBeforeAfter(*n, before, after)

	// These are ordered and grouped to match ../../pkg/go/ast/ast.go
	case *ast.Field:
		walkBeforeAfter(&n.Type, before, after)
	case *ast.FieldList:
		for _, field := range n.List {
			walkBeforeAfter(field, before, after)
		}
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.Ellipsis:
	case *ast.BasicLit:
	case *ast.FuncLit:
		walkBeforeAfter(&n.Type, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.CompositeLit:
		walkBeforeAfter(&n.Type, before, after)
		walkBeforeAfter(&n.Elts, before, after)
	case *ast.ParenExpr:
		walkBeforeAfter(&n.X, before, after)
	case *ast.SelectorExpr:
		walkBeforeAfter(&n.X, before, after)
	case *ast.IndexExpr:
		walkBeforeAfter(&n.X, before, after)
		walkBeforeAfter(&n.Index, before, after)
	case *ast.SliceExpr:
		walkBeforeAfter(&n.X, before, after)
		if n.Low != nil {
			walkBeforeAfter(&n.Low, before, after)
		}
		if n.High != nil {
			walkBeforeAfter(&n.High, before, after)
		}
	case *ast.TypeAssertExpr:
		walkBeforeAfter(&n.X, before, after)
		walkBeforeAfter(&n.Type, before, after)
	case *ast.CallExpr:
		walkBeforeAfter(&n.Fun, before, after)
		walkBeforeAfter(&n.Args, before, after)
	case *ast.StarExpr:
		walkBeforeAfter(&n.X, before, after)
	case *ast.UnaryExpr:
		walkBeforeAfter(&n.X, before, after)
	case *ast.BinaryExpr:
		walkBeforeAfter(&n.X, before, after)
		walkBeforeAfter(&n.Y, before, after)
	case *ast.KeyValueExpr:
		walkBeforeAfter(&n.Key, before, after)
		walkBeforeAfter(&n.Value, before, after)

	case *ast.ArrayType:
		walkBeforeAfter(&n.Len, before, after)
		walkBeforeAfter(&n.Elt, before, after)
	case *ast.StructType:
		walkBeforeAfter(&n.Fields, before, after)
	case *ast.FuncType:
		walkBeforeAfter(&n.Params, before, after)
		if n.Results != nil {
			walkBeforeAfter(&n.Results, before, after)
		}
	case *ast.InterfaceType:
		walkBeforeAfter(&n.Methods, before, after)
	case *ast.MapType:
		walkBeforeAfter(&n.Key, before, after)
		walkBeforeAfter(&n.Value, before, after)
	case *ast.ChanType:
		walkBeforeAfter(&n.Value, before, after)

	case *ast.BadStmt:
	case *ast.DeclStmt:
		walkBeforeAfter(&n.Decl, before, after)
	case *ast.EmptyStmt:
	case *ast.LabeledStmt:
		walkBeforeAfter(&n.Stmt, before, after)
	case *ast.ExprStmt:
		walkBeforeAfter(&n.X, before, after)
	case *ast.SendStmt:
		walkBeforeAfter(&n.Chan, before, after)
		walkBeforeAfter(&n.Value, before, after)
	case *ast.IncDecStmt:
		walkBeforeAfter(&n.X, before, after)
	case *ast.AssignStmt:
		walkBeforeAfter(&n.Lhs, before, after)
		walkBeforeAfter(&n.Rhs, before, after)
	case *ast.GoStmt:
		walkBeforeAfter(&n.Call, before, after)
	case *ast.DeferStmt:
		walkBeforeAfter(&n.Call, before, after)
	case *ast.ReturnStmt:
		walkBeforeAfter(&n.Results, before, after)
	case *ast.BranchStmt:
	case *ast.BlockStmt:
		walkBeforeAfter(&n.List, before, after)
	case *ast.IfStmt:
		walkBeforeAfter(&n.Init, before, after)
		walkBeforeAfter(&n.Cond, before, after)
		walkBeforeAfter(&n.Body, before, after)
		walkBeforeAfter(&n.Else, before, after)
	case *ast.CaseClause:
		walkBeforeAfter(&n.List, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.SwitchStmt:
		walkBeforeAfter(&n.Init, before, after)
		walkBeforeAfter(&n.Tag, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.TypeSwitchStmt:
		walkBeforeAfter(&n.Init, before, after)
		walkBeforeAfter(&n.Assign, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.CommClause:
		walkBeforeAfter(&n.Comm, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.SelectStmt:
		walkBeforeAfter(&n.Body, before, after)
	case *ast.ForStmt:
		walkBeforeAfter(&n.Init, before, after)
		walkBeforeAfter(&n.Cond, before, after)
		walkBeforeAfter(&n.Post, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.RangeStmt:
		walkBeforeAfter(&n.Key, before, after)
		walkBeforeAfter(&n.Value, before, after)
		walkBeforeAfter(&n.X, before, after)
		walkBeforeAfter(&n.Body, before, after)

	case *ast.ImportSpec:
	case *ast.ValueSpec:
		walkBeforeAfter(&n.Type, before, after)
		walkBeforeAfter(&n.Values, before, after)
	case *ast.TypeSpec:
		walkBeforeAfter(&n.Type, before, after)

	case *ast.BadDecl:
	case *ast.GenDecl:
		walkBeforeAfter(&n.Specs, before, after)
	case *ast.FuncDecl:
		if n.Recv != nil {
			walkBeforeAfter(&n.Recv, before, after)
		}
		walkBeforeAfter(&n.Type, before, after)
		if n.Body != nil {
			walkBeforeAfter(&n.Body, before, after)
		}

	case *ast.File:
		walkBeforeAfter(&n.Decls, before, after)

	case *ast.Package:
		walkBeforeAfter(&n.Files, before, after)

	case []*ast.File:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	case []ast.Decl:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	case []ast.Expr:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	case []ast.Stmt:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	case []ast.Spec:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	}
	after(x)
}

// imports returns true if f imports path.
func imports(f *ast.File, path string) bool {
	for _, s := range f.Imports {
		t, err := strconv.Unquote(s.Path.Value)
		if err == nil && t == path {
			return true
		}
	}
	return false
}

// isPkgDot returns true if t is the expression "pkg.name"
// where pkg is an imported identifier.
func isPkgDot(t ast.Expr, pkg, name string) bool {
	sel, ok := t.(*ast.SelectorExpr)
	return ok && isTopName(sel.X, pkg) && sel.Sel.String() == name
}

// isPtrPkgDot returns true if f is the expression "*pkg.name"
// where pkg is an imported identifier.
func isPtrPkgDot(t ast.Expr, pkg, name string) bool {
	ptr, ok := t.(*ast.StarExpr)
	return ok && isPkgDot(ptr.X, pkg, name)
}

// isTopName returns true if n is a top-level unresolved identifier with the given name.
func isTopName(n ast.Expr, name string) bool {
	id, ok := n.(*ast.Ident)
	return ok && id.Name == name && id.Obj == nil
}

// isName returns true if n is an identifier with the given name.
func isName(n ast.Expr, name string) bool {
	id, ok := n.(*ast.Ident)
	return ok && id.String() == name
}

// isCall returns true if t is a call to pkg.name.
func isCall(t ast.Expr, pkg, name string) bool {
	call, ok := t.(*ast.CallExpr)
	return ok && isPkgDot(call.Fun, pkg, name)
}

// If n is an *ast.Ident, isIdent returns it; otherwise isIdent returns nil.
func isIdent(n interface{}) *ast.Ident {
	id, _ := n.(*ast.Ident)
	return id
}

// refersTo returns true if n is a reference to the same object as x.
func refersTo(n ast.Node, x *ast.Ident) bool {
	id, ok := n.(*ast.Ident)
	// The test of id.Name == x.Name handles top-level unresolved
	// identifiers, which all have Obj == nil.
	return ok && id.Obj == x.Obj && id.Name == x.Name
}

// isBlank returns true if n is the blank identifier.
func isBlank(n ast.Expr) bool {
	return isName(n, "_")
}

// isEmptyString returns true if n is an empty string literal.
func isEmptyString(n ast.Expr) bool {
	lit, ok := n.(*ast.BasicLit)
	return ok && lit.Kind == token.STRING && len(lit.Value) == 2
}

func warn(pos token.Pos, msg string, args ...interface{}) {
	if pos.IsValid() {
		msg = "%s: " + msg
		arg1 := []interface{}{fset.Position(pos).String()}
		args = append(arg1, args...)
	}
	fmt.Fprintf(os.Stderr, msg+"\n", args...)
}

// countUses returns the number of uses of the identifier x in scope.
func countUses(x *ast.Ident, scope []ast.Stmt) int {
	count := 0
	ff := func(n interface{}) {
		if n, ok := n.(ast.Node); ok && refersTo(n, x) {
			count++
		}
	}
	for _, n := range scope {
		walk(n, ff)
	}
	return count
}

// rewriteUses replaces all uses of the identifier x and !x in scope
// with f(x.Pos()) and fnot(x.Pos()).
func rewriteUses(x *ast.Ident, f, fnot func(token.Pos) ast.Expr, scope []ast.Stmt) {
	var lastF ast.Expr
	ff := func(n interface{}) {
		ptr, ok := n.(*ast.Expr)
		if !ok {
			return
		}
		nn := *ptr

		// The child node was just walked and possibly replaced.
		// If it was replaced and this is a negation, replace with fnot(p).
		not, ok := nn.(*ast.UnaryExpr)
		if ok && not.Op == token.NOT && not.X == lastF {
			*ptr = fnot(nn.Pos())
			return
		}
		if refersTo(nn, x) {
			lastF = f(nn.Pos())
			*ptr = lastF
		}
	}
	for _, n := range scope {
		walk(n, ff)
	}
}

// assignsTo returns true if any of the code in scope assigns to or takes the address of x.
func assignsTo(x *ast.Ident, scope []ast.Stmt) bool {
	assigned := false
	ff := func(n interface{}) {
		if assigned {
			return
		}
		switch n := n.(type) {
		case *ast.UnaryExpr:
			// use of &x
			if n.Op == token.AND && refersTo(n.X, x) {
				assigned = true
				return
			}
		case *ast.AssignStmt:
			for _, l := range n.Lhs {
				if refersTo(l, x) {
					assigned = true
					return
				}
			}
		}
	}
	for _, n := range scope {
		if assigned {
			break
		}
		walk(n, ff)
	}
	return assigned
}

// newPkgDot returns an ast.Expr referring to "pkg.name" at position pos.
func newPkgDot(pos token.Pos, pkg, name string) ast.Expr {
	return &ast.SelectorExpr{
		X: &ast.Ident{
			NamePos: pos,
			Name:    pkg,
		},
		Sel: &ast.Ident{
			NamePos: pos,
			Name:    name,
		},
	}
}
