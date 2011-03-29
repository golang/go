// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
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

// rewrite walks the AST x, calling visit(y) for each node y in the tree but
// also with a pointer to each ast.Expr, in a bottom-up traversal.
func rewrite(x interface{}, visit func(interface{})) {
	switch n := x.(type) {
	case *ast.Expr:
		rewrite(*n, visit)

	// everything else just recurses
	default:
		panic(fmt.Errorf("unexpected type %T in walk", x, visit))

	case nil:

	// These are ordered and grouped to match ../../pkg/go/ast/ast.go
	case *ast.Field:
		rewrite(&n.Type, visit)
	case *ast.FieldList:
		for _, field := range n.List {
			rewrite(field, visit)
		}
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.Ellipsis:
	case *ast.BasicLit:
	case *ast.FuncLit:
		rewrite(n.Type, visit)
		rewrite(n.Body, visit)
	case *ast.CompositeLit:
		rewrite(&n.Type, visit)
		rewrite(n.Elts, visit)
	case *ast.ParenExpr:
		rewrite(&n.X, visit)
	case *ast.SelectorExpr:
		rewrite(&n.X, visit)
	case *ast.IndexExpr:
		rewrite(&n.X, visit)
		rewrite(&n.Index, visit)
	case *ast.SliceExpr:
		rewrite(&n.X, visit)
		if n.Low != nil {
			rewrite(&n.Low, visit)
		}
		if n.High != nil {
			rewrite(&n.High, visit)
		}
	case *ast.TypeAssertExpr:
		rewrite(&n.X, visit)
		rewrite(&n.Type, visit)
	case *ast.CallExpr:
		rewrite(&n.Fun, visit)
		rewrite(n.Args, visit)
	case *ast.StarExpr:
		rewrite(&n.X, visit)
	case *ast.UnaryExpr:
		rewrite(&n.X, visit)
	case *ast.BinaryExpr:
		rewrite(&n.X, visit)
		rewrite(&n.Y, visit)
	case *ast.KeyValueExpr:
		rewrite(&n.Key, visit)
		rewrite(&n.Value, visit)

	case *ast.ArrayType:
		rewrite(&n.Len, visit)
		rewrite(&n.Elt, visit)
	case *ast.StructType:
		rewrite(n.Fields, visit)
	case *ast.FuncType:
		rewrite(n.Params, visit)
		if n.Results != nil {
			rewrite(n.Results, visit)
		}
	case *ast.InterfaceType:
		rewrite(n.Methods, visit)
	case *ast.MapType:
		rewrite(&n.Key, visit)
		rewrite(&n.Value, visit)
	case *ast.ChanType:
		rewrite(&n.Value, visit)

	case *ast.BadStmt:
	case *ast.DeclStmt:
		rewrite(n.Decl, visit)
	case *ast.EmptyStmt:
	case *ast.LabeledStmt:
		rewrite(n.Stmt, visit)
	case *ast.ExprStmt:
		rewrite(&n.X, visit)
	case *ast.SendStmt:
		rewrite(&n.Chan, visit)
		rewrite(&n.Value, visit)
	case *ast.IncDecStmt:
		rewrite(&n.X, visit)
	case *ast.AssignStmt:
		rewrite(n.Lhs, visit)
		if len(n.Lhs) == 2 && len(n.Rhs) == 1 {
			rewrite(n.Rhs, visit)
		} else {
			rewrite(n.Rhs, visit)
		}
	case *ast.GoStmt:
		rewrite(n.Call, visit)
	case *ast.DeferStmt:
		rewrite(n.Call, visit)
	case *ast.ReturnStmt:
		rewrite(n.Results, visit)
	case *ast.BranchStmt:
	case *ast.BlockStmt:
		rewrite(n.List, visit)
	case *ast.IfStmt:
		rewrite(n.Init, visit)
		rewrite(&n.Cond, visit)
		rewrite(n.Body, visit)
		rewrite(n.Else, visit)
	case *ast.CaseClause:
		rewrite(n.List, visit)
		rewrite(n.Body, visit)
	case *ast.SwitchStmt:
		rewrite(n.Init, visit)
		rewrite(&n.Tag, visit)
		rewrite(n.Body, visit)
	case *ast.TypeSwitchStmt:
		rewrite(n.Init, visit)
		rewrite(n.Assign, visit)
		rewrite(n.Body, visit)
	case *ast.CommClause:
		rewrite(n.Comm, visit)
		rewrite(n.Body, visit)
	case *ast.SelectStmt:
		rewrite(n.Body, visit)
	case *ast.ForStmt:
		rewrite(n.Init, visit)
		rewrite(&n.Cond, visit)
		rewrite(n.Post, visit)
		rewrite(n.Body, visit)
	case *ast.RangeStmt:
		rewrite(&n.Key, visit)
		rewrite(&n.Value, visit)
		rewrite(&n.X, visit)
		rewrite(n.Body, visit)

	case *ast.ImportSpec:
	case *ast.ValueSpec:
		rewrite(&n.Type, visit)
		rewrite(n.Values, visit)
	case *ast.TypeSpec:
		rewrite(&n.Type, visit)

	case *ast.BadDecl:
	case *ast.GenDecl:
		rewrite(n.Specs, visit)
	case *ast.FuncDecl:
		if n.Recv != nil {
			rewrite(n.Recv, visit)
		}
		rewrite(n.Type, visit)
		if n.Body != nil {
			rewrite(n.Body, visit)
		}

	case *ast.File:
		rewrite(n.Decls, visit)

	case *ast.Package:
		for _, file := range n.Files {
			rewrite(file, visit)
		}

	case []ast.Decl:
		for _, d := range n {
			rewrite(d, visit)
		}
	case []ast.Expr:
		for i := range n {
			rewrite(&n[i], visit)
		}
	case []ast.Stmt:
		for _, s := range n {
			rewrite(s, visit)
		}
	case []ast.Spec:
		for _, s := range n {
			rewrite(s, visit)
		}
	}
	visit(x)
}

func imports(f *ast.File, path string) bool {
	for _, decl := range f.Decls {
		d, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range d.Specs {
			s, ok := spec.(*ast.ImportSpec)
			if !ok {
				continue
			}
			if string(s.Path.Value) == `"`+path+`"` {
				return true
			}
		}
	}
	return false
}

func isPkgDot(t ast.Expr, pkg, name string) bool {
	sel, ok := t.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	return isTopName(sel.X, pkg) && sel.Sel.String() == name
}

func isPtrPkgDot(t ast.Expr, pkg, name string) bool {
	ptr, ok := t.(*ast.StarExpr)
	if !ok {
		return false
	}
	return isPkgDot(ptr.X, pkg, name)
}

func isTopName(n ast.Expr, name string) bool {
	id, ok := n.(*ast.Ident)
	if !ok {
		return false
	}
	return id.Name == name && id.Obj == nil
}

func isName(n ast.Expr, name string) bool {
	id, ok := n.(*ast.Ident)
	if !ok {
		return false
	}
	return id.String() == name
}

func isCall(t ast.Expr, pkg, name string) bool {
	call, ok := t.(*ast.CallExpr)
	return ok && isPkgDot(call.Fun, pkg, name)
}

func refersTo(n ast.Node, x *ast.Ident) bool {
	id, ok := n.(*ast.Ident)
	if !ok {
		return false
	}
	return id.String() == x.String()
}

func isBlank(n ast.Expr) bool {
	return isName(n, "_")
}

func isEmptyString(n ast.Expr) bool {
	lit, ok := n.(*ast.BasicLit)
	if !ok {
		return false
	}
	if lit.Kind != token.STRING {
		return false
	}
	s := string(lit.Value)
	return s == `""` || s == "``"
}

func warn(pos token.Pos, msg string, args ...interface{}) {
	if pos.IsValid() {
		msg = "%s: " + msg
		arg1 := []interface{}{fset.Position(pos).String()}
		args = append(arg1, args...)
	}
	fmt.Fprintf(os.Stderr, msg+"\n", args...)
}
