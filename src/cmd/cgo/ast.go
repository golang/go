// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse input AST and prepare Prog structure.

package main

import (
	"fmt";
	"go/ast";
	"go/doc";
	"go/parser";
	"go/scanner";
	"os";
)

// A Cref refers to an expression of the form C.xxx in the AST.
type Cref struct {
	Name		string;
	Expr		*ast.Expr;
	Context		string;	// "type", "expr", or "call"
	TypeName	bool;	// whether xxx is a C type name
	Type		*Type;	// the type of xxx
	FuncType	*FuncType;
}

// A Prog collects information about a cgo program.
type Prog struct {
	AST		*ast.File;	// parsed AST
	Preamble	string;		// C preamble (doc comment on import "C")
	PackagePath	string;
	Package		string;
	Crefs		[]*Cref;
	Typedef		map[string]ast.Expr;
	Vardef		map[string]*Type;
	Funcdef		map[string]*FuncType;
	PtrSize		int64;
	GccOptions	[]string;
}

// A Type collects information about a type in both the C and Go worlds.
type Type struct {
	Size	int64;
	Align	int64;
	C	string;
	Go	ast.Expr;
}

// A FuncType collects information about a function type in both the C and Go worlds.
type FuncType struct {
	Params	[]*Type;
	Result	*Type;
	Go	*ast.FuncType;
}

func openProg(name string) *Prog {
	p := new(Prog);
	var err os.Error;
	p.AST, err = parser.ParsePkgFile("", name, parser.ParseComments);
	if err != nil {
		if list, ok := err.(scanner.ErrorList); ok {
			// If err is a scanner.ErrorList, its String will print just
			// the first error and then (+n more errors).
			// Instead, turn it into a new Error that will return
			// details for all the errors.
			for _, e := range list {
				fmt.Fprintln(os.Stderr, e)
			}
			os.Exit(2);
		}
		fatal("parsing %s: %s", name, err);
	}
	p.Package = p.AST.Name.Value;

	// Find the import "C" line and get any extra C preamble.
	// Delete the import "C" line along the way.
	sawC := false;
	w := 0;
	for _, decl := range p.AST.Decls {
		d, ok := decl.(*ast.GenDecl);
		if !ok {
			p.AST.Decls[w] = decl;
			w++;
			continue;
		}
		ws := 0;
		for _, spec := range d.Specs {
			s, ok := spec.(*ast.ImportSpec);
			if !ok || len(s.Path) != 1 || string(s.Path[0].Value) != `"C"` {
				d.Specs[ws] = spec;
				ws++;
				continue;
			}
			sawC = true;
			if s.Name != nil {
				error(s.Path[0].Pos(), `cannot rename import "C"`)
			}
			if s.Doc != nil {
				p.Preamble += doc.CommentText(s.Doc) + "\n"
			} else if len(d.Specs) == 1 && d.Doc != nil {
				p.Preamble += doc.CommentText(d.Doc) + "\n"
			}
		}
		if ws == 0 {
			continue
		}
		d.Specs = d.Specs[0:ws];
		p.AST.Decls[w] = d;
		w++;
	}
	p.AST.Decls = p.AST.Decls[0:w];

	if !sawC {
		error(noPos, `cannot find import "C"`)
	}

	// Accumulate pointers to uses of C.x.
	p.Crefs = make([]*Cref, 0, 8);
	walk(p.AST, p, "prog");
	return p;
}

func walk(x interface{}, p *Prog, context string) {
	switch n := x.(type) {
	case *ast.Expr:
		if sel, ok := (*n).(*ast.SelectorExpr); ok {
			// For now, assume that the only instance of capital C is
			// when used as the imported package identifier.
			// The parser should take care of scoping in the future,
			// so that we will be able to distinguish a "top-level C"
			// from a local C.
			if l, ok := sel.X.(*ast.Ident); ok && l.Value == "C" {
				i := len(p.Crefs);
				if i >= cap(p.Crefs) {
					new := make([]*Cref, 2*i);
					for j, v := range p.Crefs {
						new[j] = v
					}
					p.Crefs = new;
				}
				p.Crefs = p.Crefs[0 : i+1];
				p.Crefs[i] = &Cref{
					Name: sel.Sel.Value,
					Expr: n,
					Context: context,
				};
				break;
			}
		}
		walk(*n, p, context);

	// everything else just recurs
	default:
		error(noPos, "unexpected type %T in walk", x);
		panic();

	case nil:

	// These are ordered and grouped to match ../../pkg/go/ast/ast.go
	case *ast.Field:
		walk(&n.Type, p, "type")
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.Ellipsis:
	case *ast.BasicLit:
	case *ast.StringList:
	case *ast.FuncLit:
		walk(n.Type, p, "type");
		walk(n.Body, p, "stmt");
	case *ast.CompositeLit:
		walk(&n.Type, p, "type");
		walk(n.Elts, p, "expr");
	case *ast.ParenExpr:
		walk(&n.X, p, context)
	case *ast.SelectorExpr:
		walk(&n.X, p, "selector")
	case *ast.IndexExpr:
		walk(&n.X, p, "expr");
		walk(&n.Index, p, "expr");
		if n.End != nil {
			walk(&n.End, p, "expr")
		}
	case *ast.TypeAssertExpr:
		walk(&n.X, p, "expr");
		walk(&n.Type, p, "type");
	case *ast.CallExpr:
		walk(&n.Fun, p, "call");
		walk(n.Args, p, "expr");
	case *ast.StarExpr:
		walk(&n.X, p, context)
	case *ast.UnaryExpr:
		walk(&n.X, p, "expr")
	case *ast.BinaryExpr:
		walk(&n.X, p, "expr");
		walk(&n.Y, p, "expr");
	case *ast.KeyValueExpr:
		walk(&n.Key, p, "expr");
		walk(&n.Value, p, "expr");

	case *ast.ArrayType:
		walk(&n.Len, p, "expr");
		walk(&n.Elt, p, "type");
	case *ast.StructType:
		walk(n.Fields, p, "field")
	case *ast.FuncType:
		walk(n.Params, p, "field");
		walk(n.Results, p, "field");
	case *ast.InterfaceType:
		walk(n.Methods, p, "field")
	case *ast.MapType:
		walk(&n.Key, p, "type");
		walk(&n.Value, p, "type");
	case *ast.ChanType:
		walk(&n.Value, p, "type")

	case *ast.BadStmt:
	case *ast.DeclStmt:
		walk(n.Decl, p, "decl")
	case *ast.EmptyStmt:
	case *ast.LabeledStmt:
		walk(n.Stmt, p, "stmt")
	case *ast.ExprStmt:
		walk(&n.X, p, "expr")
	case *ast.IncDecStmt:
		walk(&n.X, p, "expr")
	case *ast.AssignStmt:
		walk(n.Lhs, p, "expr");
		walk(n.Rhs, p, "expr");
	case *ast.GoStmt:
		walk(n.Call, p, "expr")
	case *ast.DeferStmt:
		walk(n.Call, p, "expr")
	case *ast.ReturnStmt:
		walk(n.Results, p, "expr")
	case *ast.BranchStmt:
	case *ast.BlockStmt:
		walk(n.List, p, "stmt")
	case *ast.IfStmt:
		walk(n.Init, p, "stmt");
		walk(&n.Cond, p, "expr");
		walk(n.Body, p, "stmt");
		walk(n.Else, p, "stmt");
	case *ast.CaseClause:
		walk(n.Values, p, "expr");
		walk(n.Body, p, "stmt");
	case *ast.SwitchStmt:
		walk(n.Init, p, "stmt");
		walk(&n.Tag, p, "expr");
		walk(n.Body, p, "stmt");
	case *ast.TypeCaseClause:
		walk(n.Types, p, "type");
		walk(n.Body, p, "stmt");
	case *ast.TypeSwitchStmt:
		walk(n.Init, p, "stmt");
		walk(n.Assign, p, "stmt");
		walk(n.Body, p, "stmt");
	case *ast.CommClause:
		walk(n.Lhs, p, "expr");
		walk(n.Rhs, p, "expr");
		walk(n.Body, p, "stmt");
	case *ast.SelectStmt:
		walk(n.Body, p, "stmt")
	case *ast.ForStmt:
		walk(n.Init, p, "stmt");
		walk(&n.Cond, p, "expr");
		walk(n.Post, p, "stmt");
		walk(n.Body, p, "stmt");
	case *ast.RangeStmt:
		walk(&n.Key, p, "expr");
		walk(&n.Value, p, "expr");
		walk(&n.X, p, "expr");
		walk(n.Body, p, "stmt");

	case *ast.ImportSpec:
	case *ast.ValueSpec:
		walk(&n.Type, p, "type");
		walk(n.Values, p, "expr");
	case *ast.TypeSpec:
		walk(&n.Type, p, "type")

	case *ast.BadDecl:
	case *ast.GenDecl:
		walk(n.Specs, p, "spec")
	case *ast.FuncDecl:
		if n.Recv != nil {
			walk(n.Recv, p, "field")
		}
		walk(n.Type, p, "type");
		if n.Body != nil {
			walk(n.Body, p, "stmt")
		}

	case *ast.File:
		walk(n.Decls, p, "decl")

	case *ast.Package:
		for _, f := range n.Files {
			walk(f, p, "file")
		}

	case []ast.Decl:
		for _, d := range n {
			walk(d, p, context)
		}
	case []ast.Expr:
		for i := range n {
			walk(&n[i], p, context)
		}
	case []*ast.Field:
		for _, f := range n {
			walk(f, p, context)
		}
	case []ast.Stmt:
		for _, s := range n {
			walk(s, p, context)
		}
	case []ast.Spec:
		for _, s := range n {
			walk(s, p, context)
		}
	}
}
