// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cgo; see gmp.go for an overview.

// TODO(rsc):
//	Emit correct line number annotations.
//	Make 6g understand the annotations.
package main

import (
	"bufio";
	"container/vector";
	"debug/dwarf";
	"debug/elf";
	"flag";
	"fmt";
	"go/ast";
	"go/doc";
	"go/parser";
	"go/scanner";
	"go/token";
	"io";
	"os";
)

// Map of uses of C.xxx.  The key is the pointer
// to the use (a pointer so it can be rewritten)
// and the value is the context ("call", "expr", "type").
type cmap map[*ast.Expr] string

var noPos token.Position

func usage() {
	fmt.Fprint(os.Stderr, "usage: cgo [options] file.cgo\n");
	flag.PrintDefaults();
}

func main() {
	flag.Usage = usage;
	flag.Parse();

	args := flag.Args();
	if len(args) != 1 {
		flag.Usage();
	}
	filename := args[0];

	prog, err := parser.ParsePkgFile("", filename, parser.ParseComments);
	if err != nil {
		fatal(err);
	}

	// Find the import "C" line and get any extra C preamble.
	preamble := "";
	found := false;
	for _, d := range prog.Decls {
		d, ok := d.(*ast.GenDecl);
		if !ok {
			continue;
		}
		for _, s := range d.Specs {
			s, ok := s.(*ast.ImportSpec);
			if !ok {
				continue;
			}
			if len(s.Path) != 1 || string(s.Path[0].Value) != `"C"` {
				continue;
			}
			found = true;
			if s.Name != nil {
				error(s.Path[0].Pos(), `cannot rename import "C"`);
			}
			if s.Doc != nil {
				preamble += doc.CommentText(s.Doc) + "\n";
			}
			else if len(d.Specs) == 1 && d.Doc != nil {
				preamble += doc.CommentText(d.Doc) + "\n";
			}
		}
	}
	if !found {
		error(noPos, `cannot find import "C"`);
	}

	// Accumulate pointers to uses of C.x.
	m := make(cmap);
	walk(prog, m, "prog");

	fmt.Print(preamble);
	for p, context := range m {
		sel := (*p).(*ast.SelectorExpr);
		fmt.Printf("%s: %s as %s\n", sel.Pos(), sel.Sel.Value, context);
	}
}

func walk(x interface{}, m cmap, context string) {
	switch n := x.(type) {
	case *ast.Expr:
		if sel, ok := (*n).(*ast.SelectorExpr); ok {
			// For now, assume that the only instance of capital C is
			// when used as the imported package identifier.
			// The parser should take care of scoping in the future,
			// so that we will be able to distinguish a "top-level C"
			// from a local C.
			if l, ok := sel.X.(*ast.Ident); ok && l.Value == "C" {
				m[n] = context;
				break;
			}
		}
		walk(*n, m, context);

	// everything else just recurs
	default:
		error(noPos, "unexpected type %T in walk", x);
		panic();

	case nil:

	// These are ordered and grouped to match ../../pkg/go/ast/ast.go
	case *ast.Field:
		walk(&n.Type, m, "type");
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.Ellipsis:
	case *ast.BasicLit:
	case *ast.StringList:
	case *ast.FuncLit:
		walk(n.Type, m, "type");
		walk(n.Body, m, "stmt");
	case *ast.CompositeLit:
		walk(&n.Type, m, "type");
		walk(n.Elts, m, "expr");
	case *ast.ParenExpr:
		walk(&n.X, m, context);
	case *ast.SelectorExpr:
		walk(&n.X, m, "selector");
	case *ast.IndexExpr:
		walk(&n.X, m, "expr");
		walk(&n.Index, m, "expr");
		if n.End != nil {
			walk(&n.End, m, "expr");
		}
	case *ast.TypeAssertExpr:
		walk(&n.X, m, "expr");
		walk(&n.Type, m, "type");
	case *ast.CallExpr:
		walk(&n.Fun, m, "call");
		walk(n.Args, m, "expr");
	case *ast.StarExpr:
		walk(&n.X, m, context);
	case *ast.UnaryExpr:
		walk(&n.X, m, "expr");
	case *ast.BinaryExpr:
		walk(&n.X, m, "expr");
		walk(&n.Y, m, "expr");
	case *ast.KeyValueExpr:
		walk(&n.Key, m, "expr");
		walk(&n.Value, m, "expr");

	case *ast.ArrayType:
		walk(&n.Len, m, "expr");
		walk(&n.Elt, m, "type");
	case *ast.StructType:
		walk(n.Fields, m, "field");
	case *ast.FuncType:
		walk(n.Params, m, "field");
		walk(n.Results, m, "field");
	case *ast.InterfaceType:
		walk(n.Methods, m, "field");
	case *ast.MapType:
		walk(&n.Key, m, "type");
		walk(&n.Value, m, "type");
	case *ast.ChanType:
		walk(&n.Value, m, "type");

	case *ast.BadStmt:
	case *ast.DeclStmt:
		walk(n.Decl, m, "decl");
	case *ast.EmptyStmt:
	case *ast.LabeledStmt:
		walk(n.Stmt, m, "stmt");
	case *ast.ExprStmt:
		walk(&n.X, m, "expr");
	case *ast.IncDecStmt:
		walk(&n.X, m, "expr");
	case *ast.AssignStmt:
		walk(n.Lhs, m, "expr");
		walk(n.Rhs, m, "expr");
	case *ast.GoStmt:
		walk(&n.Call, m, "expr");
	case *ast.DeferStmt:
		walk(&n.Call, m, "expr");
	case *ast.ReturnStmt:
		walk(n.Results, m, "expr");
	case *ast.BranchStmt:
	case *ast.BlockStmt:
		walk(n.List, m, "stmt");
	case *ast.IfStmt:
		walk(n.Init, m, "stmt");
		walk(&n.Cond, m, "expr");
		walk(n.Body, m, "stmt");
		walk(n.Else, m, "stmt");
	case *ast.CaseClause:
		walk(n.Values, m, "expr");
		walk(n.Body, m, "stmt");
	case *ast.SwitchStmt:
		walk(n.Init, m, "stmt");
		walk(&n.Tag, m, "expr");
		walk(n.Body, m, "stmt");
	case *ast.TypeCaseClause:
		walk(n.Types, m, "type");
		walk(n.Body, m, "stmt");
	case *ast.TypeSwitchStmt:
		walk(n.Init, m, "stmt");
		walk(n.Assign, m, "stmt");
		walk(n.Body, m, "stmt");
	case *ast.CommClause:
		walk(n.Lhs, m, "expr");
		walk(n.Rhs, m, "expr");
		walk(n.Body, m, "stmt");
	case *ast.SelectStmt:
		walk(n.Body, m, "stmt");
	case *ast.ForStmt:
		walk(n.Init, m, "stmt");
		walk(&n.Cond, m, "expr");
		walk(n.Post, m, "stmt");
		walk(n.Body, m, "stmt");
	case *ast.RangeStmt:
		walk(&n.Key, m, "expr");
		walk(&n.Value, m, "expr");
		walk(&n.X, m, "expr");
		walk(n.Body, m, "stmt");

	case *ast.ImportSpec:
	case *ast.ValueSpec:
		walk(&n.Type, m, "type");
		walk(n.Values, m, "expr");
	case *ast.TypeSpec:
		walk(&n.Type, m, "type");

	case *ast.BadDecl:
	case *ast.GenDecl:
		walk(n.Specs, m, "spec");
	case *ast.FuncDecl:
		if n.Recv != nil {
			walk(n.Recv, m, "field");
		}
		walk(n.Type, m, "type");
		walk(n.Body, m, "stmt");

	case *ast.File:
		walk(n.Decls, m, "decl");

	case *ast.Package:
		for _, f := range n.Files {
			walk(f, m, "file");
		}

	case []ast.Decl:
		for _, d := range n {
			walk(d, m, context);
		}
	case []ast.Expr:
		for i := range n {
			walk(&n[i], m, context);
		}
	case []*ast.Field:
		for _, f := range n {
			walk(f, m, context);
		}
	case []ast.Stmt:
		for _, s := range n {
			walk(s, m, context);
		}
	case []ast.Spec:
		for _, s := range n {
			walk(s, m, context);
		}
	}
}

func fatal(err os.Error) {
	// If err is a scanner.ErrorList, its String will print just
	// the first error and then (+n more errors).
	// Instead, turn it into a new Error that will return
	// details for all the errors.
	if list, ok := err.(scanner.ErrorList); ok {
		for _, e := range list {
			fmt.Fprintln(os.Stderr, e);
		}
	} else {
		fmt.Fprintln(os.Stderr, err);
	}
	os.Exit(2);
}

var nerrors int

func error(pos token.Position, msg string, args ...) {
	nerrors++;
	if pos.IsValid() {
		fmt.Fprintf(os.Stderr, "%s: ", pos);
	}
	fmt.Fprintf(os.Stderr, msg, args);
	fmt.Fprintf(os.Stderr, "\n");
}
