// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse input AST and prepare Prog structure.

package main

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/scanner"
	"go/token"
	"os"
	"strings"
)

func parse(name string, src []byte, flags parser.Mode) *ast.File {
	ast1, err := parser.ParseFile(fset, name, src, flags)
	if err != nil {
		if list, ok := err.(scanner.ErrorList); ok {
			// If err is a scanner.ErrorList, its String will print just
			// the first error and then (+n more errors).
			// Instead, turn it into a new Error that will return
			// details for all the errors.
			for _, e := range list {
				fmt.Fprintln(os.Stderr, e)
			}
			os.Exit(2)
		}
		fatalf("parsing %s: %s", name, err)
	}
	return ast1
}

func sourceLine(n ast.Node) int {
	return fset.Position(n.Pos()).Line
}

// ParseGo populates f with information learned from the Go source code
// which was read from the named file. It gathers the C preamble
// attached to the import "C" comment, a list of references to C.xxx,
// a list of exported functions, and the actual AST, to be rewritten and
// printed.
func (f *File) ParseGo(abspath string, src []byte) {
	// Two different parses: once with comments, once without.
	// The printer is not good enough at printing comments in the
	// right place when we start editing the AST behind its back,
	// so we use ast1 to look for the doc comments on import "C"
	// and on exported functions, and we use ast2 for translating
	// and reprinting.
	// In cgo mode, we ignore ast2 and just apply edits directly
	// the text behind ast1. In godefs mode we modify and print ast2.
	ast1 := parse(abspath, src, parser.SkipObjectResolution|parser.ParseComments)
	ast2 := parse(abspath, src, parser.SkipObjectResolution)

	f.Package = ast1.Name.Name
	f.Name = make(map[string]*Name)
	f.NamePos = make(map[*Name]token.Pos)

	// In ast1, find the import "C" line and get any extra C preamble.
	sawC := false
	for _, decl := range ast1.Decls {
		switch decl := decl.(type) {
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				s, ok := spec.(*ast.ImportSpec)
				if !ok || s.Path.Value != `"C"` {
					continue
				}
				sawC = true
				if s.Name != nil {
					error_(s.Path.Pos(), `cannot rename import "C"`)
				}
				cg := s.Doc
				if cg == nil && len(decl.Specs) == 1 {
					cg = decl.Doc
				}
				if cg != nil {
					if strings.ContainsAny(abspath, "\r\n") {
						// This should have been checked when the file path was first resolved,
						// but we double check here just to be sure.
						fatalf("internal error: ParseGo: abspath contains unexpected newline character: %q", abspath)
					}
					f.Preamble += fmt.Sprintf("#line %d %q\n", sourceLine(cg), abspath)
					f.Preamble += commentText(cg) + "\n"
					f.Preamble += "#line 1 \"cgo-generated-wrapper\"\n"
				}
			}

		case *ast.FuncDecl:
			// Also, reject attempts to declare methods on C.T or *C.T.
			// (The generated code would otherwise accept this
			// invalid input; see issue #57926.)
			if decl.Recv != nil && len(decl.Recv.List) > 0 {
				recvType := decl.Recv.List[0].Type
				if recvType != nil {
					t := recvType
					if star, ok := unparen(t).(*ast.StarExpr); ok {
						t = star.X
					}
					if sel, ok := unparen(t).(*ast.SelectorExpr); ok {
						var buf strings.Builder
						format.Node(&buf, fset, recvType)
						error_(sel.Pos(), `cannot define new methods on non-local type %s`, &buf)
					}
				}
			}
		}

	}
	if !sawC {
		error_(ast1.Package, `cannot find import "C"`)
	}

	// In ast2, strip the import "C" line.
	if *godefs {
		w := 0
		for _, decl := range ast2.Decls {
			d, ok := decl.(*ast.GenDecl)
			if !ok {
				ast2.Decls[w] = decl
				w++
				continue
			}
			ws := 0
			for _, spec := range d.Specs {
				s, ok := spec.(*ast.ImportSpec)
				if !ok || s.Path.Value != `"C"` {
					d.Specs[ws] = spec
					ws++
				}
			}
			if ws == 0 {
				continue
			}
			d.Specs = d.Specs[0:ws]
			ast2.Decls[w] = d
			w++
		}
		ast2.Decls = ast2.Decls[0:w]
	} else {
		for _, decl := range ast2.Decls {
			d, ok := decl.(*ast.GenDecl)
			if !ok {
				continue
			}
			for _, spec := range d.Specs {
				if s, ok := spec.(*ast.ImportSpec); ok && s.Path.Value == `"C"` {
					// Replace "C" with _ "unsafe", to keep program valid.
					// (Deleting import statement or clause is not safe if it is followed
					// in the source by an explicit semicolon.)
					f.Edit.Replace(f.offset(s.Path.Pos()), f.offset(s.Path.End()), `_ "unsafe"`)
				}
			}
		}
	}

	// Accumulate pointers to uses of C.x.
	if f.Ref == nil {
		f.Ref = make([]*Ref, 0, 8)
	}
	f.walk(ast2, ctxProg, (*File).validateIdents)
	f.walk(ast2, ctxProg, (*File).saveExprs)

	// Accumulate exported functions.
	// The comments are only on ast1 but we need to
	// save the function bodies from ast2.
	// The first walk fills in ExpFunc, and the
	// second walk changes the entries to
	// refer to ast2 instead.
	f.walk(ast1, ctxProg, (*File).saveExport)
	f.walk(ast2, ctxProg, (*File).saveExport2)

	f.Comments = ast1.Comments
	f.AST = ast2
}

// Like ast.CommentGroup's Text method but preserves
// leading blank lines, so that line numbers line up.
func commentText(g *ast.CommentGroup) string {
	pieces := make([]string, 0, len(g.List))
	for _, com := range g.List {
		c := com.Text
		// Remove comment markers.
		// The parser has given us exactly the comment text.
		switch c[1] {
		case '/':
			//-style comment (no newline at the end)
			c = c[2:] + "\n"
		case '*':
			/*-style comment */
			c = c[2 : len(c)-2]
		}
		pieces = append(pieces, c)
	}
	return strings.Join(pieces, "")
}

func (f *File) validateIdents(x any, context astContext) {
	if x, ok := x.(*ast.Ident); ok {
		if f.isMangledName(x.Name) {
			error_(x.Pos(), "identifier %q may conflict with identifiers generated by cgo", x.Name)
		}
	}
}

// Save various references we are going to need later.
func (f *File) saveExprs(x any, context astContext) {
	switch x := x.(type) {
	case *ast.Expr:
		switch (*x).(type) {
		case *ast.SelectorExpr:
			f.saveRef(x, context)
		}
	case *ast.CallExpr:
		f.saveCall(x, context)
	}
}

// Save references to C.xxx for later processing.
func (f *File) saveRef(n *ast.Expr, context astContext) {
	sel := (*n).(*ast.SelectorExpr)
	// For now, assume that the only instance of capital C is when
	// used as the imported package identifier.
	// The parser should take care of scoping in the future, so
	// that we will be able to distinguish a "top-level C" from a
	// local C.
	if l, ok := sel.X.(*ast.Ident); !ok || l.Name != "C" {
		return
	}
	if context == ctxAssign2 {
		context = ctxExpr
	}
	if context == ctxEmbedType {
		error_(sel.Pos(), "cannot embed C type")
	}
	goname := sel.Sel.Name
	if goname == "errno" {
		error_(sel.Pos(), "cannot refer to errno directly; see documentation")
		return
	}
	if goname == "_CMalloc" {
		error_(sel.Pos(), "cannot refer to C._CMalloc; use C.malloc")
		return
	}
	if goname == "malloc" {
		goname = "_CMalloc"
	}
	name := f.Name[goname]
	if name == nil {
		name = &Name{
			Go: goname,
		}
		f.Name[goname] = name
		f.NamePos[name] = sel.Pos()
	}
	f.Ref = append(f.Ref, &Ref{
		Name:    name,
		Expr:    n,
		Context: context,
	})
}

// Save calls to C.xxx for later processing.
func (f *File) saveCall(call *ast.CallExpr, context astContext) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	if l, ok := sel.X.(*ast.Ident); !ok || l.Name != "C" {
		return
	}
	c := &Call{Call: call, Deferred: context == ctxDefer}
	f.Calls = append(f.Calls, c)
}

// If a function should be exported add it to ExpFunc.
func (f *File) saveExport(x any, context astContext) {
	n, ok := x.(*ast.FuncDecl)
	if !ok {
		return
	}

	if n.Doc == nil {
		return
	}
	for _, c := range n.Doc.List {
		if !strings.HasPrefix(c.Text, "//export ") {
			continue
		}

		name := strings.TrimSpace(c.Text[9:])
		if name == "" {
			error_(c.Pos(), "export missing name")
		}

		if name != n.Name.Name {
			error_(c.Pos(), "export comment has wrong name %q, want %q", name, n.Name.Name)
		}

		doc := ""
		for _, c1 := range n.Doc.List {
			if c1 != c {
				doc += c1.Text + "\n"
			}
		}

		f.ExpFunc = append(f.ExpFunc, &ExpFunc{
			Func:    n,
			ExpName: name,
			Doc:     doc,
		})
		break
	}
}

// Make f.ExpFunc[i] point at the Func from this AST instead of the other one.
func (f *File) saveExport2(x any, context astContext) {
	n, ok := x.(*ast.FuncDecl)
	if !ok {
		return
	}

	for _, exp := range f.ExpFunc {
		if exp.Func.Name.Name == n.Name.Name {
			exp.Func = n
			break
		}
	}
}

type astContext int

const (
	ctxProg astContext = iota
	ctxEmbedType
	ctxType
	ctxStmt
	ctxExpr
	ctxField
	ctxParam
	ctxAssign2 // assignment of a single expression to two variables
	ctxSwitch
	ctxTypeSwitch
	ctxFile
	ctxDecl
	ctxSpec
	ctxDefer
	ctxCall  // any function call other than ctxCall2
	ctxCall2 // function call whose result is assigned to two variables
	ctxSelector
)

// walk walks the AST x, calling visit(f, x, context) for each node.
func (f *File) walk(x any, context astContext, visit func(*File, any, astContext)) {
	visit(f, x, context)
	switch n := x.(type) {
	case *ast.Expr:
		f.walk(*n, context, visit)

	// everything else just recurs
	default:
		error_(token.NoPos, "unexpected type %T in walk", x)
		panic("unexpected type")

	case nil:

	// These are ordered and grouped to match ../../go/ast/ast.go
	case *ast.Field:
		if len(n.Names) == 0 && context == ctxField {
			f.walk(&n.Type, ctxEmbedType, visit)
		} else {
			f.walk(&n.Type, ctxType, visit)
		}
	case *ast.FieldList:
		for _, field := range n.List {
			f.walk(field, context, visit)
		}
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.Ellipsis:
		f.walk(&n.Elt, ctxType, visit)
	case *ast.BasicLit:
	case *ast.FuncLit:
		f.walk(n.Type, ctxType, visit)
		f.walk(n.Body, ctxStmt, visit)
	case *ast.CompositeLit:
		f.walk(&n.Type, ctxType, visit)
		f.walk(n.Elts, ctxExpr, visit)
	case *ast.ParenExpr:
		f.walk(&n.X, context, visit)
	case *ast.SelectorExpr:
		f.walk(&n.X, ctxSelector, visit)
	case *ast.IndexExpr:
		f.walk(&n.X, ctxExpr, visit)
		f.walk(&n.Index, ctxExpr, visit)
	case *ast.IndexListExpr:
		f.walk(&n.X, ctxExpr, visit)
		f.walk(n.Indices, ctxExpr, visit)
	case *ast.SliceExpr:
		f.walk(&n.X, ctxExpr, visit)
		if n.Low != nil {
			f.walk(&n.Low, ctxExpr, visit)
		}
		if n.High != nil {
			f.walk(&n.High, ctxExpr, visit)
		}
		if n.Max != nil {
			f.walk(&n.Max, ctxExpr, visit)
		}
	case *ast.TypeAssertExpr:
		f.walk(&n.X, ctxExpr, visit)
		f.walk(&n.Type, ctxType, visit)
	case *ast.CallExpr:
		if context == ctxAssign2 {
			f.walk(&n.Fun, ctxCall2, visit)
		} else {
			f.walk(&n.Fun, ctxCall, visit)
		}
		f.walk(n.Args, ctxExpr, visit)
	case *ast.StarExpr:
		f.walk(&n.X, context, visit)
	case *ast.UnaryExpr:
		f.walk(&n.X, ctxExpr, visit)
	case *ast.BinaryExpr:
		f.walk(&n.X, ctxExpr, visit)
		f.walk(&n.Y, ctxExpr, visit)
	case *ast.KeyValueExpr:
		f.walk(&n.Key, ctxExpr, visit)
		f.walk(&n.Value, ctxExpr, visit)

	case *ast.ArrayType:
		f.walk(&n.Len, ctxExpr, visit)
		f.walk(&n.Elt, ctxType, visit)
	case *ast.StructType:
		f.walk(n.Fields, ctxField, visit)
	case *ast.FuncType:
		if n.TypeParams != nil {
			f.walk(n.TypeParams, ctxParam, visit)
		}
		f.walk(n.Params, ctxParam, visit)
		if n.Results != nil {
			f.walk(n.Results, ctxParam, visit)
		}
	case *ast.InterfaceType:
		f.walk(n.Methods, ctxField, visit)
	case *ast.MapType:
		f.walk(&n.Key, ctxType, visit)
		f.walk(&n.Value, ctxType, visit)
	case *ast.ChanType:
		f.walk(&n.Value, ctxType, visit)

	case *ast.BadStmt:
	case *ast.DeclStmt:
		f.walk(n.Decl, ctxDecl, visit)
	case *ast.EmptyStmt:
	case *ast.LabeledStmt:
		f.walk(n.Stmt, ctxStmt, visit)
	case *ast.ExprStmt:
		f.walk(&n.X, ctxExpr, visit)
	case *ast.SendStmt:
		f.walk(&n.Chan, ctxExpr, visit)
		f.walk(&n.Value, ctxExpr, visit)
	case *ast.IncDecStmt:
		f.walk(&n.X, ctxExpr, visit)
	case *ast.AssignStmt:
		f.walk(n.Lhs, ctxExpr, visit)
		if len(n.Lhs) == 2 && len(n.Rhs) == 1 {
			f.walk(n.Rhs, ctxAssign2, visit)
		} else {
			f.walk(n.Rhs, ctxExpr, visit)
		}
	case *ast.GoStmt:
		f.walk(n.Call, ctxExpr, visit)
	case *ast.DeferStmt:
		f.walk(n.Call, ctxDefer, visit)
	case *ast.ReturnStmt:
		f.walk(n.Results, ctxExpr, visit)
	case *ast.BranchStmt:
	case *ast.BlockStmt:
		f.walk(n.List, context, visit)
	case *ast.IfStmt:
		f.walk(n.Init, ctxStmt, visit)
		f.walk(&n.Cond, ctxExpr, visit)
		f.walk(n.Body, ctxStmt, visit)
		f.walk(n.Else, ctxStmt, visit)
	case *ast.CaseClause:
		if context == ctxTypeSwitch {
			context = ctxType
		} else {
			context = ctxExpr
		}
		f.walk(n.List, context, visit)
		f.walk(n.Body, ctxStmt, visit)
	case *ast.SwitchStmt:
		f.walk(n.Init, ctxStmt, visit)
		f.walk(&n.Tag, ctxExpr, visit)
		f.walk(n.Body, ctxSwitch, visit)
	case *ast.TypeSwitchStmt:
		f.walk(n.Init, ctxStmt, visit)
		f.walk(n.Assign, ctxStmt, visit)
		f.walk(n.Body, ctxTypeSwitch, visit)
	case *ast.CommClause:
		f.walk(n.Comm, ctxStmt, visit)
		f.walk(n.Body, ctxStmt, visit)
	case *ast.SelectStmt:
		f.walk(n.Body, ctxStmt, visit)
	case *ast.ForStmt:
		f.walk(n.Init, ctxStmt, visit)
		f.walk(&n.Cond, ctxExpr, visit)
		f.walk(n.Post, ctxStmt, visit)
		f.walk(n.Body, ctxStmt, visit)
	case *ast.RangeStmt:
		f.walk(&n.Key, ctxExpr, visit)
		f.walk(&n.Value, ctxExpr, visit)
		f.walk(&n.X, ctxExpr, visit)
		f.walk(n.Body, ctxStmt, visit)

	case *ast.ImportSpec:
	case *ast.ValueSpec:
		f.walk(&n.Type, ctxType, visit)
		if len(n.Names) == 2 && len(n.Values) == 1 {
			f.walk(&n.Values[0], ctxAssign2, visit)
		} else {
			f.walk(n.Values, ctxExpr, visit)
		}
	case *ast.TypeSpec:
		if n.TypeParams != nil {
			f.walk(n.TypeParams, ctxParam, visit)
		}
		f.walk(&n.Type, ctxType, visit)

	case *ast.BadDecl:
	case *ast.GenDecl:
		f.walk(n.Specs, ctxSpec, visit)
	case *ast.FuncDecl:
		if n.Recv != nil {
			f.walk(n.Recv, ctxParam, visit)
		}
		f.walk(n.Type, ctxType, visit)
		if n.Body != nil {
			f.walk(n.Body, ctxStmt, visit)
		}

	case *ast.File:
		f.walk(n.Decls, ctxDecl, visit)

	case *ast.Package:
		for _, file := range n.Files {
			f.walk(file, ctxFile, visit)
		}

	case []ast.Decl:
		for _, d := range n {
			f.walk(d, context, visit)
		}
	case []ast.Expr:
		for i := range n {
			f.walk(&n[i], context, visit)
		}
	case []ast.Stmt:
		for _, s := range n {
			f.walk(s, context, visit)
		}
	case []ast.Spec:
		for _, s := range n {
			f.walk(s, context, visit)
		}
	}
}

// If x is of the form (T), unparen returns unparen(T), otherwise it returns x.
func unparen(x ast.Expr) ast.Expr {
	if p, isParen := x.(*ast.ParenExpr); isParen {
		x = unparen(p.X)
	}
	return x
}
