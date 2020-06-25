// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fillstruct defines an Analyzer that automatically
// fills in a struct declaration with zero value elements for each field.
package fillstruct

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/printer"
	"go/token"
	"go/types"
	"log"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
)

const Doc = `suggested input for incomplete struct initializations

This analyzer provides the appropriate zero values for all
uninitialized fields of an empty struct. For example, given the following struct:
	type Foo struct {
		ID   int64
		Name string
	}
the initialization
	var _ = Foo{}
will turn into
	var _ = Foo{
		ID: 0,
		Name: "",
	}
`

var Analyzer = &analysis.Analyzer{
	Name:             "fillstruct",
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	Run:              run,
	RunDespiteErrors: true,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{(*ast.CompositeLit)(nil)}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		info := pass.TypesInfo
		if info == nil {
			return
		}
		expr := n.(*ast.CompositeLit)

		// TODO: Handle partially-filled structs as well.
		if len(expr.Elts) != 0 {
			return
		}

		var file *ast.File
		for _, f := range pass.Files {
			if f.Pos() <= expr.Pos() && expr.Pos() <= f.End() {
				file = f
				break
			}
		}
		if file == nil {
			return
		}

		typ := info.TypeOf(expr)
		if typ == nil {
			return
		}

		// Find reference to the type declaration of the struct being initialized.
		for {
			p, ok := typ.Underlying().(*types.Pointer)
			if !ok {
				break
			}
			typ = p.Elem()
		}
		typ = typ.Underlying()

		obj, ok := typ.(*types.Struct)
		if !ok {
			return
		}
		fieldCount := obj.NumFields()
		// Skip any struct that is already populated or that has no fields.
		if fieldCount == 0 || fieldCount == len(expr.Elts) {
			return
		}

		var name string
		switch typ := expr.Type.(type) {
		case *ast.Ident:
			name = typ.Name
		case *ast.SelectorExpr:
			name = fmt.Sprintf("%s.%s", typ.X, typ.Sel.Name)
		default:
			log.Printf("anonymous structs are not yet supported: (%T)", expr.Type)
			return
		}

		// Use a new fileset to build up a token.File for the new composite
		// literal. We need one line for foo{, one line for }, and one line for
		// each field we're going to set. format.Node only cares about line
		// numbers, so we don't need to set columns, and each line can be
		// 1 byte long.
		fset := token.NewFileSet()
		tok := fset.AddFile("", -1, fieldCount+2)

		line := 2 // account for 1-based lines and the left brace
		var elts []ast.Expr
		for i := 0; i < fieldCount; i++ {
			field := obj.Field(i)

			// Ignore fields that are not accessible in the current package.
			if field.Pkg() != nil && field.Pkg() != pass.Pkg && !field.Exported() {
				continue
			}

			value := populateValue(pass.Fset, file, pass.Pkg, field.Type())
			if value == nil {
				continue
			}

			tok.AddLine(line - 1) // add 1 byte per line
			if line > tok.LineCount() {
				panic(fmt.Sprintf("invalid line number %v (of %v) for fillstruct %s", line, tok.LineCount(), name))
			}
			pos := tok.LineStart(line)

			kv := &ast.KeyValueExpr{
				Key: &ast.Ident{
					NamePos: pos,
					Name:    field.Name(),
				},
				Colon: pos,
				Value: value,
			}
			elts = append(elts, kv)
			line++
		}

		// If all of the struct's fields are unexported, we have nothing to do.
		if len(elts) == 0 {
			return
		}

		// Add the final line for the right brace. Offset is the number of
		// bytes already added plus 1.
		tok.AddLine(len(elts) + 1)
		line = len(elts) + 2
		if line > tok.LineCount() {
			panic(fmt.Sprintf("invalid line number %v (of %v) for fillstruct %s", line, tok.LineCount(), name))
		}

		cl := &ast.CompositeLit{
			Type:   expr.Type,
			Lbrace: tok.LineStart(1),
			Elts:   elts,
			Rbrace: tok.LineStart(line),
		}

		// Print the AST to get the the original source code.
		var b bytes.Buffer
		if err := printer.Fprint(&b, pass.Fset, file); err != nil {
			log.Printf("failed to print original file: %s", err)
			return
		}

		// Find the line on which the composite literal is declared.
		split := strings.Split(b.String(), "\n")
		lineNumber := pass.Fset.Position(expr.Type.Pos()).Line
		firstLine := split[lineNumber-1] // lines are 1-indexed

		// Trim the whitespace from the left of the line, and use the index
		// to get the amount of whitespace on the left.
		trimmed := strings.TrimLeftFunc(firstLine, unicode.IsSpace)
		index := strings.Index(firstLine, trimmed)
		whitespace := firstLine[:index]

		var newExpr bytes.Buffer
		if err := format.Node(&newExpr, fset, cl); err != nil {
			log.Printf("failed to format %s: %v", cl.Type, err)
			return
		}
		split = strings.Split(newExpr.String(), "\n")
		var newText strings.Builder
		for i, s := range split {
			// Don't add the extra indentation to the first line.
			if i != 0 {
				newText.WriteString(whitespace)
			}
			newText.WriteString(s)
			if i < len(split)-1 {
				newText.WriteByte('\n')
			}
		}
		pass.Report(analysis.Diagnostic{
			Pos: expr.Lbrace,
			End: expr.Rbrace,
			SuggestedFixes: []analysis.SuggestedFix{{
				Message: fmt.Sprintf("Fill %s with default values", name),
				TextEdits: []analysis.TextEdit{{
					Pos:     expr.Pos(),
					End:     expr.End(),
					NewText: []byte(newText.String()),
				}},
			}},
		})
	})
	return nil, nil
}

// populateValue constructs an expression to fill the value of a struct field.
//
// When the type of a struct field is a basic literal or interface, we return
// default values. For other types, such as maps, slices, and channels, we create
// expressions rather than using default values.
//
// The reasoning here is that users will call fillstruct with the intention of
// initializing the struct, in which case setting these fields to nil has no effect.
func populateValue(fset *token.FileSet, f *ast.File, pkg *types.Package, typ types.Type) ast.Expr {
	under := typ
	if n, ok := typ.(*types.Named); ok {
		under = n.Underlying()
	}
	switch u := under.(type) {
	case *types.Basic:
		switch {
		case u.Info()&types.IsNumeric != 0:
			return &ast.BasicLit{Kind: token.INT, Value: "0"}
		case u.Info()&types.IsBoolean != 0:
			return &ast.Ident{Name: "false"}
		case u.Info()&types.IsString != 0:
			return &ast.BasicLit{Kind: token.STRING, Value: `""`}
		default:
			panic("unknown basic type")
		}
	case *types.Map:
		k := analysisinternal.TypeExpr(fset, f, pkg, u.Key())
		v := analysisinternal.TypeExpr(fset, f, pkg, u.Elem())
		if k == nil || v == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: &ast.MapType{
				Key:   k,
				Value: v,
			},
		}
	case *types.Slice:
		s := analysisinternal.TypeExpr(fset, f, pkg, u.Elem())
		if s == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: &ast.ArrayType{
				Elt: s,
			},
		}
	case *types.Array:
		a := analysisinternal.TypeExpr(fset, f, pkg, u.Elem())
		if a == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: &ast.ArrayType{
				Elt: a,
				Len: &ast.BasicLit{
					Kind: token.INT, Value: fmt.Sprintf("%v", u.Len())},
			},
		}
	case *types.Chan:
		v := analysisinternal.TypeExpr(fset, f, pkg, u.Elem())
		if v == nil {
			return nil
		}
		dir := ast.ChanDir(u.Dir())
		if u.Dir() == types.SendRecv {
			dir = ast.SEND | ast.RECV
		}
		return &ast.CallExpr{
			Fun: ast.NewIdent("make"),
			Args: []ast.Expr{
				&ast.ChanType{
					Dir:   dir,
					Value: v,
				},
			},
		}
	case *types.Struct:
		s := analysisinternal.TypeExpr(fset, f, pkg, typ)
		if s == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: s,
		}
	case *types.Signature:
		var params []*ast.Field
		for i := 0; i < u.Params().Len(); i++ {
			p := analysisinternal.TypeExpr(fset, f, pkg, u.Params().At(i).Type())
			if p == nil {
				return nil
			}
			params = append(params, &ast.Field{
				Type: p,
				Names: []*ast.Ident{
					{
						Name: u.Params().At(i).Name(),
					},
				},
			})
		}
		var returns []*ast.Field
		for i := 0; i < u.Results().Len(); i++ {
			r := analysisinternal.TypeExpr(fset, f, pkg, u.Results().At(i).Type())
			if r == nil {
				return nil
			}
			returns = append(returns, &ast.Field{
				Type: r,
			})
		}
		return &ast.FuncLit{
			Type: &ast.FuncType{
				Params: &ast.FieldList{
					List: params,
				},
				Results: &ast.FieldList{
					List: returns,
				},
			},
			Body: &ast.BlockStmt{},
		}
	case *types.Pointer:
		switch u.Elem().(type) {
		case *types.Basic:
			return &ast.CallExpr{
				Fun: &ast.Ident{
					Name: "new",
				},
				Args: []ast.Expr{
					&ast.Ident{
						Name: u.Elem().String(),
					},
				},
			}
		default:
			return &ast.UnaryExpr{
				Op: token.AND,
				X:  populateValue(fset, f, pkg, u.Elem()),
			}
		}
	case *types.Interface:
		return ast.NewIdent("nil")
	}
	return nil
}
