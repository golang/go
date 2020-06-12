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
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
)

const Doc = `suggested input for incomplete struct initializations

This analyzer provides the appropriate zero values for all
uninitialized fields of a struct. For example, given the following struct:
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
		if fieldCount == 0 {
			return
		}
		var fieldSourceCode strings.Builder
		for i := 0; i < fieldCount; i++ {
			field := obj.Field(i)
			// Ignore fields that are not accessible in the current package.
			if field.Pkg() != nil && field.Pkg() != pass.Pkg && !field.Exported() {
				continue
			}

			label := field.Name()
			value := analysisinternal.ZeroValue(pass.Fset, file, pass.Pkg, field.Type())
			if value == nil {
				continue
			}
			var valBuf bytes.Buffer
			if err := format.Node(&valBuf, pass.Fset, value); err != nil {
				return
			}
			fieldSourceCode.WriteString("\n")
			fieldSourceCode.WriteString(label)
			fieldSourceCode.WriteString(" : ")
			fieldSourceCode.WriteString(valBuf.String())
			fieldSourceCode.WriteString(",")
		}

		if fieldSourceCode.Len() == 0 {
			return
		}

		fieldSourceCode.WriteString("\n")

		buf := []byte(fieldSourceCode.String())

		msg := "Fill struct with default values"
		if name, ok := expr.Type.(*ast.Ident); ok {
			msg = fmt.Sprintf("Fill %s with default values", name)
		}
		pass.Report(analysis.Diagnostic{
			Pos: expr.Lbrace,
			End: expr.Rbrace,
			SuggestedFixes: []analysis.SuggestedFix{{
				Message: msg,
				TextEdits: []analysis.TextEdit{{
					Pos:     expr.Lbrace + 1,
					End:     expr.Rbrace,
					NewText: buf,
				}},
			}},
		})
	})
	return nil, nil
}
