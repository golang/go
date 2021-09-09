// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package implementmissing defines an Analyzer that will attempt to
// automatically implement a function that is currently undeclared.
package implementmissing

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/types"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/analysisinternal"
)

const Doc = `suggested fixes for "undeclared name: %s" on a function call

This checker provides suggested fixes for type errors of the
type "undeclared name: %s" that happen for a function call. For example:
	func m() {
	  a(1)
	}
will turn into
	func m() {
	  a(1)
	}

	func a(i int) {}
`

var Analyzer = &analysis.Analyzer{
	Name:             "implementmissing",
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{},
	Run:              run,
	RunDespiteErrors: true,
}

const undeclaredNamePrefix = "undeclared name: "

func run(pass *analysis.Pass) (interface{}, error) {
	info := pass.TypesInfo
	if info == nil {
		return nil, fmt.Errorf("nil TypeInfo")
	}

	errors := analysisinternal.GetTypeErrors(pass)
	for _, typeErr := range errors {
		// Filter out the errors that are not relevant to this analyzer.
		if !FixesError(typeErr.Msg) {
			continue
		}

		var file *ast.File
		for _, f := range pass.Files {
			if f.Pos() <= typeErr.Pos && typeErr.Pos <= f.End() {
				file = f
				break
			}
		}
		if file == nil {
			continue
		}

		var buf bytes.Buffer
		if err := format.Node(&buf, pass.Fset, file); err != nil {
			continue
		}
		typeErrEndPos := analysisinternal.TypeErrorEndPos(pass.Fset, buf.Bytes(), typeErr.Pos)

		// Get the path for the relevant range.
		path, _ := astutil.PathEnclosingInterval(file, typeErr.Pos, typeErrEndPos)
		if len(path) < 2 {
			return nil, nil
		}

		// Check to make sure we're dealing with a function call, we don't want to
		// deal with undeclared variables here.
		call, ok := path[1].(*ast.CallExpr)
		if !ok {
			return nil, nil
		}

		ident, ok := path[0].(*ast.Ident)
		if !ok {
			return nil, nil
		}

		var paramNames []string
		var paramTypes []types.Type

		// keep track of all param names to later ensure uniqueness
		namesCount := map[string]int{}

		for _, arg := range call.Args {
			ty := pass.TypesInfo.TypeOf(arg)
			if ty == nil {
				return nil, nil
			}

			switch t := ty.(type) {
			// this is the case where another function call returning multiple
			// results is used as an argument
			case *types.Tuple:
				n := t.Len()
				for i := 0; i < n; i++ {
					name := typeToArgName(t.At(i).Type())
					namesCount[name]++

					paramNames = append(paramNames, name)
					paramTypes = append(paramTypes, types.Default(t.At(i).Type()))
				}

			default:
				// does the argument have a name we can reuse?
				// only happens in case of a *ast.Ident
				var name string
				if ident, ok := arg.(*ast.Ident); ok {
					name = ident.Name
				}

				if name == "" {
					name = typeToArgName(ty)
				}

				namesCount[name]++

				paramNames = append(paramNames, name)
				paramTypes = append(paramTypes, types.Default(ty))
			}
		}

		for n, c := range namesCount {
			// Any names we saw more than once will need a unique suffix added
			// on. Reset the count to 1 to act as the suffix for the first
			// occurrence of that name.
			if c >= 2 {
				namesCount[n] = 1
			} else {
				delete(namesCount, n)
			}
		}

		params := &ast.FieldList{
			List: []*ast.Field{},
		}

		for i, name := range paramNames {
			if suffix, repeats := namesCount[name]; repeats {
				namesCount[name]++
				name = fmt.Sprintf("%s%d", name, suffix)
			}

			// only worth checking after previous param in the list
			if i > 0 {
				// if type of parameter at hand is the same as the previous one,
				// add it to the previous param list of identifiers so to have:
				//  (s1, s2 string)
				// and not
				//  (s1 string, s2 string)
				if paramTypes[i] == paramTypes[i-1] {
					params.List[len(params.List)-1].Names = append(params.List[len(params.List)-1].Names, ast.NewIdent(name))
					continue
				}
			}

			params.List = append(params.List, &ast.Field{
				Names: []*ast.Ident{
					ast.NewIdent(name),
				},
				Type: analysisinternal.TypeExpr(pass.Fset, file, pass.Pkg, paramTypes[i]),
			})
		}

		eof := file.End()

		decl := &ast.FuncDecl{
			Name: &ast.Ident{
				Name: ident.Name,
			},
			Type: &ast.FuncType{
				Func:    file.End(),
				Params:  params,
				Results: &ast.FieldList{},
			},
			Body: &ast.BlockStmt{
				List: []ast.Stmt{
					&ast.ExprStmt{
						X: &ast.CallExpr{
							Fun: &ast.Ident{
								Name: "panic",
							},
							Args: []ast.Expr{
								&ast.BasicLit{
									Value: `"not implemented"`,
								},
							},
						},
					},
				},
			},
		}

		var declBuf bytes.Buffer
		if err := format.Node(&declBuf, pass.Fset, decl); err != nil {
			return nil, err
		}

		text := append([]byte("\n\n"), declBuf.Bytes()...)
		text = append(text, []byte("\n")...)

		pass.Report(analysis.Diagnostic{
			Pos:     typeErr.Pos,
			End:     typeErr.Pos,
			Message: typeErr.Msg,
			SuggestedFixes: []analysis.SuggestedFix{
				{
					Message: "Implement function " + ident.Name,
					TextEdits: []analysis.TextEdit{{
						Pos:     eof,
						End:     eof,
						NewText: text,
					}},
				},
			},
			Related: []analysis.RelatedInformation{},
		})
	}
	return nil, nil
}

func FixesError(msg string) bool {
	return strings.HasPrefix(msg, undeclaredNamePrefix)
}

func typeToArgName(ty types.Type) string {
	s := types.Default(ty).String()

	switch t := ty.(type) {
	case *types.Basic:
		// use first letter in type name for basic types
		return s[0:1]
	case *types.Slice:
		// use element type to decide var name for slices
		return typeToArgName(t.Elem())
	case *types.Array:
		// use element type to decide var name for arrays
		return typeToArgName(t.Elem())
	case *types.Chan:
		return "ch"
	}

	s = strings.TrimFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r)
	})

	if s == "error" {
		return "err"
	}

	// remove package (if present)
	// and make first letter lowercase
	parts := strings.Split(s, ".")
	a := []rune(parts[len(parts)-1])
	a[0] = unicode.ToLower(a[0])
	return string(a)
}
