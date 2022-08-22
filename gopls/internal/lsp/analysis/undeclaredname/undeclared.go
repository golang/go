// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package undeclaredname defines an Analyzer that applies suggested fixes
// to errors of the type "undeclared name: %s".
package undeclaredname

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/analysisinternal"
)

const Doc = `suggested fixes for "undeclared name: <>"

This checker provides suggested fixes for type errors of the
type "undeclared name: <>". It will either insert a new statement,
such as:

"<> := "

or a new function declaration, such as:

func <>(inferred parameters) {
	panic("implement me!")
}
`

var Analyzer = &analysis.Analyzer{
	Name:             "undeclaredname",
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{},
	Run:              run,
	RunDespiteErrors: true,
}

// The prefix for this error message changed in Go 1.20.
var undeclaredNamePrefixes = []string{"undeclared name: ", "undefined: "}

func run(pass *analysis.Pass) (interface{}, error) {
	for _, err := range pass.TypeErrors {
		runForError(pass, err)
	}
	return nil, nil
}

func runForError(pass *analysis.Pass, err types.Error) {
	var name string
	for _, prefix := range undeclaredNamePrefixes {
		if !strings.HasPrefix(err.Msg, prefix) {
			continue
		}
		name = strings.TrimPrefix(err.Msg, prefix)
	}
	if name == "" {
		return
	}
	var file *ast.File
	for _, f := range pass.Files {
		if f.Pos() <= err.Pos && err.Pos < f.End() {
			file = f
			break
		}
	}
	if file == nil {
		return
	}

	// Get the path for the relevant range.
	path, _ := astutil.PathEnclosingInterval(file, err.Pos, err.Pos)
	if len(path) < 2 {
		return
	}
	ident, ok := path[0].(*ast.Ident)
	if !ok || ident.Name != name {
		return
	}

	// Undeclared quick fixes only work in function bodies.
	inFunc := false
	for i := range path {
		if _, inFunc = path[i].(*ast.FuncDecl); inFunc {
			if i == 0 {
				return
			}
			if _, isBody := path[i-1].(*ast.BlockStmt); !isBody {
				return
			}
			break
		}
	}
	if !inFunc {
		return
	}
	// Skip selector expressions because it might be too complex
	// to try and provide a suggested fix for fields and methods.
	if _, ok := path[1].(*ast.SelectorExpr); ok {
		return
	}
	tok := pass.Fset.File(file.Pos())
	if tok == nil {
		return
	}
	offset := pass.Fset.Position(err.Pos).Offset
	end := tok.Pos(offset + len(name))
	pass.Report(analysis.Diagnostic{
		Pos:     err.Pos,
		End:     end,
		Message: err.Msg,
	})
}

func SuggestedFix(fset *token.FileSet, rng span.Range, content []byte, file *ast.File, pkg *types.Package, info *types.Info) (*analysis.SuggestedFix, error) {
	pos := rng.Start // don't use the end
	path, _ := astutil.PathEnclosingInterval(file, pos, pos)
	if len(path) < 2 {
		return nil, fmt.Errorf("no expression found")
	}
	ident, ok := path[0].(*ast.Ident)
	if !ok {
		return nil, fmt.Errorf("no identifier found")
	}

	// Check for a possible call expression, in which case we should add a
	// new function declaration.
	if len(path) > 1 {
		if _, ok := path[1].(*ast.CallExpr); ok {
			return newFunctionDeclaration(path, file, pkg, info, fset)
		}
	}

	// Get the place to insert the new statement.
	insertBeforeStmt := analysisinternal.StmtToInsertVarBefore(path)
	if insertBeforeStmt == nil {
		return nil, fmt.Errorf("could not locate insertion point")
	}

	insertBefore := fset.Position(insertBeforeStmt.Pos()).Offset

	// Get the indent to add on the line after the new statement.
	// Since this will have a parse error, we can not use format.Source().
	contentBeforeStmt, indent := content[:insertBefore], "\n"
	if nl := bytes.LastIndex(contentBeforeStmt, []byte("\n")); nl != -1 {
		indent = string(contentBeforeStmt[nl:])
	}

	// Create the new local variable statement.
	newStmt := fmt.Sprintf("%s := %s", ident.Name, indent)
	return &analysis.SuggestedFix{
		Message: fmt.Sprintf("Create variable \"%s\"", ident.Name),
		TextEdits: []analysis.TextEdit{{
			Pos:     insertBeforeStmt.Pos(),
			End:     insertBeforeStmt.Pos(),
			NewText: []byte(newStmt),
		}},
	}, nil
}

func newFunctionDeclaration(path []ast.Node, file *ast.File, pkg *types.Package, info *types.Info, fset *token.FileSet) (*analysis.SuggestedFix, error) {
	if len(path) < 3 {
		return nil, fmt.Errorf("unexpected set of enclosing nodes: %v", path)
	}
	ident, ok := path[0].(*ast.Ident)
	if !ok {
		return nil, fmt.Errorf("no name for function declaration %v (%T)", path[0], path[0])
	}
	call, ok := path[1].(*ast.CallExpr)
	if !ok {
		return nil, fmt.Errorf("no call expression found %v (%T)", path[1], path[1])
	}

	// Find the enclosing function, so that we can add the new declaration
	// below.
	var enclosing *ast.FuncDecl
	for _, n := range path {
		if n, ok := n.(*ast.FuncDecl); ok {
			enclosing = n
			break
		}
	}
	// TODO(rstambler): Support the situation when there is no enclosing
	// function.
	if enclosing == nil {
		return nil, fmt.Errorf("no enclosing function found: %v", path)
	}

	pos := enclosing.End()

	var paramNames []string
	var paramTypes []types.Type
	// keep track of all param names to later ensure uniqueness
	nameCounts := map[string]int{}
	for _, arg := range call.Args {
		typ := info.TypeOf(arg)
		if typ == nil {
			return nil, fmt.Errorf("unable to determine type for %s", arg)
		}

		switch t := typ.(type) {
		// this is the case where another function call returning multiple
		// results is used as an argument
		case *types.Tuple:
			n := t.Len()
			for i := 0; i < n; i++ {
				name := typeToArgName(t.At(i).Type())
				nameCounts[name]++

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
				name = typeToArgName(typ)
			}

			nameCounts[name]++

			paramNames = append(paramNames, name)
			paramTypes = append(paramTypes, types.Default(typ))
		}
	}

	for n, c := range nameCounts {
		// Any names we saw more than once will need a unique suffix added
		// on. Reset the count to 1 to act as the suffix for the first
		// occurrence of that name.
		if c >= 2 {
			nameCounts[n] = 1
		} else {
			delete(nameCounts, n)
		}
	}

	params := &ast.FieldList{}

	for i, name := range paramNames {
		if suffix, repeats := nameCounts[name]; repeats {
			nameCounts[name]++
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
			Type: analysisinternal.TypeExpr(file, pkg, paramTypes[i]),
		})
	}

	decl := &ast.FuncDecl{
		Name: ast.NewIdent(ident.Name),
		Type: &ast.FuncType{
			Params: params,
			// TODO(rstambler): Also handle result parameters here.
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				&ast.ExprStmt{
					X: &ast.CallExpr{
						Fun: ast.NewIdent("panic"),
						Args: []ast.Expr{
							&ast.BasicLit{
								Value: `"unimplemented"`,
							},
						},
					},
				},
			},
		},
	}

	b := bytes.NewBufferString("\n\n")
	if err := format.Node(b, fset, decl); err != nil {
		return nil, err
	}
	return &analysis.SuggestedFix{
		Message: fmt.Sprintf("Create function \"%s\"", ident.Name),
		TextEdits: []analysis.TextEdit{{
			Pos:     pos,
			End:     pos,
			NewText: b.Bytes(),
		}},
	}, nil
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
	a := []rune(s[strings.LastIndexByte(s, '.')+1:])
	a[0] = unicode.ToLower(a[0])
	return string(a)
}
