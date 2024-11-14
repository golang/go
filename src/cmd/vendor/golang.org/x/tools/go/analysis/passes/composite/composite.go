// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package composite defines an Analyzer that checks for unkeyed
// composite literals.
package composite

import (
	"fmt"
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/aliases"
	"golang.org/x/tools/internal/typeparams"
)

const Doc = `check for unkeyed composite literals

This analyzer reports a diagnostic for composite literals of struct
types imported from another package that do not use the field-keyed
syntax. Such literals are fragile because the addition of a new field
(even if unexported) to the struct will cause compilation to fail.

As an example,

	err = &net.DNSConfigError{err}

should be replaced by:

	err = &net.DNSConfigError{Err: err}
`

var Analyzer = &analysis.Analyzer{
	Name:             "composites",
	Doc:              Doc,
	URL:              "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/composite",
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	RunDespiteErrors: true,
	Run:              run,
}

var whitelist = true

func init() {
	Analyzer.Flags.BoolVar(&whitelist, "whitelist", whitelist, "use composite white list; for testing only")
}

// runUnkeyedLiteral checks if a composite literal is a struct literal with
// unkeyed fields.
func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CompositeLit)(nil),
	}
	inspect.Preorder(nodeFilter, func { n ->
		cl := n.(*ast.CompositeLit)

		typ := pass.TypesInfo.Types[cl].Type
		if typ == nil {
			// cannot determine composite literals' type, skip it
			return
		}
		typeName := typ.String()
		if whitelist && unkeyedLiteral[typeName] {
			// skip whitelisted types
			return
		}
		var structuralTypes []types.Type
		switch typ := aliases.Unalias(typ).(type) {
		case *types.TypeParam:
			terms, err := typeparams.StructuralTerms(typ)
			if err != nil {
				return // invalid type
			}
			for _, term := range terms {
				structuralTypes = append(structuralTypes, term.Type())
			}
		default:
			structuralTypes = append(structuralTypes, typ)
		}

		for _, typ := range structuralTypes {
			strct, ok := typeparams.Deref(typ).Underlying().(*types.Struct)
			if !ok {
				// skip non-struct composite literals
				continue
			}
			if isLocalType(pass, typ) {
				// allow unkeyed locally defined composite literal
				continue
			}

			// check if the struct contains an unkeyed field
			allKeyValue := true
			var suggestedFixAvailable = len(cl.Elts) == strct.NumFields()
			var missingKeys []analysis.TextEdit
			for i, e := range cl.Elts {
				if _, ok := e.(*ast.KeyValueExpr); !ok {
					allKeyValue = false
					if i >= strct.NumFields() {
						break
					}
					field := strct.Field(i)
					if !field.Exported() {
						// Adding unexported field names for structs not defined
						// locally will not work.
						suggestedFixAvailable = false
						break
					}
					missingKeys = append(missingKeys, analysis.TextEdit{
						Pos:     e.Pos(),
						End:     e.Pos(),
						NewText: []byte(fmt.Sprintf("%s: ", field.Name())),
					})
				}
			}
			if allKeyValue {
				// all the struct fields are keyed
				continue
			}

			diag := analysis.Diagnostic{
				Pos:     cl.Pos(),
				End:     cl.End(),
				Message: fmt.Sprintf("%s struct literal uses unkeyed fields", typeName),
			}
			if suggestedFixAvailable {
				diag.SuggestedFixes = []analysis.SuggestedFix{{
					Message:   "Add field names to struct literal",
					TextEdits: missingKeys,
				}}
			}
			pass.Report(diag)
			return
		}
	})
	return nil, nil
}

// isLocalType reports whether typ belongs to the same package as pass.
// TODO(adonovan): local means "internal to a function"; rename to isSamePackageType.
func isLocalType(pass *analysis.Pass, typ types.Type) bool {
	switch x := aliases.Unalias(typ).(type) {
	case *types.Struct:
		// struct literals are local types
		return true
	case *types.Pointer:
		return isLocalType(pass, x.Elem())
	case interface{ Obj() *types.TypeName }: // *Named or *TypeParam (aliases were removed already)
		// names in package foo are local to foo_test too
		return strings.TrimSuffix(x.Obj().Pkg().Path(), "_test") == strings.TrimSuffix(pass.Pkg.Path(), "_test")
	}
	return false
}
