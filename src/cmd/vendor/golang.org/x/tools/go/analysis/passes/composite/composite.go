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
	"slices"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
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

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	for curLit := range inspect.Root().Preorder((*ast.CompositeLit)(nil)) {
		complit := curLit.Node().(*ast.CompositeLit)

		// Skip empty or partly/fully keyed literals.
		if len(complit.Elts) == 0 ||
			slices.ContainsFunc(complit.Elts, func(e ast.Expr) bool { return is[*ast.KeyValueExpr](e) }) {
			continue
		}

		// Find struct type.
		// (For a type parameter, choose an arbitrary term.)
		typ := pass.TypesInfo.Types[complit].Type
		if typ == nil {
			continue // no type info
		}
		terms, err := typeparams.NormalTerms(typ)
		if err != nil || len(terms) == 0 {
			continue // invalid or empty type
		}
		t := terms[0].Type()
		strct, ok := typeparams.Deref(t).Underlying().(*types.Struct)
		if !ok {
			continue // not a struct literal
		}
		if isSamePackageType(pass, t) {
			continue // allow unkeyed literals for structs in same package
		}

		// Allow whitelisted types.
		typeName := typ.String()
		if whitelist && unkeyedLiteral[typeName] {
			continue
		}

		// If there is one value per field,
		// offer to fill in the field names.
		var fixes []analysis.SuggestedFix
		if len(complit.Elts) == strct.NumFields() {
			var edits []analysis.TextEdit
			for i, elt := range complit.Elts {
				field := strct.Field(i)
				// We cannot fill in the name of an
				// exported field from another package.
				if !field.Exported() {
					edits = nil
					break
				}
				edits = append(edits, analysis.TextEdit{
					Pos:     elt.Pos(),
					End:     elt.Pos(),
					NewText: fmt.Appendf(nil, "%s: ", field.Name()),
				})
			}
			if edits != nil {
				fixes = []analysis.SuggestedFix{{
					Message:   "Add field names to struct literal",
					TextEdits: edits,
				}}
			}
		}

		pass.Report(analysis.Diagnostic{
			Pos:            complit.Pos(),
			End:            complit.End(),
			Message:        fmt.Sprintf("%s struct literal uses unkeyed fields", typeName),
			SuggestedFixes: fixes,
		})
	}
	return nil, nil
}

// isSamePackageType reports whether typ belongs to the same package as pass.
func isSamePackageType(pass *analysis.Pass, typ types.Type) bool {
	switch x := types.Unalias(typ).(type) {
	case *types.Struct:
		// struct literals are local types
		return true
	case *types.Pointer:
		return isSamePackageType(pass, x.Elem())
	case interface{ Obj() *types.TypeName }: // *Named or *TypeParam (aliases were removed already)
		// names in package foo are local to foo_test too
		return x.Obj().Pkg() != nil &&
			strings.TrimSuffix(x.Obj().Pkg().Path(), "_test") == strings.TrimSuffix(pass.Pkg.Path(), "_test")
	}
	return false
}

func is[T any](x any) bool {
	_, ok := x.(T)
	return ok
}
