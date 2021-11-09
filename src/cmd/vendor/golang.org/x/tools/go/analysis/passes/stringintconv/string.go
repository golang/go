// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stringintconv defines an Analyzer that flags type conversions
// from integers to strings.
package stringintconv

import (
	"fmt"
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typeparams"
)

const Doc = `check for string(int) conversions

This checker flags conversions of the form string(x) where x is an integer
(but not byte or rune) type. Such conversions are discouraged because they
return the UTF-8 representation of the Unicode code point x, and not a decimal
string representation of x as one might expect. Furthermore, if x denotes an
invalid code point, the conversion cannot be statically rejected.

For conversions that intend on using the code point, consider replacing them
with string(rune(x)). Otherwise, strconv.Itoa and its equivalents return the
string representation of the value in the desired base.
`

var Analyzer = &analysis.Analyzer{
	Name:     "stringintconv",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

// describe returns a string describing the type typ contained within the type
// set of inType. If non-empty, inName is used as the name of inType (this is
// necessary so that we can use alias type names that may not be reachable from
// inType itself).
func describe(typ, inType types.Type, inName string) string {
	name := inName
	if typ != inType {
		name = typeName(typ)
	}
	if name == "" {
		return ""
	}

	var parentheticals []string
	if underName := typeName(typ.Underlying()); underName != "" && underName != name {
		parentheticals = append(parentheticals, underName)
	}

	if typ != inType && inName != "" && inName != name {
		parentheticals = append(parentheticals, "in "+inName)
	}

	if len(parentheticals) > 0 {
		name += " (" + strings.Join(parentheticals, ", ") + ")"
	}

	return name
}

func typeName(typ types.Type) string {
	if v, _ := typ.(interface{ Name() string }); v != nil {
		return v.Name()
	}
	if v, _ := typ.(interface{ Obj() *types.TypeName }); v != nil {
		return v.Obj().Name()
	}
	return ""
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)

		if len(call.Args) != 1 {
			return
		}
		arg := call.Args[0]

		// Retrieve target type name.
		var tname *types.TypeName
		switch fun := call.Fun.(type) {
		case *ast.Ident:
			tname, _ = pass.TypesInfo.Uses[fun].(*types.TypeName)
		case *ast.SelectorExpr:
			tname, _ = pass.TypesInfo.Uses[fun.Sel].(*types.TypeName)
		}
		if tname == nil {
			return
		}

		// In the conversion T(v) of a value v of type V to a target type T, we
		// look for types T0 in the type set of T and V0 in the type set of V, such
		// that V0->T0 is a problematic conversion. If T and V are not type
		// parameters, this amounts to just checking if V->T is a problematic
		// conversion.

		// First, find a type T0 in T that has an underlying type of string.
		T := tname.Type()
		tterms, err := typeparams.StructuralTerms(T)
		if err != nil {
			return // invalid type
		}

		var T0 types.Type // string type in the type set of T

		for _, term := range tterms {
			u, _ := term.Type().Underlying().(*types.Basic)
			if u != nil && u.Kind() == types.String {
				T0 = term.Type()
				break
			}
		}

		if T0 == nil {
			// No target types have an underlying type of string.
			return
		}

		// Next, find a type V0 in V that has an underlying integral type that is
		// not byte or rune.
		V := pass.TypesInfo.TypeOf(arg)
		vterms, err := typeparams.StructuralTerms(V)
		if err != nil {
			return // invalid type
		}

		var V0 types.Type // integral type in the type set of V

		for _, term := range vterms {
			u, _ := term.Type().Underlying().(*types.Basic)
			if u != nil && u.Info()&types.IsInteger != 0 {
				switch u.Kind() {
				case types.Byte, types.Rune, types.UntypedRune:
					continue
				}
				V0 = term.Type()
				break
			}
		}

		if V0 == nil {
			// No source types are non-byte or rune integer types.
			return
		}

		convertibleToRune := true // if true, we can suggest a fix
		for _, term := range vterms {
			if !types.ConvertibleTo(term.Type(), types.Typ[types.Rune]) {
				convertibleToRune = false
				break
			}
		}

		target := describe(T0, T, tname.Name())
		source := describe(V0, V, typeName(V))

		if target == "" || source == "" {
			return // something went wrong
		}

		diag := analysis.Diagnostic{
			Pos:     n.Pos(),
			Message: fmt.Sprintf("conversion from %s to %s yields a string of one rune, not a string of digits (did you mean fmt.Sprint(x)?)", source, target),
		}

		if convertibleToRune {
			diag.SuggestedFixes = []analysis.SuggestedFix{
				{
					Message: "Did you mean to convert a rune to a string?",
					TextEdits: []analysis.TextEdit{
						{
							Pos:     arg.Pos(),
							End:     arg.Pos(),
							NewText: []byte("rune("),
						},
						{
							Pos:     arg.End(),
							End:     arg.End(),
							NewText: []byte(")"),
						},
					},
				},
			}
		}
		pass.Report(diag)
	})
	return nil, nil
}
