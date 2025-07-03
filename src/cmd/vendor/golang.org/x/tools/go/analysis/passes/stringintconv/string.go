// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stringintconv

import (
	_ "embed"
	"fmt"
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "stringintconv",
	Doc:      analysisutil.MustExtractDoc(doc, "stringintconv"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/stringintconv",
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

func typeName(t types.Type) string {
	if basic, ok := t.(*types.Basic); ok {
		return basic.Name() // may be (e.g.) "untyped int", which has no TypeName
	}
	if tname := typesinternal.TypeNameFor(t); tname != nil {
		return tname.Name()
	}
	return ""
}

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.File)(nil),
		(*ast.CallExpr)(nil),
	}
	var file *ast.File
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		if n, ok := n.(*ast.File); ok {
			file = n
			return
		}
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
		ttypes, err := structuralTypes(T)
		if err != nil {
			return // invalid type
		}

		var T0 types.Type // string type in the type set of T

		for _, tt := range ttypes {
			u, _ := tt.Underlying().(*types.Basic)
			if u != nil && u.Kind() == types.String {
				T0 = tt
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
		vtypes, err := structuralTypes(V)
		if err != nil {
			return // invalid type
		}

		var V0 types.Type // integral type in the type set of V

		for _, vt := range vtypes {
			u, _ := vt.Underlying().(*types.Basic)
			if u != nil && u.Info()&types.IsInteger != 0 {
				switch u.Kind() {
				case types.Byte, types.Rune, types.UntypedRune:
					continue
				}
				V0 = vt
				break
			}
		}

		if V0 == nil {
			// No source types are non-byte or rune integer types.
			return
		}

		convertibleToRune := true // if true, we can suggest a fix
		for _, t := range vtypes {
			if !types.ConvertibleTo(t, types.Typ[types.Rune]) {
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
			Message: fmt.Sprintf("conversion from %s to %s yields a string of one rune, not a string of digits", source, target),
		}
		addFix := func(message string, edits []analysis.TextEdit) {
			diag.SuggestedFixes = append(diag.SuggestedFixes, analysis.SuggestedFix{
				Message:   message,
				TextEdits: edits,
			})
		}

		// Fix 1: use fmt.Sprint(x)
		//
		// Prefer fmt.Sprint over strconv.Itoa, FormatInt,
		// or FormatUint, as it works for any type.
		// Add an import of "fmt" as needed.
		//
		// Unless the type is exactly string, we must retain the conversion.
		//
		// Do not offer this fix if type parameters are involved,
		// as there are too many combinations and subtleties.
		// Consider x = rune | int16 | []byte: in all cases,
		// string(x) is legal, but the appropriate diagnostic
		// and fix differs. Similarly, don't offer the fix if
		// the type has methods, as some {String,GoString,Format}
		// may change the behavior of fmt.Sprint.
		if len(ttypes) == 1 && len(vtypes) == 1 && types.NewMethodSet(V0).Len() == 0 {
			_, prefix, importEdits := analysisinternal.AddImport(pass.TypesInfo, file, "fmt", "fmt", "Sprint", arg.Pos())
			if types.Identical(T0, types.Typ[types.String]) {
				// string(x) -> fmt.Sprint(x)
				addFix("Format the number as a decimal", append(importEdits,
					analysis.TextEdit{
						Pos:     call.Fun.Pos(),
						End:     call.Fun.End(),
						NewText: []byte(prefix + "Sprint"),
					}),
				)
			} else {
				// mystring(x) -> mystring(fmt.Sprint(x))
				addFix("Format the number as a decimal", append(importEdits,
					analysis.TextEdit{
						Pos:     call.Lparen + 1,
						End:     call.Lparen + 1,
						NewText: []byte(prefix + "Sprint("),
					},
					analysis.TextEdit{
						Pos:     call.Rparen,
						End:     call.Rparen,
						NewText: []byte(")"),
					}),
				)
			}
		}

		// Fix 2: use string(rune(x))
		if convertibleToRune {
			addFix("Convert a single rune to a string", []analysis.TextEdit{
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
			})
		}
		pass.Report(diag)
	})
	return nil, nil
}

func structuralTypes(t types.Type) ([]types.Type, error) {
	var structuralTypes []types.Type
	if tp, ok := types.Unalias(t).(*types.TypeParam); ok {
		terms, err := typeparams.StructuralTerms(tp)
		if err != nil {
			return nil, err
		}
		for _, term := range terms {
			structuralTypes = append(structuralTypes, term.Type())
		}
	} else {
		structuralTypes = append(structuralTypes, t)
	}
	return structuralTypes, nil
}
