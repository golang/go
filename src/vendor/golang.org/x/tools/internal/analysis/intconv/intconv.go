// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package intconv implements the logic of the stringintconv
// analyzer, which flags string(int) conversions. It is shared by
// the analyzer and by gopls, which offers the same fixes for the
// type error that go1.28 reports for such conversions.
package intconv

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"math"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
)

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

// Check reports whether the call is a conversion string(x) from a
// non-byte, non-rune integer x, and if so, returns a diagnostic
// with suggested fixes.
//
// Such conversions are rejected by the type checker in files
// using go1.28 or later, so in practice the analyzer only sees
// them in older files. (An exception is a tool built with a
// pre-go1.28 go/types analyzing a go1.28 file, in which case
// the diagnostic and its fix are just as useful.) However,
// Check does not require that the package be well typed, so
// gopls uses it to offer fixes for the type error.
func Check(info *types.Info, curCall inspector.Cursor) (_ analysis.Diagnostic, ok bool) {
	call := curCall.Node().(*ast.CallExpr)

	if len(call.Args) != 1 {
		return
	}
	arg := call.Args[0]

	// Retrieve target type name.
	//
	// TODO(adonovan): simplify. info.Types[call.Fun].IsType() yields the
	// conversion's target type directly (and handles parenthesized and
	// instantiated types too), though describe still needs a name
	// to report aliases.
	var tname *types.TypeName
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		tname, _ = info.Uses[fun].(*types.TypeName)
	case *ast.SelectorExpr:
		tname, _ = info.Uses[fun.Sel].(*types.TypeName)
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
	var T0 types.Type // string type in the type set of T

	ttypes := typeparams.TypeSetOf(T)
	for term := range ttypes.Terms() {
		tt := term.Type()
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
	V := info.TypeOf(arg)
	vtypes := typeparams.TypeSetOf(V)

	var V0 types.Type // integral type in the type set of V

	for term := range vtypes.Terms() {
		vt := term.Type()
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
	for term := range vtypes.Terms() {
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
		Pos:     call.Pos(),
		Message: fmt.Sprintf("conversion from %s to %s yields a string of one rune, not a string of digits", source, target),
	}
	if !convertibleToRune {
		return diag, true // no fixes
	}
	addFix := func(message string, edits ...analysis.TextEdit) {
		diag.SuggestedFixes = append(diag.SuggestedFixes, analysis.SuggestedFix{
			Message:   message,
			TextEdits: edits,
		})
	}
	edit := func(start, end token.Pos, text string) analysis.TextEdit {
		return analysis.TextEdit{Pos: start, End: end, NewText: []byte(text)}
	}

	// sprintf returns edits that replace x by fmt.Sprintf("%verb", x),
	// adding an import of "fmt" as needed.
	// The conversion is retained unless T is exactly string:
	//
	//	string(x)   -> fmt.Sprintf("%verb", x)
	//	mystring(x) -> mystring(fmt.Sprintf("%verb", x))
	sprintf := func(verb string) []analysis.TextEdit {
		start, end := arg.Pos(), arg.End()
		if types.Identical(T, types.Typ[types.String]) {
			start, end = call.Pos(), call.End()
		}
		file := astutil.EnclosingFile(curCall)
		prefix, importEdits := refactor.AddImport(info, file, "fmt", "fmt", "Sprintf", arg.Pos())
		return append(importEdits,
			edit(start, arg.Pos(), prefix+`Sprintf("%`+verb+`", `),
			edit(arg.End(), end, ")"))
	}

	// sprintfOK reports whether fmt.Sprintf may be used.
	//
	// Do not use it if type parameters are involved,
	// as there are too many combinations and subtleties.
	// Consider x = rune | int16 | []byte: in all cases,
	// string(x) is legal, but the appropriate diagnostic
	// and fix differs. Similarly, don't use it if the type
	// has methods, as a Format method may change the behavior
	// of fmt.Sprintf.
	sprintfOK := ttypes.Len() == 1 && vtypes.Len() == 1 && types.NewMethodSet(V0).Len() == 0

	// The first fix must preserve the existing behavior,
	// since it is the one applied by "go fix". (This lets
	// "go fix" migrate code before its go.mod file is
	// upgraded to go1.28, at which point string(int)
	// conversions become a compile error.)
	//
	// If no behavior-preserving fix is available,
	// we offer no fixes at all.

	// Fix 1: use string(rune(x)) or fmt.Sprintf("%c", x),
	// preserving behavior.
	//
	// string(x) yields "\uFFFD" for any x outside the range of valid
	// code points, whereas rune(x) truncates x to 32 bits, which may
	// turn an invalid code point into a valid one. So rune(x) is
	// exact only if x is a constant that fits in 32 bits, or if every
	// type in the type set of V is at most 32 bits wide.
	// (For uint32, values ≥ 1<<31 become negative runes, which are
	// just as invalid.)
	//
	// For a wider type we use fmt.Sprintf("%c", x), which is exactly
	// equivalent: fmt converts x to uint64 and treats any value
	// above MaxRune (including negative values) as "\uFFFD".
	// In addition to the sprintfOK conditions, we avoid a
	// single-term type parameter (e.g. ~int64), since it
	// could be instantiated by a type with a Format method,
	// and untyped constants (whose default type int might
	// overflow).
	if runeIsExact(info, arg, vtypes) {
		addFix("Convert a single rune to a string",
			edit(arg.Pos(), arg.Pos(), "rune("),
			edit(arg.End(), arg.End(), ")"))

	} else if sprintfOK &&
		!is[*types.TypeParam](types.Unalias(V)) &&
		!isUntypedConst(info, arg) {
		addFix("Convert single rune to string (preserves behavior)", sprintf("c")...)

	} else {
		return diag, true // no behavior-preserving fix, so no fixes
	}

	// Fix 2: use fmt.Sprintf("%d", x), which changes behavior.
	//
	// Prefer fmt.Sprintf over strconv.Itoa, FormatInt,
	// or FormatUint, as it works for any integer type.
	if sprintfOK {
		addFix("Format number as decimal (changes behavior)", sprintf("d")...)
	}

	return diag, true
}

// runeIsExact reports whether string(rune(x)) is equivalent to
// string(x) for the argument x, whose type set is vtypes.
func runeIsExact(info *types.Info, x ast.Expr, vtypes typeparams.TypeSet) bool {
	// Constant that fits in 32 bits?
	if tv := info.Types[x]; tv.Value != nil {
		v, exact := constant.Int64Val(constant.ToInt(tv.Value))
		return exact && math.MinInt32 <= v && v <= math.MaxInt32
	}

	// All types in the type set at most 32 bits wide?
	for under := range vtypes.UnderlyingTypes() {
		u, ok := under.(*types.Basic)
		if !ok {
			return false
		}
		switch u.Kind() {
		case types.Int8, types.Int16, types.Int32,
			types.Uint8, types.Uint16, types.Uint32:
		default:
			return false // int, int64, uint, uint64, uintptr, etc
		}
	}
	return true
}

// isUntypedConst reports whether x is an untyped constant.
func isUntypedConst(info *types.Info, x ast.Expr) bool {
	tv := info.Types[x]
	if tv.Value == nil {
		return false
	}
	u, ok := tv.Type.(*types.Basic)
	return ok && u.Info()&types.IsUntyped != 0
}

func is[T any](x any) bool {
	_, ok := x.(T)
	return ok
}
