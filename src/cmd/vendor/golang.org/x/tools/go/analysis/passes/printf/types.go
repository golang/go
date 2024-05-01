// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package printf

import (
	"fmt"
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/aliases"
	"golang.org/x/tools/internal/typeparams"
)

var errorType = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)

// matchArgType reports an error if printf verb t is not appropriate for
// operand arg.
//
// If arg is a type parameter, the verb t must be appropriate for every type in
// the type parameter type set.
func matchArgType(pass *analysis.Pass, t printfArgType, arg ast.Expr) (reason string, ok bool) {
	// %v, %T accept any argument type.
	if t == anyType {
		return "", true
	}

	typ := pass.TypesInfo.Types[arg].Type
	if typ == nil {
		return "", true // probably a type check problem
	}

	m := &argMatcher{t: t, seen: make(map[types.Type]bool)}
	ok = m.match(typ, true)
	return m.reason, ok
}

// argMatcher recursively matches types against the printfArgType t.
//
// To short-circuit recursion, it keeps track of types that have already been
// matched (or are in the process of being matched) via the seen map. Recursion
// arises from the compound types {map,chan,slice} which may be printed with %d
// etc. if that is appropriate for their element types, as well as from type
// parameters, which are expanded to the constituents of their type set.
//
// The reason field may be set to report the cause of the mismatch.
type argMatcher struct {
	t      printfArgType
	seen   map[types.Type]bool
	reason string
}

// match checks if typ matches m's printf arg type. If topLevel is true, typ is
// the actual type of the printf arg, for which special rules apply. As a
// special case, top level type parameters pass topLevel=true when checking for
// matches among the constituents of their type set, as type arguments will
// replace the type parameter at compile time.
func (m *argMatcher) match(typ types.Type, topLevel bool) bool {
	// %w accepts only errors.
	if m.t == argError {
		return types.ConvertibleTo(typ, errorType)
	}

	// If the type implements fmt.Formatter, we have nothing to check.
	if isFormatter(typ) {
		return true
	}

	// If we can use a string, might arg (dynamically) implement the Stringer or Error interface?
	if m.t&argString != 0 && isConvertibleToString(typ) {
		return true
	}

	if typ, _ := aliases.Unalias(typ).(*types.TypeParam); typ != nil {
		// Avoid infinite recursion through type parameters.
		if m.seen[typ] {
			return true
		}
		m.seen[typ] = true
		terms, err := typeparams.StructuralTerms(typ)
		if err != nil {
			return true // invalid type (possibly an empty type set)
		}

		if len(terms) == 0 {
			// No restrictions on the underlying of typ. Type parameters implementing
			// error, fmt.Formatter, or fmt.Stringer were handled above, and %v and
			// %T was handled in matchType. We're about to check restrictions the
			// underlying; if the underlying type is unrestricted there must be an
			// element of the type set that violates one of the arg type checks
			// below, so we can safely return false here.

			if m.t == anyType { // anyType must have already been handled.
				panic("unexpected printfArgType")
			}
			return false
		}

		// Only report a reason if typ is the argument type, otherwise it won't
		// make sense. Note that it is not sufficient to check if topLevel == here,
		// as type parameters can have a type set consisting of other type
		// parameters.
		reportReason := len(m.seen) == 1

		for _, term := range terms {
			if !m.match(term.Type(), topLevel) {
				if reportReason {
					if term.Tilde() {
						m.reason = fmt.Sprintf("contains ~%s", term.Type())
					} else {
						m.reason = fmt.Sprintf("contains %s", term.Type())
					}
				}
				return false
			}
		}
		return true
	}

	typ = typ.Underlying()
	if m.seen[typ] {
		// We've already considered typ, or are in the process of considering it.
		// In case we've already considered typ, it must have been valid (else we
		// would have stopped matching). In case we're in the process of
		// considering it, we must avoid infinite recursion.
		//
		// There are some pathological cases where returning true here is
		// incorrect, for example `type R struct { F []R }`, but these are
		// acceptable false negatives.
		return true
	}
	m.seen[typ] = true

	switch typ := typ.(type) {
	case *types.Signature:
		return m.t == argPointer

	case *types.Map:
		if m.t == argPointer {
			return true
		}
		// Recur: map[int]int matches %d.
		return m.match(typ.Key(), false) && m.match(typ.Elem(), false)

	case *types.Chan:
		return m.t&argPointer != 0

	case *types.Array:
		// Same as slice.
		if types.Identical(typ.Elem().Underlying(), types.Typ[types.Byte]) && m.t&argString != 0 {
			return true // %s matches []byte
		}
		// Recur: []int matches %d.
		return m.match(typ.Elem(), false)

	case *types.Slice:
		// Same as array.
		if types.Identical(typ.Elem().Underlying(), types.Typ[types.Byte]) && m.t&argString != 0 {
			return true // %s matches []byte
		}
		if m.t == argPointer {
			return true // %p prints a slice's 0th element
		}
		// Recur: []int matches %d. But watch out for
		//	type T []T
		// If the element is a pointer type (type T[]*T), it's handled fine by the Pointer case below.
		return m.match(typ.Elem(), false)

	case *types.Pointer:
		// Ugly, but dealing with an edge case: a known pointer to an invalid type,
		// probably something from a failed import.
		if typ.Elem() == types.Typ[types.Invalid] {
			return true // special case
		}
		// If it's actually a pointer with %p, it prints as one.
		if m.t == argPointer {
			return true
		}

		if typeparams.IsTypeParam(typ.Elem()) {
			return true // We don't know whether the logic below applies. Give up.
		}

		under := typ.Elem().Underlying()
		switch under.(type) {
		case *types.Struct: // see below
		case *types.Array: // see below
		case *types.Slice: // see below
		case *types.Map: // see below
		default:
			// Check whether the rest can print pointers.
			return m.t&argPointer != 0
		}
		// If it's a top-level pointer to a struct, array, slice, type param, or
		// map, that's equivalent in our analysis to whether we can
		// print the type being pointed to. Pointers in nested levels
		// are not supported to minimize fmt running into loops.
		if !topLevel {
			return false
		}
		return m.match(under, false)

	case *types.Struct:
		// report whether all the elements of the struct match the expected type. For
		// instance, with "%d" all the elements must be printable with the "%d" format.
		for i := 0; i < typ.NumFields(); i++ {
			typf := typ.Field(i)
			if !m.match(typf.Type(), false) {
				return false
			}
			if m.t&argString != 0 && !typf.Exported() && isConvertibleToString(typf.Type()) {
				// Issue #17798: unexported Stringer or error cannot be properly formatted.
				return false
			}
		}
		return true

	case *types.Interface:
		// There's little we can do.
		// Whether any particular verb is valid depends on the argument.
		// The user may have reasonable prior knowledge of the contents of the interface.
		return true

	case *types.Basic:
		switch typ.Kind() {
		case types.UntypedBool,
			types.Bool:
			return m.t&argBool != 0

		case types.UntypedInt,
			types.Int,
			types.Int8,
			types.Int16,
			types.Int32,
			types.Int64,
			types.Uint,
			types.Uint8,
			types.Uint16,
			types.Uint32,
			types.Uint64,
			types.Uintptr:
			return m.t&argInt != 0

		case types.UntypedFloat,
			types.Float32,
			types.Float64:
			return m.t&argFloat != 0

		case types.UntypedComplex,
			types.Complex64,
			types.Complex128:
			return m.t&argComplex != 0

		case types.UntypedString,
			types.String:
			return m.t&argString != 0

		case types.UnsafePointer:
			return m.t&(argPointer|argInt) != 0

		case types.UntypedRune:
			return m.t&(argInt|argRune) != 0

		case types.UntypedNil:
			return false

		case types.Invalid:
			return true // Probably a type check problem.
		}
		panic("unreachable")
	}

	return false
}

func isConvertibleToString(typ types.Type) bool {
	if bt, ok := aliases.Unalias(typ).(*types.Basic); ok && bt.Kind() == types.UntypedNil {
		// We explicitly don't want untyped nil, which is
		// convertible to both of the interfaces below, as it
		// would just panic anyway.
		return false
	}
	if types.ConvertibleTo(typ, errorType) {
		return true // via .Error()
	}

	// Does it implement fmt.Stringer?
	if obj, _, _ := types.LookupFieldOrMethod(typ, false, nil, "String"); obj != nil {
		if fn, ok := obj.(*types.Func); ok {
			sig := fn.Type().(*types.Signature)
			if sig.Params().Len() == 0 &&
				sig.Results().Len() == 1 &&
				sig.Results().At(0).Type() == types.Typ[types.String] {
				return true
			}
		}
	}

	return false
}
