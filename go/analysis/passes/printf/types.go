package printf

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
)

var errorType = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)

// matchArgType reports an error if printf verb t is not appropriate
// for operand arg.
//
// typ is used only for recursive calls; external callers must supply nil.
//
// (Recursion arises from the compound types {map,chan,slice} which
// may be printed with %d etc. if that is appropriate for their element
// types.)
func matchArgType(pass *analysis.Pass, t printfArgType, typ types.Type, arg ast.Expr) bool {
	return matchArgTypeInternal(pass, t, typ, arg, make(map[types.Type]bool))
}

// matchArgTypeInternal is the internal version of matchArgType. It carries a map
// remembering what types are in progress so we don't recur when faced with recursive
// types or mutually recursive types.
func matchArgTypeInternal(pass *analysis.Pass, t printfArgType, typ types.Type, arg ast.Expr, inProgress map[types.Type]bool) bool {
	// %v, %T accept any argument type.
	if t == anyType {
		return true
	}
	if typ == nil {
		// external call
		typ = pass.TypesInfo.Types[arg].Type
		if typ == nil {
			return true // probably a type check problem
		}
	}
	// If the type implements fmt.Formatter, we have nothing to check.
	if isFormatter(pass, typ) {
		return true
	}
	// If we can use a string, might arg (dynamically) implement the Stringer or Error interface?
	if t&argString != 0 && isConvertibleToString(pass, typ) {
		return true
	}

	typ = typ.Underlying()
	if inProgress[typ] {
		// We're already looking at this type. The call that started it will take care of it.
		return true
	}
	inProgress[typ] = true

	switch typ := typ.(type) {
	case *types.Signature:
		return t == argPointer

	case *types.Map:
		return t == argPointer ||
			// Recur: map[int]int matches %d.
			(matchArgTypeInternal(pass, t, typ.Key(), arg, inProgress) && matchArgTypeInternal(pass, t, typ.Elem(), arg, inProgress))

	case *types.Chan:
		return t&argPointer != 0

	case *types.Array:
		// Same as slice.
		if types.Identical(typ.Elem().Underlying(), types.Typ[types.Byte]) && t&argString != 0 {
			return true // %s matches []byte
		}
		// Recur: []int matches %d.
		return matchArgTypeInternal(pass, t, typ.Elem(), arg, inProgress)

	case *types.Slice:
		// Same as array.
		if types.Identical(typ.Elem().Underlying(), types.Typ[types.Byte]) && t&argString != 0 {
			return true // %s matches []byte
		}
		if t == argPointer {
			return true // %p prints a slice's 0th element
		}
		// Recur: []int matches %d. But watch out for
		//	type T []T
		// If the element is a pointer type (type T[]*T), it's handled fine by the Pointer case below.
		return matchArgTypeInternal(pass, t, typ.Elem(), arg, inProgress)

	case *types.Pointer:
		// Ugly, but dealing with an edge case: a known pointer to an invalid type,
		// probably something from a failed import.
		if typ.Elem().String() == "invalid type" {
			if false {
				pass.Reportf(arg.Pos(), "printf argument %v is pointer to invalid or unknown type", analysisutil.Format(pass.Fset, arg))
			}
			return true // special case
		}
		// If it's actually a pointer with %p, it prints as one.
		if t == argPointer {
			return true
		}

		under := typ.Elem().Underlying()
		switch under.(type) {
		case *types.Struct: // see below
		case *types.Array: // see below
		case *types.Slice: // see below
		case *types.Map: // see below
		default:
			// Check whether the rest can print pointers.
			return t&argPointer != 0
		}
		// If it's a top-level pointer to a struct, array, slice, or
		// map, that's equivalent in our analysis to whether we can
		// print the type being pointed to. Pointers in nested levels
		// are not supported to minimize fmt running into loops.
		if len(inProgress) > 1 {
			return false
		}
		return matchArgTypeInternal(pass, t, under, arg, inProgress)

	case *types.Struct:
		return matchStructArgType(pass, t, typ, arg, inProgress)

	case *types.Interface:
		// There's little we can do.
		// Whether any particular verb is valid depends on the argument.
		// The user may have reasonable prior knowledge of the contents of the interface.
		return true

	case *types.Basic:
		switch typ.Kind() {
		case types.UntypedBool,
			types.Bool:
			return t&argBool != 0

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
			return t&argInt != 0

		case types.UntypedFloat,
			types.Float32,
			types.Float64:
			return t&argFloat != 0

		case types.UntypedComplex,
			types.Complex64,
			types.Complex128:
			return t&argComplex != 0

		case types.UntypedString,
			types.String:
			return t&argString != 0

		case types.UnsafePointer:
			return t&(argPointer|argInt) != 0

		case types.UntypedRune:
			return t&(argInt|argRune) != 0

		case types.UntypedNil:
			return false

		case types.Invalid:
			if false {
				pass.Reportf(arg.Pos(), "printf argument %v has invalid or unknown type", analysisutil.Format(pass.Fset, arg))
			}
			return true // Probably a type check problem.
		}
		panic("unreachable")
	}

	return false
}

func isConvertibleToString(pass *analysis.Pass, typ types.Type) bool {
	if bt, ok := typ.(*types.Basic); ok && bt.Kind() == types.UntypedNil {
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

// hasBasicType reports whether x's type is a types.Basic with the given kind.
func hasBasicType(pass *analysis.Pass, x ast.Expr, kind types.BasicKind) bool {
	t := pass.TypesInfo.Types[x].Type
	if t != nil {
		t = t.Underlying()
	}
	b, ok := t.(*types.Basic)
	return ok && b.Kind() == kind
}

// matchStructArgType reports whether all the elements of the struct match the expected
// type. For instance, with "%d" all the elements must be printable with the "%d" format.
func matchStructArgType(pass *analysis.Pass, t printfArgType, typ *types.Struct, arg ast.Expr, inProgress map[types.Type]bool) bool {
	for i := 0; i < typ.NumFields(); i++ {
		typf := typ.Field(i)
		if !matchArgTypeInternal(pass, t, typf.Type(), arg, inProgress) {
			return false
		}
		if t&argString != 0 && !typf.Exported() && isConvertibleToString(pass, typf.Type()) {
			// Issue #17798: unexported Stringer or error cannot be properly fomatted.
			return false
		}
	}
	return true
}
