// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements commonly used type predicates.

package types

func isNamed(typ Type) bool {
	if _, ok := typ.(*Basic); ok {
		return ok
	}
	_, ok := typ.(*NamedType)
	return ok
}

func isBoolean(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsBoolean != 0
}

func isInteger(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsInteger != 0
}

func isUnsigned(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsUnsigned != 0
}

func isFloat(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsFloat != 0
}

func isComplex(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsComplex != 0
}

func isNumeric(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsNumeric != 0
}

func isString(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsString != 0
}

func isUntyped(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsUntyped != 0
}

func isOrdered(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsOrdered != 0
}

func isConstType(typ Type) bool {
	t, ok := underlying(typ).(*Basic)
	return ok && t.Info&IsConstType != 0
}

func isComparable(typ Type) bool {
	switch t := underlying(typ).(type) {
	case *Basic:
		return t.Kind != Invalid && t.Kind != UntypedNil
	case *Pointer, *Interface, *Chan:
		// assumes types are equal for pointers and channels
		return true
	case *Struct:
		for _, f := range t.Fields {
			if !isComparable(f.Type) {
				return false
			}
		}
		return true
	case *Array:
		return isComparable(t.Elt)
	}
	return false
}

func hasNil(typ Type) bool {
	switch underlying(typ).(type) {
	case *Slice, *Pointer, *Signature, *Interface, *Map, *Chan:
		return true
	}
	return false
}

// IsIdentical returns true if x and y are identical.
func IsIdentical(x, y Type) bool {
	if x == y {
		return true
	}

	switch x := x.(type) {
	case *Basic:
		// Basic types are singletons except for the rune and byte
		// aliases, thus we cannot solely rely on the x == y check
		// above.
		if y, ok := y.(*Basic); ok {
			return x.Kind == y.Kind
		}

	case *Array:
		// Two array types are identical if they have identical element types
		// and the same array length.
		if y, ok := y.(*Array); ok {
			return x.Len == y.Len && IsIdentical(x.Elt, y.Elt)
		}

	case *Slice:
		// Two slice types are identical if they have identical element types.
		if y, ok := y.(*Slice); ok {
			return IsIdentical(x.Elt, y.Elt)
		}

	case *Struct:
		// Two struct types are identical if they have the same sequence of fields,
		// and if corresponding fields have the same names, and identical types,
		// and identical tags. Two anonymous fields are considered to have the same
		// name. Lower-case field names from different packages are always different.
		if y, ok := y.(*Struct); ok {
			if len(x.Fields) == len(y.Fields) {
				for i, f := range x.Fields {
					g := y.Fields[i]
					if !f.QualifiedName.IsSame(g.QualifiedName) ||
						!IsIdentical(f.Type, g.Type) ||
						f.Tag != g.Tag ||
						f.IsAnonymous != g.IsAnonymous {
						return false
					}
				}
				return true
			}
		}

	case *Pointer:
		// Two pointer types are identical if they have identical base types.
		if y, ok := y.(*Pointer); ok {
			return IsIdentical(x.Base, y.Base)
		}

	case *Signature:
		// Two function types are identical if they have the same number of parameters
		// and result values, corresponding parameter and result types are identical,
		// and either both functions are variadic or neither is. Parameter and result
		// names are not required to match.
		if y, ok := y.(*Signature); ok {
			return identicalTypes(x.Params, y.Params) &&
				identicalTypes(x.Results, y.Results) &&
				x.IsVariadic == y.IsVariadic
		}

	case *Interface:
		// Two interface types are identical if they have the same set of methods with
		// the same names and identical function types. Lower-case method names from
		// different packages are always different. The order of the methods is irrelevant.
		if y, ok := y.(*Interface); ok {
			return identicalMethods(x.Methods, y.Methods) // methods are sorted
		}

	case *Map:
		// Two map types are identical if they have identical key and value types.
		if y, ok := y.(*Map); ok {
			return IsIdentical(x.Key, y.Key) && IsIdentical(x.Elt, y.Elt)
		}

	case *Chan:
		// Two channel types are identical if they have identical value types
		// and the same direction.
		if y, ok := y.(*Chan); ok {
			return x.Dir == y.Dir && IsIdentical(x.Elt, y.Elt)
		}

	case *NamedType:
		// Two named types are identical if their type names originate
		// in the same type declaration.
		if y, ok := y.(*NamedType); ok {
			return x.Obj == y.Obj
		}
	}

	return false
}

// identicalTypes returns true if both lists a and b have the
// same length and corresponding objects have identical types.
func identicalTypes(a, b []*Var) bool {
	if len(a) != len(b) {
		return false
	}
	for i, x := range a {
		y := b[i]
		if !IsIdentical(x.Type, y.Type) {
			return false
		}
	}
	return true
}

// identicalMethods returns true if both lists a and b have the
// same length and corresponding methods have identical types.
// TODO(gri) make this more efficient
func identicalMethods(a, b []*Method) bool {
	if len(a) != len(b) {
		return false
	}
	m := make(map[QualifiedName]*Method)
	for _, x := range a {
		assert(m[x.QualifiedName] == nil) // method list must not have duplicate entries
		m[x.QualifiedName] = x
	}
	for _, y := range b {
		if x := m[y.QualifiedName]; x == nil || !IsIdentical(x.Type, y.Type) {
			return false
		}
	}
	return true
}

// underlying returns the underlying type of typ.
func underlying(typ Type) Type {
	// Basic types are representing themselves directly even though they are named.
	if typ, ok := typ.(*NamedType); ok {
		return typ.Underlying // underlying types are never NamedTypes
	}
	return typ
}

// deref returns a pointer's base type; otherwise it returns typ.
func deref(typ Type) Type {
	if typ, ok := underlying(typ).(*Pointer); ok {
		return typ.Base
	}
	return typ
}

// defaultType returns the default "typed" type for an "untyped" type;
// it returns the incoming type for all other types. If there is no
// corresponding untyped type, the result is Typ[Invalid].
//
func defaultType(typ Type) Type {
	if t, ok := typ.(*Basic); ok {
		k := Invalid
		switch t.Kind {
		// case UntypedNil:
		//      There is no default type for nil. For a good error message,
		//      catch this case before calling this function.
		case UntypedBool:
			k = Bool
		case UntypedInt:
			k = Int
		case UntypedRune:
			k = Rune
		case UntypedFloat:
			k = Float64
		case UntypedComplex:
			k = Complex128
		case UntypedString:
			k = String
		}
		typ = Typ[k]
	}
	return typ
}

// missingMethod returns (nil, false) if typ implements T, otherwise
// it returns the first missing method required by T and whether it
// is missing or simply has the wrong type.
//
func missingMethod(typ Type, T *Interface) (method *Method, wrongType bool) {
	// TODO(gri): this needs to correctly compare method names (taking package into account)
	// TODO(gri): distinguish pointer and non-pointer receivers
	// an interface type implements T if it has no methods with conflicting signatures
	// Note: This is stronger than the current spec. Should the spec require this?
	if ityp, _ := underlying(typ).(*Interface); ityp != nil {
		for _, m := range T.Methods {
			res := lookupField(ityp, m.QualifiedName) // TODO(gri) no need to go via lookupField
			if res.mode != invalid && !IsIdentical(res.typ, m.Type) {
				return m, true
			}
		}
		return
	}

	// a concrete type implements T if it implements all methods of T.
	for _, m := range T.Methods {
		res := lookupField(typ, m.QualifiedName)
		if res.mode == invalid {
			return m, false
		}
		if !IsIdentical(res.typ, m.Type) {
			return m, true
		}
	}
	return
}
