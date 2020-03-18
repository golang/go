// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements commonly used type predicates.

package types

import (
	"go/token"
	"sort"
)

func isNamed(typ Type) bool {
	if _, ok := typ.(*Basic); ok {
		return ok
	}
	_, ok := typ.(*Named)
	return ok
}

// isGeneric reports whether a type is a generic, uninstantiated type (generic signatures are not included).
func isGeneric(typ Type) bool {
	// A parameterized type is only instantiated if it doesn't have an instantiation already.
	named, _ := typ.(*Named)
	return named != nil && named.obj != nil && named.tparams != nil && named.targs == nil
}

func is(typ Type, what BasicInfo) bool {
	switch t := typ.Underlying().(type) {
	case *Basic:
		return t.info&what != 0
	case *TypeParam:
		return t.Interface().is(func(typ Type) bool { return is(typ, what) })
	}
	return false
}

func isBoolean(typ Type) bool  { return is(typ, IsBoolean) }
func isInteger(typ Type) bool  { return is(typ, IsInteger) }
func isUnsigned(typ Type) bool { return is(typ, IsUnsigned) }
func isFloat(typ Type) bool    { return is(typ, IsFloat) }
func isComplex(typ Type) bool  { return is(typ, IsComplex) }
func isNumeric(typ Type) bool  { return is(typ, IsNumeric) }
func isString(typ Type) bool   { return is(typ, IsString) }

func isTyped(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return !ok || t.info&IsUntyped == 0
}

func isUntyped(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsUntyped != 0
}

func isOrdered(typ Type) bool { return is(typ, IsOrdered) }

func isConstType(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsConstType != 0
}

// IsInterface reports whether typ is an interface type.
func IsInterface(typ Type) bool {
	_, ok := typ.Underlying().(*Interface)
	return ok
}

// Comparable reports whether values of type T are comparable.
func Comparable(T Type) bool {
	switch t := T.Underlying().(type) {
	case *Basic:
		// assume invalid types to be comparable
		// to avoid follow-up errors
		return t.kind != UntypedNil
	case *Pointer, *Interface, *Chan:
		return true
	case *Struct:
		for _, f := range t.fields {
			if !Comparable(f.typ) {
				return false
			}
		}
		return true
	case *Array:
		return Comparable(t.elem)
	case *TypeParam:
		iface := t.Interface()
		// If the magic method == exists, the type parameter is comparable.
		_, m := lookupMethod(iface.allMethods, nil, "==")
		return m != nil || iface.is(Comparable)
	}
	return false
}

// hasNil reports whether a type includes the nil value.
func hasNil(typ Type) bool {
	switch t := typ.Underlying().(type) {
	case *Basic:
		return t.kind == UnsafePointer
	case *Slice, *Pointer, *Signature, *Interface, *Map, *Chan:
		return true
	}
	return false
}

// identical reports whether x and y are identical types.
// Receivers of Signature types are ignored.
func (check *Checker) identical(x, y Type) bool {
	return check.identical0(x, y, true, nil, nil)
}

// identicalIgnoreTags reports whether x and y are identical types if tags are ignored.
// Receivers of Signature types are ignored.
func (check *Checker) identicalIgnoreTags(x, y Type) bool {
	return check.identical0(x, y, false, nil, nil)
}

// An ifacePair is a node in a stack of interface type pairs compared for identity.
type ifacePair struct {
	x, y *Interface
	prev *ifacePair
}

func (p *ifacePair) identical(q *ifacePair) bool {
	return p.x == q.x && p.y == q.y || p.x == q.y && p.y == q.x
}

// If a non-nil tparams is provided, type inference is done for type parameters in x.
func (check *Checker) identical0(x, y Type, cmpTags bool, p *ifacePair, tparams []Type) bool {
	// If we want type inference, do not shortcut for equal types. Instead
	// keep comparing them element-wise so we can infer the matching (and
	// equal type parameter types). A simple test case where this matters
	// is: func f(type T)(x T) { f(x) } .
	// (If we know that the types are equal, we could optimize this case by
	// simply extracting the type parameters used and then populate tparams
	// accordingly. Not clear it's worthwhile the parallel, if slightly more
	// efficient code structure.)
	if tparams == nil && x == y {
		return true
	}

	switch x := x.(type) {
	case *Basic:
		// Basic types are singletons except for the rune and byte
		// aliases, thus we cannot solely rely on the x == y check
		// above. See also comment in TypeName.IsAlias.
		if y, ok := y.(*Basic); ok {
			return x.kind == y.kind
		}

	case *Array:
		// Two array types are identical if they have identical element types
		// and the same array length.
		if y, ok := y.(*Array); ok {
			// If one or both array lengths are unknown (< 0) due to some error,
			// assume they are the same to avoid spurious follow-on errors.
			return (x.len < 0 || y.len < 0 || x.len == y.len) && check.identical0(x.elem, y.elem, cmpTags, p, tparams)
		}

	case *Slice:
		// Two slice types are identical if they have identical element types.
		if y, ok := y.(*Slice); ok {
			return check.identical0(x.elem, y.elem, cmpTags, p, tparams)
		}

	case *Struct:
		// Two struct types are identical if they have the same sequence of fields,
		// and if corresponding fields have the same names, and identical types,
		// and identical tags. Two embedded fields are considered to have the same
		// name. Lower-case field names from different packages are always different.
		if y, ok := y.(*Struct); ok {
			if x.NumFields() == y.NumFields() {
				for i, f := range x.fields {
					g := y.fields[i]
					if f.embedded != g.embedded ||
						cmpTags && x.Tag(i) != y.Tag(i) ||
						!f.sameId(g.pkg, g.name) ||
						!check.identical0(f.typ, g.typ, cmpTags, p, tparams) {
						return false
					}
				}
				return true
			}
		}

	case *Pointer:
		// Two pointer types are identical if they have identical base types.
		if y, ok := y.(*Pointer); ok {
			return check.identical0(x.base, y.base, cmpTags, p, tparams)
		}

	case *Tuple:
		// Two tuples types are identical if they have the same number of elements
		// and corresponding elements have identical types.
		if y, ok := y.(*Tuple); ok {
			if x.Len() == y.Len() {
				if x != nil {
					for i, v := range x.vars {
						w := y.vars[i]
						if !check.identical0(v.typ, w.typ, cmpTags, p, tparams) {
							return false
						}
					}
				}
				return true
			}
		}

	case *Signature:
		// Two function types are identical if they have the same number of parameters
		// and result values, corresponding parameter and result types are identical,
		// and either both functions are variadic or neither is. Parameter and result
		// names are not required to match.
		// Generic functions must also have matching type parameter lists, but for the
		// parameter names.
		if y, ok := y.(*Signature); ok {
			return x.variadic == y.variadic &&
				check.identicalTParams(x.tparams, y.tparams, cmpTags, p, tparams) &&
				check.identical0(x.params, y.params, cmpTags, p, tparams) &&
				check.identical0(x.results, y.results, cmpTags, p, tparams)
		}

	case *Interface:
		// Two interface types are identical if they have the same set of methods with
		// the same names and identical function types. Lower-case method names from
		// different packages are always different. The order of the methods is irrelevant.
		if y, ok := y.(*Interface); ok {
			// If identical0 is called (indirectly) via an external API entry point
			// (such as Identical, IdenticalIgnoreTags, etc.), check is nil. But in
			// that case, interfaces are expected to be complete and lazy completion
			// here is not needed.
			if check != nil {
				check.completeInterface(token.NoPos, x)
				check.completeInterface(token.NoPos, y)
			}
			a := x.allMethods
			b := y.allMethods
			if len(a) == len(b) {
				// Interface types are the only types where cycles can occur
				// that are not "terminated" via named types; and such cycles
				// can only be created via method parameter types that are
				// anonymous interfaces (directly or indirectly) embedding
				// the current interface. Example:
				//
				//    type T interface {
				//        m() interface{T}
				//    }
				//
				// If two such (differently named) interfaces are compared,
				// endless recursion occurs if the cycle is not detected.
				//
				// If x and y were compared before, they must be equal
				// (if they were not, the recursion would have stopped);
				// search the ifacePair stack for the same pair.
				//
				// This is a quadratic algorithm, but in practice these stacks
				// are extremely short (bounded by the nesting depth of interface
				// type declarations that recur via parameter types, an extremely
				// rare occurrence). An alternative implementation might use a
				// "visited" map, but that is probably less efficient overall.
				q := &ifacePair{x, y, p}
				for p != nil {
					if p.identical(q) {
						return true // same pair was compared before
					}
					p = p.prev
				}
				if debug {
					assert(sort.IsSorted(byUniqueMethodName(a)))
					assert(sort.IsSorted(byUniqueMethodName(b)))
				}
				for i, f := range a {
					g := b[i]
					if f.Id() != g.Id() || !check.identical0(f.typ, g.typ, cmpTags, q, tparams) {
						return false
					}
				}
				return true
			}
		}

	case *Map:
		// Two map types are identical if they have identical key and value types.
		if y, ok := y.(*Map); ok {
			return check.identical0(x.key, y.key, cmpTags, p, tparams) && check.identical0(x.elem, y.elem, cmpTags, p, tparams)
		}

	case *Chan:
		// Two channel types are identical if they have identical value types
		// and the same direction. For type inference, channel direction is ignored.
		if y, ok := y.(*Chan); ok {
			return (tparams != nil || x.dir == y.dir) && check.identical0(x.elem, y.elem, cmpTags, p, tparams)
		}

	case *Named:
		// Two named types are identical if their type names originate
		// in the same type declaration.
		// if y, ok := y.(*Named); ok {
		// 	return x.obj == y.obj
		// }
		if y, ok := y.(*Named); ok {
			// Without type inference, type parameters (if any) must match
			// exactly and thus the type names must match exactly as well.
			if tparams == nil {
				// TODO(gri) Why is x == y not sufficient? And if it is,
				//           we can just return false here because x == y
				//           is caught in the very beginning of this function.
				return x.obj == y.obj
			}

			// TODO(gri) This is not always correct: two types may have the same names
			//           in the same package if one of them is nested in a function.
			//           Extremely unlikely but we need an always correct solution.
			if x.obj.pkg == y.obj.pkg && stripArgNames(x.obj.name) == stripArgNames(y.obj.name) {
				assert(len(x.targs) == len(y.targs))
				for i, x := range x.targs {
					if !check.identical0(x, y.targs[i], cmpTags, p, tparams) {
						return false
					}
				}
				return true
			}
		}

	case *TypeParam:
		// TODO(gri) do we need to look at type names here?
		// - consider type-checking a generic function calling another generic function
		// - what about self-recursive calls?
		// (may need a map keyed by type parameters with the values the respective inferred types)
		if tparams == nil {
			return false // x and y being equal is caught in the very beginning of this function
		}
		// tparams != nil
		if x := tparams[x.index]; x != nil {
			// If we have inferred a type x and it matches y, we're
			// done. check.identical0 won't do this check if we run
			// in inference mode (tparams != nil), so do it here to
			// avoid endless recursion.
			if x == y {
				return true
			}
			return check.identical0(x, y, cmpTags, p, tparams)
		}
		tparams[x.index] = y // infer type from y
		return true

	case nil:

	default:
		unreachable()
	}

	return false
}

func (check *Checker) identicalTParams(x, y []*TypeName, cmpTags bool, p *ifacePair, tparams []Type) bool {
	if len(x) != len(y) {
		return false
	}
	for i, x := range x {
		y := y[i]
		if !check.identical0(x.typ.(*TypeParam).bound, y.typ.(*TypeParam).bound, cmpTags, p, tparams) {
			return false
		}
	}
	return true
}

// Default returns the default "typed" type for an "untyped" type;
// it returns the incoming type for all other types. The default type
// for untyped nil is untyped nil.
//
func Default(typ Type) Type {
	if t, ok := typ.(*Basic); ok {
		switch t.kind {
		case UntypedBool:
			return Typ[Bool]
		case UntypedInt:
			return Typ[Int]
		case UntypedRune:
			return universeRune // use 'rune' name
		case UntypedFloat:
			return Typ[Float64]
		case UntypedComplex:
			return Typ[Complex128]
		case UntypedString:
			return Typ[String]
		}
	}
	return typ
}
