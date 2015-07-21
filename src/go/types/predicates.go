// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements commonly used type predicates.

package types

import "sort"

func isNamed(typ Type) bool {
	if _, ok := typ.(*Basic); ok {
		return ok
	}
	_, ok := typ.(*Named)
	return ok
}

func isBoolean(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsBoolean != 0
}

func isInteger(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsInteger != 0
}

func isUnsigned(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsUnsigned != 0
}

func isFloat(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsFloat != 0
}

func isComplex(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsComplex != 0
}

func isNumeric(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsNumeric != 0
}

func isString(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsString != 0
}

func isTyped(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return !ok || t.info&IsUntyped == 0
}

func isUntyped(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsUntyped != 0
}

func isOrdered(typ Type) bool {
	t, ok := typ.Underlying().(*Basic)
	return ok && t.info&IsOrdered != 0
}

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

// Identical reports whether x and y are identical.
func Identical(x, y Type) bool {
	return identical(x, y, nil)
}

// An ifacePair is a node in a stack of interface type pairs compared for identity.
type ifacePair struct {
	x, y *Interface
	prev *ifacePair
}

func (p *ifacePair) identical(q *ifacePair) bool {
	return p.x == q.x && p.y == q.y || p.x == q.y && p.y == q.x
}

func identical(x, y Type, p *ifacePair) bool {
	if x == y {
		return true
	}

	switch x := x.(type) {
	case *Basic:
		// Basic types are singletons except for the rune and byte
		// aliases, thus we cannot solely rely on the x == y check
		// above.
		if y, ok := y.(*Basic); ok {
			return x.kind == y.kind
		}

	case *Array:
		// Two array types are identical if they have identical element types
		// and the same array length.
		if y, ok := y.(*Array); ok {
			return x.len == y.len && identical(x.elem, y.elem, p)
		}

	case *Slice:
		// Two slice types are identical if they have identical element types.
		if y, ok := y.(*Slice); ok {
			return identical(x.elem, y.elem, p)
		}

	case *Struct:
		// Two struct types are identical if they have the same sequence of fields,
		// and if corresponding fields have the same names, and identical types,
		// and identical tags. Two anonymous fields are considered to have the same
		// name. Lower-case field names from different packages are always different.
		if y, ok := y.(*Struct); ok {
			if x.NumFields() == y.NumFields() {
				for i, f := range x.fields {
					g := y.fields[i]
					if f.anonymous != g.anonymous ||
						x.Tag(i) != y.Tag(i) ||
						!f.sameId(g.pkg, g.name) ||
						!identical(f.typ, g.typ, p) {
						return false
					}
				}
				return true
			}
		}

	case *Pointer:
		// Two pointer types are identical if they have identical base types.
		if y, ok := y.(*Pointer); ok {
			return identical(x.base, y.base, p)
		}

	case *Tuple:
		// Two tuples types are identical if they have the same number of elements
		// and corresponding elements have identical types.
		if y, ok := y.(*Tuple); ok {
			if x.Len() == y.Len() {
				if x != nil {
					for i, v := range x.vars {
						w := y.vars[i]
						if !identical(v.typ, w.typ, p) {
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
		if y, ok := y.(*Signature); ok {
			return x.variadic == y.variadic &&
				identical(x.params, y.params, p) &&
				identical(x.results, y.results, p)
		}

	case *Interface:
		// Two interface types are identical if they have the same set of methods with
		// the same names and identical function types. Lower-case method names from
		// different packages are always different. The order of the methods is irrelevant.
		if y, ok := y.(*Interface); ok {
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
					if f.Id() != g.Id() || !identical(f.typ, g.typ, q) {
						return false
					}
				}
				return true
			}
		}

	case *Map:
		// Two map types are identical if they have identical key and value types.
		if y, ok := y.(*Map); ok {
			return identical(x.key, y.key, p) && identical(x.elem, y.elem, p)
		}

	case *Chan:
		// Two channel types are identical if they have identical value types
		// and the same direction.
		if y, ok := y.(*Chan); ok {
			return x.dir == y.dir && identical(x.elem, y.elem, p)
		}

	case *Named:
		// Two named types are identical if their type names originate
		// in the same type declaration.
		if y, ok := y.(*Named); ok {
			return x.obj == y.obj
		}

	default:
		unreachable()
	}

	return false
}

// defaultType returns the default "typed" type for an "untyped" type;
// it returns the incoming type for all other types. The default type
// for untyped nil is untyped nil.
//
func defaultType(typ Type) Type {
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
