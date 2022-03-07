// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements commonly used type predicates.

package types2

// The isX predicates below report whether t is an X.
// If t is a type parameter the result is false; i.e.,
// these predicates don't look inside a type parameter.

func isBoolean(t Type) bool        { return isBasic(t, IsBoolean) }
func isInteger(t Type) bool        { return isBasic(t, IsInteger) }
func isUnsigned(t Type) bool       { return isBasic(t, IsUnsigned) }
func isFloat(t Type) bool          { return isBasic(t, IsFloat) }
func isComplex(t Type) bool        { return isBasic(t, IsComplex) }
func isNumeric(t Type) bool        { return isBasic(t, IsNumeric) }
func isString(t Type) bool         { return isBasic(t, IsString) }
func isIntegerOrFloat(t Type) bool { return isBasic(t, IsInteger|IsFloat) }
func isConstType(t Type) bool      { return isBasic(t, IsConstType) }

// isBasic reports whether under(t) is a basic type with the specified info.
// If t is a type parameter the result is false; i.e.,
// isBasic does not look inside a type parameter.
func isBasic(t Type, info BasicInfo) bool {
	u, _ := under(t).(*Basic)
	return u != nil && u.info&info != 0
}

// The allX predicates below report whether t is an X.
// If t is a type parameter the result is true if isX is true
// for all specified types of the type parameter's type set.
// allX is an optimized version of isX(coreType(t)) (which
// is the same as underIs(t, isX)).

func allBoolean(t Type) bool         { return allBasic(t, IsBoolean) }
func allInteger(t Type) bool         { return allBasic(t, IsInteger) }
func allUnsigned(t Type) bool        { return allBasic(t, IsUnsigned) }
func allNumeric(t Type) bool         { return allBasic(t, IsNumeric) }
func allString(t Type) bool          { return allBasic(t, IsString) }
func allOrdered(t Type) bool         { return allBasic(t, IsOrdered) }
func allNumericOrString(t Type) bool { return allBasic(t, IsNumeric|IsString) }

// allBasic reports whether under(t) is a basic type with the specified info.
// If t is a type parameter, the result is true if isBasic(t, info) is true
// for all specific types of the type parameter's type set.
// allBasic(t, info) is an optimized version of isBasic(coreType(t), info).
func allBasic(t Type, info BasicInfo) bool {
	if tpar, _ := t.(*TypeParam); tpar != nil {
		return tpar.is(func(t *term) bool { return t != nil && isBasic(t.typ, info) })
	}
	return isBasic(t, info)
}

// hasName reports whether t has a name. This includes
// predeclared types, defined types, and type parameters.
// hasName may be called with types that are not fully set up.
func hasName(t Type) bool {
	switch t.(type) {
	case *Basic, *Named, *TypeParam:
		return true
	}
	return false
}

// isTyped reports whether t is typed; i.e., not an untyped
// constant or boolean. isTyped may be called with types that
// are not fully set up.
func isTyped(t Type) bool {
	// isTyped is called with types that are not fully
	// set up. Must not call under()!
	b, _ := t.(*Basic)
	return b == nil || b.info&IsUntyped == 0
}

// isUntyped(t) is the same as !isTyped(t).
func isUntyped(t Type) bool {
	return !isTyped(t)
}

// IsInterface reports whether t is an interface type.
func IsInterface(t Type) bool {
	_, ok := under(t).(*Interface)
	return ok
}

// isTypeParam reports whether t is a type parameter.
func isTypeParam(t Type) bool {
	_, ok := t.(*TypeParam)
	return ok
}

// isGeneric reports whether a type is a generic, uninstantiated type
// (generic signatures are not included).
// TODO(gri) should we include signatures or assert that they are not present?
func isGeneric(t Type) bool {
	// A parameterized type is only generic if it doesn't have an instantiation already.
	named, _ := t.(*Named)
	return named != nil && named.obj != nil && named.targs == nil && named.TypeParams() != nil
}

// Comparable reports whether values of type T are comparable.
func Comparable(T Type) bool {
	return comparable(T, true, nil, nil)
}

// If dynamic is set, non-type parameter interfaces are always comparable.
// If reportf != nil, it may be used to report why T is not comparable.
func comparable(T Type, dynamic bool, seen map[Type]bool, reportf func(string, ...interface{})) bool {
	if seen[T] {
		return true
	}
	if seen == nil {
		seen = make(map[Type]bool)
	}
	seen[T] = true

	switch t := under(T).(type) {
	case *Basic:
		// assume invalid types to be comparable
		// to avoid follow-up errors
		return t.kind != UntypedNil
	case *Pointer, *Chan:
		return true
	case *Struct:
		for _, f := range t.fields {
			if !comparable(f.typ, dynamic, seen, nil) {
				if reportf != nil {
					reportf("struct containing %s cannot be compared", f.typ)
				}
				return false
			}
		}
		return true
	case *Array:
		if !comparable(t.elem, dynamic, seen, nil) {
			if reportf != nil {
				reportf("%s cannot be compared", t)
			}
			return false
		}
		return true
	case *Interface:
		return dynamic && !isTypeParam(T) || t.typeSet().IsComparable(seen)
	}
	return false
}

// hasNil reports whether type t includes the nil value.
func hasNil(t Type) bool {
	switch u := under(t).(type) {
	case *Basic:
		return u.kind == UnsafePointer
	case *Slice, *Pointer, *Signature, *Map, *Chan:
		return true
	case *Interface:
		return !isTypeParam(t) || u.typeSet().underIs(func(u Type) bool {
			return u != nil && hasNil(u)
		})
	}
	return false
}

// An ifacePair is a node in a stack of interface type pairs compared for identity.
type ifacePair struct {
	x, y *Interface
	prev *ifacePair
}

func (p *ifacePair) identical(q *ifacePair) bool {
	return p.x == q.x && p.y == q.y || p.x == q.y && p.y == q.x
}

// For changes to this code the corresponding changes should be made to unifier.nify.
func identical(x, y Type, cmpTags bool, p *ifacePair) bool {
	if x == y {
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
			return (x.len < 0 || y.len < 0 || x.len == y.len) && identical(x.elem, y.elem, cmpTags, p)
		}

	case *Slice:
		// Two slice types are identical if they have identical element types.
		if y, ok := y.(*Slice); ok {
			return identical(x.elem, y.elem, cmpTags, p)
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
						!identical(f.typ, g.typ, cmpTags, p) {
						return false
					}
				}
				return true
			}
		}

	case *Pointer:
		// Two pointer types are identical if they have identical base types.
		if y, ok := y.(*Pointer); ok {
			return identical(x.base, y.base, cmpTags, p)
		}

	case *Tuple:
		// Two tuples types are identical if they have the same number of elements
		// and corresponding elements have identical types.
		if y, ok := y.(*Tuple); ok {
			if x.Len() == y.Len() {
				if x != nil {
					for i, v := range x.vars {
						w := y.vars[i]
						if !identical(v.typ, w.typ, cmpTags, p) {
							return false
						}
					}
				}
				return true
			}
		}

	case *Signature:
		y, _ := y.(*Signature)
		if y == nil {
			return false
		}

		// Two function types are identical if they have the same number of
		// parameters and result values, corresponding parameter and result types
		// are identical, and either both functions are variadic or neither is.
		// Parameter and result names are not required to match, and type
		// parameters are considered identical modulo renaming.

		if x.TypeParams().Len() != y.TypeParams().Len() {
			return false
		}

		// In the case of generic signatures, we will substitute in yparams and
		// yresults.
		yparams := y.params
		yresults := y.results

		if x.TypeParams().Len() > 0 {
			// We must ignore type parameter names when comparing x and y. The
			// easiest way to do this is to substitute x's type parameters for y's.
			xtparams := x.TypeParams().list()
			ytparams := y.TypeParams().list()

			var targs []Type
			for i := range xtparams {
				targs = append(targs, x.TypeParams().At(i))
			}
			smap := makeSubstMap(ytparams, targs)

			var check *Checker // ok to call subst on a nil *Checker

			// Constraints must be pair-wise identical, after substitution.
			for i, xtparam := range xtparams {
				ybound := check.subst(nopos, ytparams[i].bound, smap, nil)
				if !identical(xtparam.bound, ybound, cmpTags, p) {
					return false
				}
			}

			yparams = check.subst(nopos, y.params, smap, nil).(*Tuple)
			yresults = check.subst(nopos, y.results, smap, nil).(*Tuple)
		}

		return x.variadic == y.variadic &&
			identical(x.params, yparams, cmpTags, p) &&
			identical(x.results, yresults, cmpTags, p)

	case *Union:
		if y, _ := y.(*Union); y != nil {
			// TODO(rfindley): can this be reached during type checking? If so,
			// consider passing a type set map.
			unionSets := make(map[*Union]*_TypeSet)
			xset := computeUnionTypeSet(nil, unionSets, nopos, x)
			yset := computeUnionTypeSet(nil, unionSets, nopos, y)
			return xset.terms.equal(yset.terms)
		}

	case *Interface:
		// Two interface types are identical if they describe the same type sets.
		// With the existing implementation restriction, this simplifies to:
		//
		// Two interface types are identical if they have the same set of methods with
		// the same names and identical function types, and if any type restrictions
		// are the same. Lower-case method names from different packages are always
		// different. The order of the methods is irrelevant.
		if y, ok := y.(*Interface); ok {
			xset := x.typeSet()
			yset := y.typeSet()
			if xset.comparable != yset.comparable {
				return false
			}
			if !xset.terms.equal(yset.terms) {
				return false
			}
			a := xset.methods
			b := yset.methods
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
					assertSortedMethods(a)
					assertSortedMethods(b)
				}
				for i, f := range a {
					g := b[i]
					if f.Id() != g.Id() || !identical(f.typ, g.typ, cmpTags, q) {
						return false
					}
				}
				return true
			}
		}

	case *Map:
		// Two map types are identical if they have identical key and value types.
		if y, ok := y.(*Map); ok {
			return identical(x.key, y.key, cmpTags, p) && identical(x.elem, y.elem, cmpTags, p)
		}

	case *Chan:
		// Two channel types are identical if they have identical value types
		// and the same direction.
		if y, ok := y.(*Chan); ok {
			return x.dir == y.dir && identical(x.elem, y.elem, cmpTags, p)
		}

	case *Named:
		// Two named types are identical if their type names originate
		// in the same type declaration.
		if y, ok := y.(*Named); ok {
			xargs := x.TypeArgs().list()
			yargs := y.TypeArgs().list()

			if len(xargs) != len(yargs) {
				return false
			}

			if len(xargs) > 0 {
				// Instances are identical if their original type and type arguments
				// are identical.
				if !Identical(x.orig, y.orig) {
					return false
				}
				for i, xa := range xargs {
					if !Identical(xa, yargs[i]) {
						return false
					}
				}
				return true
			}

			// TODO(gri) Why is x == y not sufficient? And if it is,
			//           we can just return false here because x == y
			//           is caught in the very beginning of this function.
			return x.obj == y.obj
		}

	case *TypeParam:
		// nothing to do (x and y being equal is caught in the very beginning of this function)

	case nil:
		// avoid a crash in case of nil type

	default:
		unreachable()
	}

	return false
}

// identicalInstance reports if two type instantiations are identical.
// Instantiations are identical if their origin and type arguments are
// identical.
func identicalInstance(xorig Type, xargs []Type, yorig Type, yargs []Type) bool {
	if len(xargs) != len(yargs) {
		return false
	}

	for i, xa := range xargs {
		if !Identical(xa, yargs[i]) {
			return false
		}
	}

	return Identical(xorig, yorig)
}

// Default returns the default "typed" type for an "untyped" type;
// it returns the incoming type for all other types. The default type
// for untyped nil is untyped nil.
func Default(t Type) Type {
	if t, ok := t.(*Basic); ok {
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
	return t
}
