// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type unification.

package types

import (
	"go/token"
	"sort"
)

// A unifier maintains the current type parameters for x and y
// and the respective types inferred for each type parameter.
// A uninifier is created by calling Checker.unifier.
type unifier struct {
	check *Checker
	exact bool
	x, y  typeDesc // x and y must initialized via typeDesc.init
	types []Type   // inferred types, shared by x and y
}

// newUnifier returns a new unifier.
// If exact is set, unification requires unified types to match
// exactly. If exact is not set, a named type's underlying type
// is considered if unification would fail otherwise, and the
// direction of channels is ignored.
func (check *Checker) newUnifier(exact bool) *unifier {
	u := &unifier{check: check, exact: exact}
	u.x.uplink = u
	u.y.uplink = u
	return u
}

// unify attempts to unify x and y and reports whether it succeeded.
func (u *unifier) unify(x, y Type) bool {
	return u.nify(x, y, nil)
}

// A typeDesc describes a list of type parameters and the types inferred for them.
type typeDesc struct {
	uplink  *unifier
	tparams []*TypeName
	indices []int // len(d.indices) == len(d.tparams)
}

func (d *typeDesc) init(tparams []*TypeName) {
	if len(tparams) == 0 {
		return
	}
	d.tparams = tparams
	d.indices = make([]int, len(tparams))
}

// at returns the type inferred (via unification) for the i'th type parameter; or nil.
// The index i must be a valid type parameter index: 0 <= i < len(d.tparams).
func (d *typeDesc) at(i int) Type {
	if i := d.indices[i]; i != 0 {
		typ := d.uplink.types[i-1]
		assert(typ != nil)
		return typ
	}
	return nil
}

// types returns the list of inferred types (via unification) for the type parameters
// described by d, and an index. If all types were inferred, the returned index is < 0.
// Otherwise, it is the index of the first type parameter which couldn't be inferred
// and for which list[index] is nil.
func (d *typeDesc) types() (list []Type, index int) {
	list = make([]Type, len(d.tparams))
	index = -1
	for i := range d.tparams {
		t := d.at(i)
		list[i] = t
		if index < 0 && t == nil {
			index = i
		}
	}
	return
}

// set sets the type typ inferred (via unification) for the i'th type parameter; typ must not be nil.
// The index i must be a valid type parameter index: 0 <= i < len(d.tparams).
func (d *typeDesc) set(i int, typ Type) {
	assert(typ != nil)
	u := d.uplink
	u.types = append(u.types, typ)
	d.indices[i] = len(u.types)
}

// If typ is a type parameter in d.tparams, index returns the
// corresponding d.tparams index. Otherwise, the result is < 0.
func (d *typeDesc) index(typ Type) int {
	if t, ok := typ.(*TypeParam); ok {
		// typ is a type parameter; check that it belongs to the (enclosing) type
		if i := t.index; i < len(d.tparams) && d.tparams[i].typ == t {
			return i
		}
	}
	return -1
}

// nify must only be called by unifier.unify.
// nify implements the core unification algorithm which is an
// adapted version of Checker.identical0. For changes to that
// code the corresponding changes should be made here.
func (u *unifier) nify(x, y Type, p *ifacePair) bool {
	// types must be expanded for comparison
	x = expand(x)
	y = expand(y)

	if !u.exact {
		// If exact unification is known to fail because we attempt to
		// match a type name against an unnamed type literal, consider
		// the underlying type of the named type.
		// (Subtle: We use isNamed to include any type with a name (incl.
		// basic types and type parameters. We use Named() because we only
		// want *Named types.)
		switch {
		case !isNamed(x) && y != nil && y.Named() != nil:
			return u.nify(x, y.Under(), p)
		case x != nil && x.Named() != nil && !isNamed(y):
			return u.nify(x.Under(), y, p)
		}
	}

	//u.check.dump("### u.nify(%s, %s)", x, y)
	i := u.x.index(x)
	j := u.y.index(y)
	switch {
	case i >= 0 && j >= 0:
		//u.check.dump("### i = %d, j = %d", i, j)
		// x and y are type parameters
		// This code is only needed for bidirectional type inference.
		// TODO(gri) We should be able to combine this code with the simple case.
		tx := u.x.at(i)
		ty := u.y.at(j)
		switch {
		case tx != nil && ty != nil:
			// both x and y have an inferred type - they must match
			if tx == ty {
				return true
			}
			return u.nify(tx, ty, p)

		case tx != nil:
			// x has an inferred type
			// TODO(gri) fill this in (only needed for bidirection type inference)
			panic("unimplemented: x has an inferred type")

		case ty != nil:
			// y has an inferred type
			// TODO(gri) fill this in (only needed for bidirection type inference)
			panic("unimplemented: y has an inferred type")

		default:
			// neither x nor y have an inferred type - unify the type parameters
			// TODO(gri) fill this in (only needed for bidirection type inference)
			panic("unimplemented: neither x nor y have an inferred type")
		}

	case i >= 0:
		//u.check.dump("### i = %d", i)
		// x is a type parameter
		if tx := u.x.at(i); tx != nil {
			// If we have inferred a type tx and it matches y, we
			// are done. u.nify won't do this check, so do it now
			// to avoid endless recursion.
			if tx == y {
				return true
			}
			return u.nify(tx, y, p)
		}
		// otherwise, infer type from y (which is known not to be a type parameter)
		u.x.set(i, y)
		return true

	case j >= 0:
		//u.check.dump("### j = %d", j)
		// y is a type parameter
		if ty := u.y.at(j); ty != nil {
			// If we have inferred a type ty and it matches x, we
			// are done. u.nify won't do this check, so do it now
			// to avoid endless recursion.
			if x == ty {
				return true
			}
			return u.nify(x, ty, p)
		}
		// otherwise, infer type from x (which is known not to be a type parameter)
		u.y.set(i, x)
		return true

	}

	// For type unification, do not shortcut (x == y) for identical
	// types. Instead keep comparing them element-wise to unify the
	// matching (and equal type parameter types). A simple test case
	// where this matters is: func f(type T)(x T) { f(x) } .

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
			return (x.len < 0 || y.len < 0 || x.len == y.len) && u.nify(x.elem, y.elem, p)
		}

	case *Slice:
		// Two slice types are identical if they have identical element types.
		if y, ok := y.(*Slice); ok {
			return u.nify(x.elem, y.elem, p)
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
						x.Tag(i) != y.Tag(i) ||
						!f.sameId(g.pkg, g.name) ||
						!u.nify(f.typ, g.typ, p) {
						return false
					}
				}
				return true
			}
		}

	case *Pointer:
		// Two pointer types are identical if they have identical base types.
		if y, ok := y.(*Pointer); ok {
			return u.nify(x.base, y.base, p)
		}

	case *Tuple:
		// Two tuples types are identical if they have the same number of elements
		// and corresponding elements have identical types.
		if y, ok := y.(*Tuple); ok {
			if x.Len() == y.Len() {
				if x != nil {
					for i, v := range x.vars {
						w := y.vars[i]
						if !u.nify(v.typ, w.typ, p) {
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
		// TODO(gri) handle type parameters or document why we can ignore them.
		if y, ok := y.(*Signature); ok {
			return x.variadic == y.variadic &&
				u.nify(x.params, y.params, p) &&
				u.nify(x.results, y.results, p)
		}

	case *Sum:
		// This should not happen with the current internal use of sum types.
		panic("type inference across sum types not implemented")

	case *Interface:
		// Two interface types are identical if they have the same set of methods with
		// the same names and identical function types. Lower-case method names from
		// different packages are always different. The order of the methods is irrelevant.
		if y, ok := y.(*Interface); ok {
			// If identical0 is called (indirectly) via an external API entry point
			// (such as Identical, IdenticalIgnoreTags, etc.), check is nil. But in
			// that case, interfaces are expected to be complete and lazy completion
			// here is not needed.
			if u.check != nil {
				u.check.completeInterface(token.NoPos, x)
				u.check.completeInterface(token.NoPos, y)
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
					if f.Id() != g.Id() || !u.nify(f.typ, g.typ, q) {
						return false
					}
				}
				return true
			}
		}

	case *Map:
		// Two map types are identical if they have identical key and value types.
		if y, ok := y.(*Map); ok {
			return u.nify(x.key, y.key, p) && u.nify(x.elem, y.elem, p)
		}

	case *Chan:
		// Two channel types are identical if they have identical value types.
		if y, ok := y.(*Chan); ok {
			return (!u.exact || x.dir == y.dir) && u.nify(x.elem, y.elem, p)
		}

	case *Named:
		// Two named types are identical if their type names originate
		// in the same type declaration.
		// if y, ok := y.(*Named); ok {
		// 	return x.obj == y.obj
		// }
		if y, ok := y.(*Named); ok {
			// TODO(gri) This is not always correct: two types may have the same names
			//           in the same package if one of them is nested in a function.
			//           Extremely unlikely but we need an always correct solution.
			if x.obj.pkg == y.obj.pkg && x.obj.name == y.obj.name {
				assert(len(x.targs) == len(y.targs))
				for i, x := range x.targs {
					if !u.nify(x, y.targs[i], p) {
						return false
					}
				}
				return true
			}
		}

	case *TypeParam:
		// Two type parameters (which are not part of the type parameters of the
		// enclosing type as those are handled in the beginning of this function)
		// are identical if they originate in the same declaration.
		return x == y

	// case *instance:
	//	unreachable since types are expanded

	case nil:
		// avoid a crash in case of nil type

	default:
		//u.check.dump("### u.nify(%s, %s), u.x.tparams = %s", x, y, u.x.tparams)
		unreachable()
	}

	return false
}
