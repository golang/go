// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type unification.

package types

import (
	"bytes"
	"fmt"
	"strings"
)

// The unifier maintains two separate sets of type parameters x and y
// which are used to resolve type parameters in the x and y arguments
// provided to the unify call. For unidirectional unification, only
// one of these sets (say x) is provided, and then type parameters are
// only resolved for the x argument passed to unify, not the y argument
// (even if that also contains possibly the same type parameters). This
// is crucial to infer the type parameters of self-recursive calls:
//
//	func f[P any](a P) { f(a) }
//
// For the call f(a) we want to infer that the type argument for P is P.
// During unification, the parameter type P must be resolved to the type
// parameter P ("x" side), but the argument type P must be left alone so
// that unification resolves the type parameter P to P.
//
// For bidirectional unification, both sets are provided. This enables
// unification to go from argument to parameter type and vice versa.
// For constraint type inference, we use bidirectional unification
// where both the x and y type parameters are identical. This is done
// by setting up one of them (using init) and then assigning its value
// to the other.

const (
	// Upper limit for recursion depth. Used to catch infinite recursions
	// due to implementation issues (e.g., see issues #48619, #48656).
	unificationDepthLimit = 50

	// Whether to panic when unificationDepthLimit is reached.
	// If disabled, a recursion depth overflow results in a (quiet)
	// unification failure.
	panicAtUnificationDepthLimit = true

	// If enableCoreTypeUnification is set, unification will consider
	// the core types, if any, of non-local (unbound) type parameters.
	enableCoreTypeUnification = true

	// If traceInference is set, unification will print a trace of its operation.
	// Interpretation of trace:
	//   x ≡ y    attempt to unify types x and y
	//   p ➞ y    type parameter p is set to type y (p is inferred to be y)
	//   p ⇄ q    type parameters p and q match (p is inferred to be q and vice versa)
	//   x ≢ y    types x and y cannot be unified
	//   [p, q, ...] ➞ [x, y, ...]    mapping from type parameters to types
	traceInference = false
)

// A unifier maintains the current type parameters for x and y
// and the respective types inferred for each type parameter.
// A unifier is created by calling newUnifier.
type unifier struct {
	exact bool
	x, y  tparamsList // x and y must initialized via tparamsList.init
	types []Type      // inferred types, shared by x and y
	depth int         // recursion depth during unification
}

// newUnifier returns a new unifier.
// If exact is set, unification requires unified types to match
// exactly. If exact is not set, a named type's underlying type
// is considered if unification would fail otherwise, and the
// direction of channels is ignored.
// TODO(gri) exact is not set anymore by a caller. Consider removing it.
func newUnifier(exact bool) *unifier {
	u := &unifier{exact: exact}
	u.x.unifier = u
	u.y.unifier = u
	return u
}

// unify attempts to unify x and y and reports whether it succeeded.
func (u *unifier) unify(x, y Type) bool {
	return u.nify(x, y, nil)
}

func (u *unifier) tracef(format string, args ...interface{}) {
	fmt.Println(strings.Repeat(".  ", u.depth) + sprintf(nil, nil, true, format, args...))
}

// A tparamsList describes a list of type parameters and the types inferred for them.
type tparamsList struct {
	unifier *unifier
	tparams []*TypeParam
	// For each tparams element, there is a corresponding type slot index in indices.
	// index  < 0: unifier.types[-index-1] == nil
	// index == 0: no type slot allocated yet
	// index  > 0: unifier.types[index-1] == typ
	// Joined tparams elements share the same type slot and thus have the same index.
	// By using a negative index for nil types we don't need to check unifier.types
	// to see if we have a type or not.
	indices []int // len(d.indices) == len(d.tparams)
}

// String returns a string representation for a tparamsList. For debugging.
func (d *tparamsList) String() string {
	var buf bytes.Buffer
	w := newTypeWriter(&buf, nil)
	w.byte('[')
	for i, tpar := range d.tparams {
		if i > 0 {
			w.string(", ")
		}
		w.typ(tpar)
		w.string(": ")
		w.typ(d.at(i))
	}
	w.byte(']')
	return buf.String()
}

// init initializes d with the given type parameters.
// The type parameters must be in the order in which they appear in their declaration
// (this ensures that the tparams indices match the respective type parameter index).
func (d *tparamsList) init(tparams []*TypeParam) {
	if len(tparams) == 0 {
		return
	}
	if debug {
		for i, tpar := range tparams {
			assert(i == tpar.index)
		}
	}
	d.tparams = tparams
	d.indices = make([]int, len(tparams))
}

// join unifies the i'th type parameter of x with the j'th type parameter of y.
// If both type parameters already have a type associated with them and they are
// not joined, join fails and returns false.
func (u *unifier) join(i, j int) bool {
	if traceInference {
		u.tracef("%s ⇄ %s", u.x.tparams[i], u.y.tparams[j])
	}
	ti := u.x.indices[i]
	tj := u.y.indices[j]
	switch {
	case ti == 0 && tj == 0:
		// Neither type parameter has a type slot associated with them.
		// Allocate a new joined nil type slot (negative index).
		u.types = append(u.types, nil)
		u.x.indices[i] = -len(u.types)
		u.y.indices[j] = -len(u.types)
	case ti == 0:
		// The type parameter for x has no type slot yet. Use slot of y.
		u.x.indices[i] = tj
	case tj == 0:
		// The type parameter for y has no type slot yet. Use slot of x.
		u.y.indices[j] = ti

	// Both type parameters have a slot: ti != 0 && tj != 0.
	case ti == tj:
		// Both type parameters already share the same slot. Nothing to do.
		break
	case ti > 0 && tj > 0:
		// Both type parameters have (possibly different) inferred types. Cannot join.
		// TODO(gri) Should we check if types are identical? Investigate.
		return false
	case ti > 0:
		// Only the type parameter for x has an inferred type. Use x slot for y.
		u.y.setIndex(j, ti)
	// This case is handled like the default case.
	// case tj > 0:
	// 	// Only the type parameter for y has an inferred type. Use y slot for x.
	// 	u.x.setIndex(i, tj)
	default:
		// Neither type parameter has an inferred type. Use y slot for x
		// (or x slot for y, it doesn't matter).
		u.x.setIndex(i, tj)
	}
	return true
}

// If typ is a type parameter of d, index returns the type parameter index.
// Otherwise, the result is < 0.
func (d *tparamsList) index(typ Type) int {
	if tpar, ok := typ.(*TypeParam); ok {
		return tparamIndex(d.tparams, tpar)
	}
	return -1
}

// If tpar is a type parameter in list, tparamIndex returns the type parameter index.
// Otherwise, the result is < 0. tpar must not be nil.
func tparamIndex(list []*TypeParam, tpar *TypeParam) int {
	// Once a type parameter is bound its index is >= 0. However, there are some
	// code paths (namely tracing and type hashing) by which it is possible to
	// arrive here with a type parameter that has not been bound, hence the check
	// for 0 <= i below.
	// TODO(rfindley): investigate a better approach for guarding against using
	// unbound type parameters.
	if i := tpar.index; 0 <= i && i < len(list) && list[i] == tpar {
		return i
	}
	return -1
}

// setIndex sets the type slot index for the i'th type parameter
// (and all its joined parameters) to tj. The type parameter
// must have a (possibly nil) type slot associated with it.
func (d *tparamsList) setIndex(i, tj int) {
	ti := d.indices[i]
	assert(ti != 0 && tj != 0)
	for k, tk := range d.indices {
		if tk == ti {
			d.indices[k] = tj
		}
	}
}

// at returns the type set for the i'th type parameter; or nil.
func (d *tparamsList) at(i int) Type {
	if ti := d.indices[i]; ti > 0 {
		return d.unifier.types[ti-1]
	}
	return nil
}

// set sets the type typ for the i'th type parameter;
// typ must not be nil and it must not have been set before.
func (d *tparamsList) set(i int, typ Type) {
	assert(typ != nil)
	u := d.unifier
	if traceInference {
		u.tracef("%s ➞ %s", d.tparams[i], typ)
	}
	switch ti := d.indices[i]; {
	case ti < 0:
		u.types[-ti-1] = typ
		d.setIndex(i, -ti)
	case ti == 0:
		u.types = append(u.types, typ)
		d.indices[i] = len(u.types)
	default:
		panic("type already set")
	}
}

// unknowns returns the number of type parameters for which no type has been set yet.
func (d *tparamsList) unknowns() int {
	n := 0
	for _, ti := range d.indices {
		if ti <= 0 {
			n++
		}
	}
	return n
}

// types returns the list of inferred types (via unification) for the type parameters
// described by d, and an index. If all types were inferred, the returned index is < 0.
// Otherwise, it is the index of the first type parameter which couldn't be inferred;
// i.e., for which list[index] is nil.
func (d *tparamsList) types() (list []Type, index int) {
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

func (u *unifier) nifyEq(x, y Type, p *ifacePair) bool {
	return x == y || u.nify(x, y, p)
}

// nify implements the core unification algorithm which is an
// adapted version of Checker.identical. For changes to that
// code the corresponding changes should be made here.
// Must not be called directly from outside the unifier.
func (u *unifier) nify(x, y Type, p *ifacePair) (result bool) {
	if traceInference {
		u.tracef("%s ≡ %s", x, y)
	}

	// Stop gap for cases where unification fails.
	if u.depth >= unificationDepthLimit {
		if traceInference {
			u.tracef("depth %d >= %d", u.depth, unificationDepthLimit)
		}
		if panicAtUnificationDepthLimit {
			panic("unification reached recursion depth limit")
		}
		return false
	}
	u.depth++
	defer func() {
		u.depth--
		if traceInference && !result {
			u.tracef("%s ≢ %s", x, y)
		}
	}()

	if !u.exact {
		// If exact unification is known to fail because we attempt to
		// match a type name against an unnamed type literal, consider
		// the underlying type of the named type.
		// (We use !hasName to exclude any type with a name, including
		// basic types and type parameters; the rest are unamed types.)
		if nx, _ := x.(*Named); nx != nil && !hasName(y) {
			if traceInference {
				u.tracef("under %s ≡ %s", nx, y)
			}
			return u.nify(nx.under(), y, p)
		} else if ny, _ := y.(*Named); ny != nil && !hasName(x) {
			if traceInference {
				u.tracef("%s ≡ under %s", x, ny)
			}
			return u.nify(x, ny.under(), p)
		}
	}

	// Cases where at least one of x or y is a type parameter.
	switch i, j := u.x.index(x), u.y.index(y); {
	case i >= 0 && j >= 0:
		// both x and y are type parameters
		if u.join(i, j) {
			return true
		}
		// both x and y have an inferred type - they must match
		return u.nifyEq(u.x.at(i), u.y.at(j), p)

	case i >= 0:
		// x is a type parameter, y is not
		if tx := u.x.at(i); tx != nil {
			return u.nifyEq(tx, y, p)
		}
		// otherwise, infer type from y
		u.x.set(i, y)
		return true

	case j >= 0:
		// y is a type parameter, x is not
		if ty := u.y.at(j); ty != nil {
			return u.nifyEq(x, ty, p)
		}
		// otherwise, infer type from x
		u.y.set(j, x)
		return true
	}

	// If we get here and x or y is a type parameter, they are type parameters
	// from outside our declaration list. Try to unify their core types, if any
	// (see issue #50755 for a test case).
	if enableCoreTypeUnification && !u.exact {
		if isTypeParam(x) && !hasName(y) {
			// When considering the type parameter for unification
			// we look at the adjusted core term (adjusted core type
			// with tilde information).
			// If the adjusted core type is a named type N; the
			// corresponding core type is under(N). Since !u.exact
			// and y doesn't have a name, unification will end up
			// comparing under(N) to y, so we can just use the core
			// type instead. And we can ignore the tilde because we
			// already look at the underlying types on both sides
			// and we have known types on both sides.
			// Optimization.
			if cx := coreType(x); cx != nil {
				if traceInference {
					u.tracef("core %s ≡ %s", x, y)
				}
				return u.nify(cx, y, p)
			}
		} else if isTypeParam(y) && !hasName(x) {
			// see comment above
			if cy := coreType(y); cy != nil {
				if traceInference {
					u.tracef("%s ≡ core %s", x, y)
				}
				return u.nify(x, cy, p)
			}
		}
	}

	// For type unification, do not shortcut (x == y) for identical
	// types. Instead keep comparing them element-wise to unify the
	// matching (and equal type parameter types). A simple test case
	// where this matters is: func f[P any](a P) { f(a) } .

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

	case *Interface:
		// Two interface types are identical if they have the same set of methods with
		// the same names and identical function types. Lower-case method names from
		// different packages are always different. The order of the methods is irrelevant.
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
		// TODO(gri) This code differs now from the parallel code in Checker.identical. Investigate.
		if y, ok := y.(*Named); ok {
			xargs := x.targs.list()
			yargs := y.targs.list()

			if len(xargs) != len(yargs) {
				return false
			}

			// TODO(gri) This is not always correct: two types may have the same names
			//           in the same package if one of them is nested in a function.
			//           Extremely unlikely but we need an always correct solution.
			if x.obj.pkg == y.obj.pkg && x.obj.name == y.obj.name {
				for i, x := range xargs {
					if !u.nify(x, yargs[i], p) {
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

	case nil:
		// avoid a crash in case of nil type

	default:
		panic(sprintf(nil, nil, true, "u.nify(%s, %s), u.x.tparams = %s", x, y, u.x.tparams))
	}

	return false
}
