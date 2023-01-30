// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type unification.

package types2

import (
	"bytes"
	"fmt"
	"strings"
)

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

	// If exactUnification is set, unification requires (named) types
	// to match exactly. If it is not set, the underlying types are
	// considered when unification is known to fail otherwise.
	exactUnification = false
)

// A unifier maintains a list of type parameters and
// corresponding types inferred for each type parameter.
// A unifier is created by calling newUnifier.
type unifier struct {
	// tparams is the initial list of type parameters provided.
	// Only used to print/return types in reproducible order.
	tparams []*TypeParam
	// handles maps each type parameter to its inferred type through
	// an indirection *Type called (inferred type) "handle".
	// Initially, each type parameter has its own, separate handle,
	// with a nil (i.e., not yet inferred) type.
	// After a type parameter P is unified with a type parameter Q,
	// P and Q share the same handle (and thus type). This ensures
	// that inferring the type for a given type parameter P will
	// automatically infer the same type for all other parameters
	// unified (joined) with P.
	handles map[*TypeParam]*Type
	depth   int // recursion depth during unification
}

// newUnifier returns a new unifier initialized with the given type parameter
// and corresponding type argument lists. The type argument list may be shorter
// than the type parameter list, and it may contain nil types. Matching type
// parameters and arguments must have the same index.
func newUnifier(tparams []*TypeParam, targs []Type) *unifier {
	assert(len(tparams) >= len(targs))
	handles := make(map[*TypeParam]*Type, len(tparams))
	// Allocate all handles up-front: in a correct program, all type parameters
	// must be resolved and thus eventually will get a handle.
	// Also, sharing of handles caused by unified type parameters is rare and
	// so it's ok to not optimize for that case (and delay handle allocation).
	for i, x := range tparams {
		var t Type
		if i < len(targs) {
			t = targs[i]
		}
		handles[x] = &t
	}
	return &unifier{tparams, handles, 0}
}

// unify attempts to unify x and y and reports whether it succeeded.
// As a side-effect, types may be inferred for type parameters.
func (u *unifier) unify(x, y Type) bool {
	return u.nify(x, y, nil)
}

func (u *unifier) tracef(format string, args ...interface{}) {
	fmt.Println(strings.Repeat(".  ", u.depth) + sprintf(nil, true, format, args...))
}

// String returns a string representation of the current mapping
// from type parameters to types.
func (u *unifier) String() string {
	var buf bytes.Buffer
	w := newTypeWriter(&buf, nil)
	w.byte('[')
	for i, x := range u.tparams {
		if i > 0 {
			w.string(", ")
		}
		w.typ(x)
		w.string(": ")
		w.typ(u.at(x))
	}
	w.byte(']')
	return buf.String()
}

// join unifies the given type parameters x and y.
// If both type parameters already have a type associated with them
// and they are not joined, join fails and returns false.
func (u *unifier) join(x, y *TypeParam) bool {
	if traceInference {
		u.tracef("%s ⇄ %s", x, y)
	}
	switch hx, hy := u.handles[x], u.handles[y]; {
	case hx == hy:
		// Both type parameters already share the same handle. Nothing to do.
	case *hx != nil && *hy != nil:
		// Both type parameters have (possibly different) inferred types. Cannot join.
		return false
	case *hx != nil:
		// Only type parameter x has an inferred type. Use handle of x.
		u.setHandle(y, hx)
	// This case is treated like the default case.
	// case *hy != nil:
	// 	// Only type parameter y has an inferred type. Use handle of y.
	//	u.setHandle(x, hy)
	default:
		// Neither type parameter has an inferred type. Use handle of y.
		u.setHandle(x, hy)
	}
	return true
}

// asTypeParam returns x.(*TypeParam) if x is a type parameter recorded with u.
// Otherwise, the result is nil.
func (u *unifier) asTypeParam(x Type) *TypeParam {
	if x, _ := x.(*TypeParam); x != nil {
		if _, found := u.handles[x]; found {
			return x
		}
	}
	return nil
}

// setHandle sets the handle for type parameter x
// (and all its joined type parameters) to h.
func (u *unifier) setHandle(x *TypeParam, h *Type) {
	hx := u.handles[x]
	assert(hx != nil)
	for y, hy := range u.handles {
		if hy == hx {
			u.handles[y] = h
		}
	}
}

// at returns the (possibly nil) type for type parameter x.
func (u *unifier) at(x *TypeParam) Type {
	return *u.handles[x]
}

// set sets the type t for type parameter x;
// t must not be nil and it must not have been set before.
func (u *unifier) set(x *TypeParam, t Type) {
	assert(t != nil)
	if traceInference {
		u.tracef("%s ➞ %s", x, t)
	}
	h := u.handles[x]
	assert(*h == nil)
	*h = t
}

// unknowns returns the number of type parameters for which no type has been set yet.
func (u *unifier) unknowns() int {
	n := 0
	for _, h := range u.handles {
		if *h == nil {
			n++
		}
	}
	return n
}

// inferred returns the list of inferred types (via unification) for the type parameters
// recorded with u, and an index. If all types were inferred, the returned index is < 0.
// Otherwise, it is the index of the first type parameter which couldn't be inferred;
// i.e., for which list[index] is nil.
func (u *unifier) inferred() (list []Type, index int) {
	list = make([]Type, len(u.tparams))
	index = -1
	for i, x := range u.tparams {
		t := u.at(x)
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

	if !exactUnification {
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

	// Cases where at least one of x or y is a type parameter recorded with u.
	switch px, py := u.asTypeParam(x), u.asTypeParam(y); {
	case px != nil && py != nil:
		// both x and y are type parameters
		if u.join(px, py) {
			return true
		}
		// both x and y have an inferred type - they must match
		return u.nifyEq(u.at(px), u.at(py), p)

	case px != nil:
		// x is a type parameter, y is not
		if tx := u.at(px); tx != nil {
			return u.nifyEq(tx, y, p)
		}
		// otherwise, infer type from y
		u.set(px, y)
		return true

	case py != nil:
		// y is a type parameter, x is not
		if ty := u.at(py); ty != nil {
			return u.nifyEq(x, ty, p)
		}
		// otherwise, infer type from x
		u.set(py, x)
		return true
	}

	// If we get here and x or y is a type parameter, they are type parameters
	// from outside our declaration list. Try to unify their core types, if any
	// (see go.dev/issue/50755 for a test case).
	if enableCoreTypeUnification && !exactUnification {
		if isTypeParam(x) && !hasName(y) {
			// When considering the type parameter for unification
			// we look at the adjusted core term (adjusted core type
			// with tilde information).
			// If the adjusted core type is a named type N; the
			// corresponding core type is under(N). Since !exactUnification
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
			return (!exactUnification || x.dir == y.dir) && u.nify(x.elem, y.elem, p)
		}

	case *Named:
		// TODO(gri) This code differs now from the parallel code in Checker.identical. Investigate.
		if y, ok := y.(*Named); ok {
			xargs := x.TypeArgs().list()
			yargs := y.TypeArgs().list()

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
		panic(sprintf(nil, true, "u.nify(%s, %s), u.tparams = %s", x, y, u.tparams))
	}

	return false
}
