// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"fmt"
	"iter"
	"maps"
	"slices"
)

type Closure struct {
	val *Value
	env envSet
}

func NewSum(vs ...*Value) Closure {
	id := &ident{name: "sum"}
	return Closure{NewValue(Var{id}), topEnv.bind(id, vs...)}
}

// IsBottom returns whether c consists of no values.
func (c Closure) IsBottom() bool {
	return c.val.Domain == nil
}

// Summands returns the top-level Values of c. This assumes the top-level of c
// was constructed as a sum, and is mostly useful for debugging.
func (c Closure) Summands() iter.Seq[*Value] {
	return func(yield func(*Value) bool) {
		var rec func(v *Value, env envSet) bool
		rec = func(v *Value, env envSet) bool {
			switch d := v.Domain.(type) {
			case Var:
				parts := env.partitionBy(d.id)
				for _, part := range parts {
					// It may be a sum of sums. Walk into this value.
					if !rec(part.value, part.env) {
						return false
					}
				}
				return true
			default:
				return yield(v)
			}
		}
		rec(c.val, c.env)
	}
}

// All enumerates all possible concrete values of c by substituting variables
// from the environment.
//
// E.g., enumerating this Value
//
//	a: !sum [1, 2]
//	b: !sum [3, 4]
//
// results in
//
//   - {a: 1, b: 3}
//   - {a: 1, b: 4}
//   - {a: 2, b: 3}
//   - {a: 2, b: 4}
func (c Closure) All() iter.Seq[*Value] {
	// In order to enumerate all concrete values under all possible variable
	// bindings, we use a "non-deterministic continuation passing style" to
	// implement this. We use CPS to traverse the Value tree, threading the
	// (possibly narrowing) environment through that CPS following an Euler
	// tour. Where the environment permits multiple choices, we invoke the same
	// continuation for each choice. Similar to a yield function, the
	// continuation can return false to stop the non-deterministic walk.
	return func(yield func(*Value) bool) {
		c.val.all1(c.env, func(v *Value, e envSet) bool {
			return yield(v)
		})
	}
}

func (v *Value) all1(e envSet, cont func(*Value, envSet) bool) bool {
	switch d := v.Domain.(type) {
	default:
		panic(fmt.Sprintf("unknown domain type %T", d))

	case nil:
		return true

	case Top, String:
		return cont(v, e)

	case Def:
		fields := d.keys()
		// We can reuse this parts slice because we're doing a DFS through the
		// state space. (Otherwise, we'd have to do some messy threading of an
		// immutable slice-like value through allElt.)
		parts := make(map[string]*Value, len(fields))

		// TODO: If there are no Vars or Sums under this Def, then nothing can
		// change the Value or env, so we could just cont(v, e).
		var allElt func(elt int, e envSet) bool
		allElt = func(elt int, e envSet) bool {
			if elt == len(fields) {
				// Build a new Def from the concrete parts. Clone parts because
				// we may reuse it on other non-deterministic branches.
				nVal := newValueFrom(Def{maps.Clone(parts)}, v)
				return cont(nVal, e)
			}

			return d.fields[fields[elt]].all1(e, func(v *Value, e envSet) bool {
				parts[fields[elt]] = v
				return allElt(elt+1, e)
			})
		}
		return allElt(0, e)

	case Tuple:
		// Essentially the same as Def.
		if d.repeat != nil {
			// There's nothing we can do with this.
			return cont(v, e)
		}
		parts := make([]*Value, len(d.vs))
		var allElt func(elt int, e envSet) bool
		allElt = func(elt int, e envSet) bool {
			if elt == len(d.vs) {
				// Build a new tuple from the concrete parts. Clone parts because
				// we may reuse it on other non-deterministic branches.
				nVal := newValueFrom(Tuple{vs: slices.Clone(parts)}, v)
				return cont(nVal, e)
			}

			return d.vs[elt].all1(e, func(v *Value, e envSet) bool {
				parts[elt] = v
				return allElt(elt+1, e)
			})
		}
		return allElt(0, e)

	case Var:
		// Go each way this variable can be bound.
		for _, ePart := range e.partitionBy(d.id) {
			// d.id is no longer bound in this environment partition. We'll may
			// need it later in the Euler tour, so bind it back to this single
			// value.
			env := ePart.env.bind(d.id, ePart.value)
			if !ePart.value.all1(env, cont) {
				return false
			}
		}
		return true
	}
}
