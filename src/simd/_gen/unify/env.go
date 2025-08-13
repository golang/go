// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"fmt"
	"iter"
	"reflect"
	"strings"
)

// An envSet is an immutable set of environments, where each environment is a
// mapping from [ident]s to [Value]s.
//
// To keep this compact, we use an algebraic representation similar to
// relational algebra. The atoms are zero, unit, or a singular binding:
//
// - A singular binding is an environment set consisting of a single environment
// that binds a single ident to a single value.
//
// - Zero is the empty set.
//
// - Unit is an environment set consisting of a single, empty environment (no
// bindings).
//
// From these, we build up more complex sets of environments using sums and
// cross products:
//
// - A sum is simply the union of the two environment sets.
//
// - A cross product is the Cartesian product of the two environment sets,
// followed by combining each pair of environments. Combining simply merges the
// two mappings, but fails if the mappings overlap.
//
// For example, to represent {{x: 1, y: 1}, {x: 2, y: 2}}, we build the two
// environments and sum them:
//
//	({x: 1} ⨯ {y: 1}) + ({x: 2} ⨯ {y: 2})
//
// If we add a third variable z that can be 1 or 2, independent of x and y, we
// get four logical environments:
//
//	{x: 1, y: 1, z: 1}
//	{x: 2, y: 2, z: 1}
//	{x: 1, y: 1, z: 2}
//	{x: 2, y: 2, z: 2}
//
// This could be represented as a sum of all four environments, but because z is
// independent, we can use a more compact representation:
//
//	(({x: 1} ⨯ {y: 1}) + ({x: 2} ⨯ {y: 2})) ⨯ ({z: 1} + {z: 2})
//
// Environment sets obey commutative algebra rules:
//
//	e + 0 = e
//	e ⨯ 0 = 0
//	e ⨯ 1 = e
//	e + f = f + e
//	e ⨯ f = f ⨯ e
type envSet struct {
	root *envExpr
}

type envExpr struct {
	// TODO: A tree-based data structure for this may not be ideal, since it
	// involves a lot of walking to find things and we often have to do deep
	// rewrites anyway for partitioning. Would some flattened array-style
	// representation be better, possibly combined with an index of ident uses?
	// We could even combine that with an immutable array abstraction (ala
	// Clojure) that could enable more efficient construction operations.

	kind envExprKind

	// For envBinding
	id  *ident
	val *Value

	// For sum or product. Len must be >= 2 and none of the elements can have
	// the same kind as this node.
	operands []*envExpr
}

type envExprKind byte

const (
	envZero envExprKind = iota
	envUnit
	envProduct
	envSum
	envBinding
)

var (
	// topEnv is the unit value (multiplicative identity) of a [envSet].
	topEnv = envSet{envExprUnit}
	// bottomEnv is the zero value (additive identity) of a [envSet].
	bottomEnv = envSet{envExprZero}

	envExprZero = &envExpr{kind: envZero}
	envExprUnit = &envExpr{kind: envUnit}
)

// bind binds id to each of vals in e.
//
// Its panics if id is already bound in e.
//
// Environments are typically initially constructed by starting with [topEnv]
// and calling bind one or more times.
func (e envSet) bind(id *ident, vals ...*Value) envSet {
	if e.isEmpty() {
		return bottomEnv
	}

	// TODO: If any of vals are _, should we just drop that val? We're kind of
	// inconsistent about whether an id missing from e means id is invalid or
	// means id is _.

	// Check that id isn't present in e.
	for range e.root.bindings(id) {
		panic("id " + id.name + " already present in environment")
	}

	// Create a sum of all the values.
	bindings := make([]*envExpr, 0, 1)
	for _, val := range vals {
		bindings = append(bindings, &envExpr{kind: envBinding, id: id, val: val})
	}

	// Multiply it in.
	return envSet{newEnvExprProduct(e.root, newEnvExprSum(bindings...))}
}

func (e envSet) isEmpty() bool {
	return e.root.kind == envZero
}

// bindings yields all [envBinding] nodes in e with the given id. If id is nil,
// it yields all binding nodes.
func (e *envExpr) bindings(id *ident) iter.Seq[*envExpr] {
	// This is just a pre-order walk and it happens this is the only thing we
	// need a pre-order walk for.
	return func(yield func(*envExpr) bool) {
		var rec func(e *envExpr) bool
		rec = func(e *envExpr) bool {
			if e.kind == envBinding && (id == nil || e.id == id) {
				if !yield(e) {
					return false
				}
			}
			for _, o := range e.operands {
				if !rec(o) {
					return false
				}
			}
			return true
		}
		rec(e)
	}
}

// newEnvExprProduct constructs a product node from exprs, performing
// simplifications. It does NOT check that bindings are disjoint.
func newEnvExprProduct(exprs ...*envExpr) *envExpr {
	factors := make([]*envExpr, 0, 2)
	for _, expr := range exprs {
		switch expr.kind {
		case envZero:
			return envExprZero
		case envUnit:
			// No effect on product
		case envProduct:
			factors = append(factors, expr.operands...)
		default:
			factors = append(factors, expr)
		}
	}

	if len(factors) == 0 {
		return envExprUnit
	} else if len(factors) == 1 {
		return factors[0]
	}
	return &envExpr{kind: envProduct, operands: factors}
}

// newEnvExprSum constructs a sum node from exprs, performing simplifications.
func newEnvExprSum(exprs ...*envExpr) *envExpr {
	// TODO: If all of envs are products (or bindings), factor any common terms.
	// E.g., x * y + x * z ==> x * (y + z). This is easy to do for binding
	// terms, but harder to do for more general terms.

	var have smallSet[*envExpr]
	terms := make([]*envExpr, 0, 2)
	for _, expr := range exprs {
		switch expr.kind {
		case envZero:
			// No effect on sum
		case envSum:
			for _, expr1 := range expr.operands {
				if have.Add(expr1) {
					terms = append(terms, expr1)
				}
			}
		default:
			if have.Add(expr) {
				terms = append(terms, expr)
			}
		}
	}

	if len(terms) == 0 {
		return envExprZero
	} else if len(terms) == 1 {
		return terms[0]
	}
	return &envExpr{kind: envSum, operands: terms}
}

func crossEnvs(env1, env2 envSet) envSet {
	// Confirm that envs have disjoint idents.
	var ids1 smallSet[*ident]
	for e := range env1.root.bindings(nil) {
		ids1.Add(e.id)
	}
	for e := range env2.root.bindings(nil) {
		if ids1.Has(e.id) {
			panic(fmt.Sprintf("%s bound on both sides of cross-product", e.id.name))
		}
	}

	return envSet{newEnvExprProduct(env1.root, env2.root)}
}

func unionEnvs(envs ...envSet) envSet {
	exprs := make([]*envExpr, len(envs))
	for i := range envs {
		exprs[i] = envs[i].root
	}
	return envSet{newEnvExprSum(exprs...)}
}

// envPartition is a subset of an env where id is bound to value in all
// deterministic environments.
type envPartition struct {
	id    *ident
	value *Value
	env   envSet
}

// partitionBy splits e by distinct bindings of id and removes id from each
// partition.
//
// If there are environments in e where id is not bound, they will not be
// reflected in any partition.
//
// It panics if e is bottom, since attempting to partition an empty environment
// set almost certainly indicates a bug.
func (e envSet) partitionBy(id *ident) []envPartition {
	if e.isEmpty() {
		// We could return zero partitions, but getting here at all almost
		// certainly indicates a bug.
		panic("cannot partition empty environment set")
	}

	// Emit a partition for each value of id.
	var seen smallSet[*Value]
	var parts []envPartition
	for n := range e.root.bindings(id) {
		if !seen.Add(n.val) {
			// Already emitted a partition for this value.
			continue
		}

		parts = append(parts, envPartition{
			id:    id,
			value: n.val,
			env:   envSet{e.root.substitute(id, n.val)},
		})
	}

	return parts
}

// substitute replaces bindings of id to val with 1 and bindings of id to any
// other value with 0 and simplifies the result.
func (e *envExpr) substitute(id *ident, val *Value) *envExpr {
	switch e.kind {
	default:
		panic("bad kind")

	case envZero, envUnit:
		return e

	case envBinding:
		if e.id != id {
			return e
		} else if e.val != val {
			return envExprZero
		} else {
			return envExprUnit
		}

	case envProduct, envSum:
		// Substitute each operand. Sometimes, this won't change anything, so we
		// build the new operands list lazily.
		var nOperands []*envExpr
		for i, op := range e.operands {
			nOp := op.substitute(id, val)
			if nOperands == nil && op != nOp {
				// Operand diverged; initialize nOperands.
				nOperands = make([]*envExpr, 0, len(e.operands))
				nOperands = append(nOperands, e.operands[:i]...)
			}
			if nOperands != nil {
				nOperands = append(nOperands, nOp)
			}
		}
		if nOperands == nil {
			// Nothing changed.
			return e
		}
		if e.kind == envProduct {
			return newEnvExprProduct(nOperands...)
		} else {
			return newEnvExprSum(nOperands...)
		}
	}
}

// A smallSet is a set optimized for stack allocation when small.
type smallSet[T comparable] struct {
	array [32]T
	n     int

	m map[T]struct{}
}

// Has returns whether val is in set.
func (s *smallSet[T]) Has(val T) bool {
	arr := s.array[:s.n]
	for i := range arr {
		if arr[i] == val {
			return true
		}
	}
	_, ok := s.m[val]
	return ok
}

// Add adds val to the set and returns true if it was added (not already
// present).
func (s *smallSet[T]) Add(val T) bool {
	// Test for presence.
	if s.Has(val) {
		return false
	}

	// Add it
	if s.n < len(s.array) {
		s.array[s.n] = val
		s.n++
	} else {
		if s.m == nil {
			s.m = make(map[T]struct{})
		}
		s.m[val] = struct{}{}
	}
	return true
}

type ident struct {
	_    [0]func() // Not comparable (only compare *ident)
	name string
}

type Var struct {
	id *ident
}

func (d Var) Exact() bool {
	// These can't appear in concrete Values.
	panic("Exact called on non-concrete Value")
}

func (d Var) WhyNotExact() string {
	// These can't appear in concrete Values.
	return "WhyNotExact called on non-concrete Value"
}

func (d Var) decode(rv reflect.Value) error {
	return &inexactError{"var", rv.Type().String()}
}

func (d Var) unify(w *Value, e envSet, swap bool, uf *unifier) (Domain, envSet, error) {
	// TODO: Vars from !sums in the input can have a huge number of values.
	// Unifying these could be way more efficient with some indexes over any
	// exact values we can pull out, like Def fields that are exact Strings.
	// Maybe we try to produce an array of yes/no/maybe matches and then we only
	// have to do deeper evaluation of the maybes. We could probably cache this
	// on an envTerm. It may also help to special-case Var/Var unification to
	// pick which one to index versus enumerate.

	if vd, ok := w.Domain.(Var); ok && d.id == vd.id {
		// Unifying $x with $x results in $x. If we descend into this we'll have
		// problems because we strip $x out of the environment to keep ourselves
		// honest and then can't find it on the other side.
		//
		// TODO: I'm not positive this is the right fix.
		return vd, e, nil
	}

	// We need to unify w with the value of d in each possible environment. We
	// can save some work by grouping environments by the value of d, since
	// there will be a lot of redundancy here.
	var nEnvs []envSet
	envParts := e.partitionBy(d.id)
	for i, envPart := range envParts {
		exit := uf.enterVar(d.id, i)
		// Each branch logically gets its own copy of the initial environment
		// (narrowed down to just this binding of the variable), and each branch
		// may result in different changes to that starting environment.
		res, e2, err := w.unify(envPart.value, envPart.env, swap, uf)
		exit.exit()
		if err != nil {
			return nil, envSet{}, err
		}
		if res.Domain == nil {
			// This branch entirely failed to unify, so it's gone.
			continue
		}
		nEnv := e2.bind(d.id, res)
		nEnvs = append(nEnvs, nEnv)
	}

	if len(nEnvs) == 0 {
		// All branches failed
		return nil, bottomEnv, nil
	}

	// The effect of this is entirely captured in the environment. We can return
	// back the same Bind node.
	return d, unionEnvs(nEnvs...), nil
}

// An identPrinter maps [ident]s to unique string names.
type identPrinter struct {
	ids   map[*ident]string
	idGen map[string]int
}

func (p *identPrinter) unique(id *ident) string {
	if p.ids == nil {
		p.ids = make(map[*ident]string)
		p.idGen = make(map[string]int)
	}

	name, ok := p.ids[id]
	if !ok {
		gen := p.idGen[id.name]
		p.idGen[id.name]++
		if gen == 0 {
			name = id.name
		} else {
			name = fmt.Sprintf("%s#%d", id.name, gen)
		}
		p.ids[id] = name
	}

	return name
}

func (p *identPrinter) slice(ids []*ident) string {
	var strs []string
	for _, id := range ids {
		strs = append(strs, p.unique(id))
	}
	return fmt.Sprintf("[%s]", strings.Join(strs, ", "))
}
