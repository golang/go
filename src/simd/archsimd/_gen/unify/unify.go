// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unify implements unification of structured values.
//
// A [Value] represents a possibly infinite set of concrete values, where a
// value is either a string ([String]), a tuple of values ([Tuple]), or a
// string-keyed map of values called a "def" ([Def]). These sets can be further
// constrained by variables ([Var]). A [Value] combined with bindings of
// variables is a [Closure].
//
// [Unify] finds a [Closure] that satisfies two or more other [Closure]s. This
// can be thought of as intersecting the sets represented by these Closures'
// values, or as the greatest lower bound/infimum of these Closures. If no such
// Closure exists, the result of unification is "bottom", or the empty set.
//
// # Examples
//
// The regular expression "a*" is the infinite set of strings of zero or more
// "a"s. "a*" can be unified with "a" or "aa" or "aaa", and the result is just
// "a", "aa", or "aaa", respectively. However, unifying "a*" with "b" fails
// because there are no values that satisfy both.
//
// Sums express sets directly. For example, !sum [a, b] is the set consisting of
// "a" and "b". Unifying this with !sum [b, c] results in just "b". This also
// makes it easy to demonstrate that unification isn't necessarily a single
// concrete value. For example, unifying !sum [a, b, c] with !sum [b, c, d]
// results in two concrete values: "b" and "c".
//
// The special value _ or "top" represents all possible values. Unifying _ with
// any value x results in x.
//
// Unifying composite values—tuples and defs—unifies their elements.
//
// The value [a*, aa] is an infinite set of tuples. If we unify that with the
// value [aaa, a*], the only possible value that satisfies both is [aaa, aa].
// Likewise, this is the intersection of the sets described by these two values.
//
// Defs are similar to tuples, but they are indexed by strings and don't have a
// fixed length. For example, {x: a, y: b} is a def with two fields. Any field
// not mentioned in a def is implicitly top. Thus, unifying this with {y: b, z:
// c} results in {x: a, y: b, z: c}.
//
// Variables constrain values. For example, the value [$x, $x] represents all
// tuples whose first and second values are the same, but doesn't otherwise
// constrain that value. Thus, this set includes [a, a] as well as [[b, c, d],
// [b, c, d]], but it doesn't include [a, b].
//
// Sums are internally implemented as fresh variables that are simultaneously
// bound to all values of the sum. That is !sum [a, b] is actually $var (where
// var is some fresh name), closed under the environment $var=a | $var=b.
package unify

import (
	"errors"
	"fmt"
	"slices"
)

// Unify computes a Closure that satisfies each input Closure. If no such
// Closure exists, it returns bottom.
func Unify(closures ...Closure) (Closure, error) {
	if len(closures) == 0 {
		return Closure{topValue, topEnv}, nil
	}

	var trace *tracer
	if Debug.UnifyLog != nil || Debug.HTML != nil {
		trace = &tracer{
			logw:     Debug.UnifyLog,
			saveTree: Debug.HTML != nil,
		}
	}

	unified := closures[0]
	for _, c := range closures[1:] {
		var err error
		uf := newUnifier()
		uf.tracer = trace
		e := crossEnvs(unified.env, c.env)
		unified.val, unified.env, err = unified.val.unify(c.val, e, false, uf)
		if Debug.HTML != nil {
			uf.writeHTML(Debug.HTML)
		}
		if err != nil {
			return Closure{}, err
		}
	}

	return unified, nil
}

type unifier struct {
	*tracer
}

func newUnifier() *unifier {
	return &unifier{}
}

// errDomains is a sentinel error used between unify and unify1 to indicate that
// unify1 could not unify the domains of the two values.
var errDomains = errors.New("cannot unify domains")

func (v *Value) unify(w *Value, e envSet, swap bool, uf *unifier) (*Value, envSet, error) {
	if swap {
		// Put the values in order. This just happens to be a handy choke-point
		// to do this at.
		v, w = w, v
	}

	uf.traceUnify(v, w, e)

	d, e2, err := v.unify1(w, e, false, uf)
	if err == errDomains {
		// Try the other order.
		d, e2, err = w.unify1(v, e, true, uf)
		if err == errDomains {
			// Okay, we really can't unify these.
			err = fmt.Errorf("cannot unify %T (%s) and %T (%s): kind mismatch", v.Domain, v.PosString(), w.Domain, w.PosString())
		}
	}
	if err != nil {
		uf.traceDone(nil, envSet{}, err)
		return nil, envSet{}, err
	}
	res := unified(d, v, w)
	uf.traceDone(res, e2, nil)
	if d == nil {
		// Double check that a bottom Value also has a bottom env.
		if !e2.isEmpty() {
			panic("bottom Value has non-bottom environment")
		}
	}

	return res, e2, nil
}

func (v *Value) unify1(w *Value, e envSet, swap bool, uf *unifier) (Domain, envSet, error) {
	// TODO: If there's an error, attach position information to it.

	vd, wd := v.Domain, w.Domain

	// Bottom returns bottom, and eliminates all possible environments.
	if vd == nil || wd == nil {
		return nil, bottomEnv, nil
	}

	// Top always returns the other.
	if _, ok := vd.(Top); ok {
		return wd, e, nil
	}

	// Variables
	if vd, ok := vd.(Var); ok {
		return vd.unify(w, e, swap, uf)
	}

	// Composite values
	if vd, ok := vd.(Def); ok {
		if wd, ok := wd.(Def); ok {
			return vd.unify(wd, e, swap, uf)
		}
	}
	if vd, ok := vd.(Tuple); ok {
		if wd, ok := wd.(Tuple); ok {
			return vd.unify(wd, e, swap, uf)
		}
	}

	// Scalar values
	if vd, ok := vd.(String); ok {
		if wd, ok := wd.(String); ok {
			res := vd.unify(wd)
			if res == nil {
				e = bottomEnv
			}
			return res, e, nil
		}
	}

	return nil, envSet{}, errDomains
}

func (d Def) unify(o Def, e envSet, swap bool, uf *unifier) (Domain, envSet, error) {
	out := Def{fields: make(map[string]*Value)}

	// Check keys of d against o.
	for key, dv := range d.All() {
		ov, ok := o.fields[key]
		if !ok {
			// ov is implicitly Top. Bypass unification.
			out.fields[key] = dv
			continue
		}
		exit := uf.enter("%s", key)
		res, e2, err := dv.unify(ov, e, swap, uf)
		exit.exit()
		if err != nil {
			return nil, envSet{}, err
		} else if res.Domain == nil {
			// No match.
			return nil, bottomEnv, nil
		}
		out.fields[key] = res
		e = e2
	}
	// Check keys of o that we didn't already check. These all implicitly match
	// because we know the corresponding fields in d are all Top.
	for key, dv := range o.All() {
		if _, ok := d.fields[key]; !ok {
			out.fields[key] = dv
		}
	}
	return out, e, nil
}

func (v Tuple) unify(w Tuple, e envSet, swap bool, uf *unifier) (Domain, envSet, error) {
	if v.repeat != nil && w.repeat != nil {
		// Since we generate the content of these lazily, there's not much we
		// can do but just stick them on a list to unify later.
		return Tuple{repeat: concat(v.repeat, w.repeat)}, e, nil
	}

	// Expand any repeated tuples.
	tuples := make([]Tuple, 0, 2)
	if v.repeat == nil {
		tuples = append(tuples, v)
	} else {
		v2, e2 := v.doRepeat(e, len(w.vs))
		tuples = append(tuples, v2...)
		e = e2
	}
	if w.repeat == nil {
		tuples = append(tuples, w)
	} else {
		w2, e2 := w.doRepeat(e, len(v.vs))
		tuples = append(tuples, w2...)
		e = e2
	}

	// Now unify all of the tuples (usually this will be just 2 tuples)
	out := tuples[0]
	for _, t := range tuples[1:] {
		if len(out.vs) != len(t.vs) {
			uf.logf("tuple length mismatch")
			return nil, bottomEnv, nil
		}
		zs := make([]*Value, len(out.vs))
		for i, v1 := range out.vs {
			exit := uf.enter("%d", i)
			z, e2, err := v1.unify(t.vs[i], e, swap, uf)
			exit.exit()
			if err != nil {
				return nil, envSet{}, err
			} else if z.Domain == nil {
				return nil, bottomEnv, nil
			}
			zs[i] = z
			e = e2
		}
		out = Tuple{vs: zs}
	}

	return out, e, nil
}

// doRepeat creates a fixed-length tuple from a repeated tuple. The caller is
// expected to unify the returned tuples.
func (v Tuple) doRepeat(e envSet, n int) ([]Tuple, envSet) {
	res := make([]Tuple, len(v.repeat))
	for i, gen := range v.repeat {
		res[i].vs = make([]*Value, n)
		for j := range n {
			res[i].vs[j], e = gen(e)
		}
	}
	return res, e
}

// unify intersects the domains of two [String]s. If it can prove that this
// domain is empty, it returns nil (bottom).
//
// TODO: Consider splitting literals and regexps into two domains.
func (v String) unify(w String) Domain {
	// Unification is symmetric, so put them in order of string kind so we only
	// have to deal with half the cases.
	if v.kind > w.kind {
		v, w = w, v
	}

	switch v.kind {
	case stringRegex:
		switch w.kind {
		case stringRegex:
			// Construct a match against all of the regexps
			return String{kind: stringRegex, re: slices.Concat(v.re, w.re)}
		case stringExact:
			for _, re := range v.re {
				if !re.MatchString(w.exact) {
					return nil
				}
			}
			return w
		}
	case stringExact:
		if v.exact != w.exact {
			return nil
		}
		return v
	}
	panic("bad string kind")
}

func concat[T any](s1, s2 []T) []T {
	// Reuse s1 or s2 if possible.
	if len(s1) == 0 {
		return s2
	}
	return append(s1[:len(s1):len(s1)], s2...)
}
