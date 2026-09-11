// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"simd/archsimd/_gen/specgen"
	"slices"
)

// A result is one figure: how many of how many.
type result struct{ n, d int }

func (r result) miss() int { return r.d - r.n }

// A tally counts distinct keys at one grain, and how many of them are done. A
// key counts as done if any row says so, which is well defined because every
// predicate here is a function of the key rather than of the row.
type tally[K comparable] struct {
	all map[K]bool
	hit map[K]bool
}

func (t *tally[K]) add(k K, done bool) {
	if t.all == nil {
		t.all, t.hit = map[K]bool{}, map[K]bool{}
	}
	t.all[k] = true
	if done {
		t.hit[k] = true
	}
}

func (t *tally[K]) result() result { return result{len(t.hit), len(t.all)} }

// has reports whether the key was counted at all; done, whether it is done.
func (t *tally[K]) has(k K) bool  { return t.all[k] }
func (t *tally[K]) done(k K) bool { return t.hit[k] }

// A pivot is a tally split by the value of some dimension: the report's
// recurring shape is "for each X, what fraction of Y is done".
type pivot[K comparable] struct {
	by map[string]*tally[K]
}

func (p *pivot[K]) add(dim string, k K, done bool) {
	if p.by == nil {
		p.by = map[string]*tally[K]{}
	}
	t := p.by[dim]
	if t == nil {
		t = new(tally[K])
		p.by[dim] = t
	}
	t.add(k, done)
}

// fold merges the tally for one dimension value into another, for reporting a
// roll-up (simdgen's per-target runs) alongside its parts.
func (p *pivot[K]) fold(from, to string) {
	src := p.by[from]
	for k := range src.all {
		p.add(to, k, src.done(k))
	}
}

func (p *pivot[K]) dims() []string { return sortedKeys(p.by) }

func (p *pivot[K]) result(dim string) result {
	if t := p.by[dim]; t != nil {
		return t.result()
	}
	return result{}
}

// tallyBy is the common case: one pass over the fact table, keyed at a grain,
// scored by a predicate.
func tallyBy[K comparable](facts []fact, key func(fact) K, done func(fact) bool) *tally[K] {
	t := new(tally[K])
	for _, f := range facts {
		t.add(key(f), done(f))
	}
	return t
}

// pivotBy is tallyBy, split by a dimension.
func pivotBy[K comparable](facts []fact, dim func(fact) string, key func(fact) K, done func(fact) bool) *pivot[K] {
	p := new(pivot[K])
	for _, f := range facts {
		p.add(dim(f), key(f), done(f))
	}
	return p
}

// countBy counts distinct keys among the rows, for figures that are a plain
// count rather than a fraction.
func countBy[K comparable](facts []fact, key func(fact) K) int {
	seen := map[K]bool{}
	for _, f := range facts {
		seen[key(f)] = true
	}
	return len(seen)
}

func groupBy[T any, K comparable](s []T, key func(T) K) map[K][]T {
	m := map[K][]T{}
	for _, v := range s {
		m[key(v)] = append(m[key(v)], v)
	}
	return m
}

func sortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	slices.Sort(ks)
	return ks
}

// equalBy reports whether two parameter lists agree under f. Types and names
// are compared with the same code because the only difference is the field.
func equalBy(a, b []param, f func(param) string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if f(a[i]) != f(b[i]) {
			return false
		}
	}
	return true
}

func paramType(p param) string { return p.typ }

// specParams converts a spec signature's arguments to the same representation
// as the API's, so one comparison serves both directions.
func specParams(as []specgen.Arg) []param {
	ps := make([]param, len(as))
	for i, a := range as {
		ps[i] = param{a.Name, a.Type.String()}
	}
	return ps
}

// named reports whether p has a name a doc or body could refer to. As in
// specdoc.Fill, the blank identifier does not count.
func named(p param) bool { return p.name != "" && p.name != "_" }

// namesClash reports whether any argument named in both api and spec has
// different names on the two sides. specdoc.Fill reports these as errors. Lists
// of different lengths are a type mismatch, not a name one.
func namesClash(api, spec []param) bool {
	if len(api) != len(spec) {
		return false
	}
	for i := range api {
		if named(api[i]) && named(spec[i]) && api[i].name != spec[i].name {
			return true
		}
	}
	return false
}

// namesMissing reports whether any argument spec names is unnamed in api.
// specdoc.Fill fills these in from spec.
func namesMissing(api, spec []param) bool {
	if len(api) != len(spec) {
		return false
	}
	for i := range api {
		if named(spec[i]) && !named(api[i]) {
			return true
		}
	}
	return false
}

// namesConsistent reports whether argument lists agree on names, where an
// unnamed argument agrees with any name. Lists of different lengths disagree.
func namesConsistent(lists [][]param) bool {
	for _, l := range lists[1:] {
		if len(l) != len(lists[0]) {
			return false
		}
	}
	for i := range lists[0] {
		var name string
		for _, l := range lists {
			if !named(l[i]) {
				continue
			}
			if name == "" {
				name = l[i].name
			} else if l[i].name != name {
				return false
			}
		}
	}
	return true
}
