// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"cmp"
	"slices"
	"strings"
)

func sortedEntries(m Value) (keys, vals []Value) {
	ks, vs := m.MapEntries()
	order := make([]int, len(ks))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int { return CompareKeys(ks[a], ks[b]) })
	keys = make([]Value, len(order))
	vals = make([]Value, len(order))
	for i, j := range order {
		keys[i] = copyValue(ks[j])
		vals[i] = copyValue(vs[j])
	}
	return keys, vals
}

func CompareKeys(a, b Value) int {
	t := a.Type()
	switch {
	case t.Kind == KString:
		return strings.Compare(a.Str(), b.Str())
	case t.Kind == KBool:
		return compareBools(a.Bool(), b.Bool())
	case isFloat(t.Kind):
		return cmp.Compare(keyFloat(a), keyFloat(b))
	case isComplex(t.Kind):
		if t.Kind == KUntypedComplex && a.constant && b.constant {
			ar, ai := complexParts(a)
			br, bi := complexParts(b)
			if c := ar.Cmp(br); c != 0 {
				return c
			}
			return ai.Cmp(bi)
		}
		ca, cb := keyComplex(a), keyComplex(b)
		if c := cmp.Compare(real(ca), real(cb)); c != 0 {
			return c
		}
		return cmp.Compare(imag(ca), imag(cb))
	case isInteger(t.Kind):
		return a.BigInt().Cmp(b.BigInt())
	case t.Kind == KStruct:
		for i, f := range t.Fields {
			if f.Name == "_" {
				continue
			}
			if c := CompareKeys(a.Fields()[i], b.Fields()[i]); c != 0 {
				return c
			}
		}
	case t.Kind == KArray:
		eb := b.Elems()
		for i, e := range a.Elems() {
			if c := CompareKeys(e, eb[i]); c != 0 {
				return c
			}
		}
	}
	return 0
}

func compareBools(a, b bool) int {
	switch {
	case a == b:
		return 0
	case !a:
		return -1
	}
	return 1
}

func keyFloat(v Value) float64 {
	if v.Type().Kind == KFloat32 {
		return float64(float32(v.Float64()))
	}
	return v.Float64()
}

func keyComplex(v Value) complex128 {
	c := v.Complex()
	if v.Type().Kind == KComplex64 {
		return complex(float64(float32(real(c))), float64(float32(imag(c))))
	}
	return c
}
