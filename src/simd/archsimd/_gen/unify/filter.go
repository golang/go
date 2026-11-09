// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

// A valueFilter quickly rules out values that cannot unify with a given value,
// without unifying them.
//
// Unifying a variable with a value unifies each of the variable's values with
// that value. When the variable comes from a large !sum, few of these
// typically succeed, and most fail because of a String field that doesn't
// match. Unifying each value is expensive, since it partitions the environment
// set, but comparing these String fields is cheap. So we perform this small,
// heuristic piece of unification first, and throw out most of the values
// before we do the full unification on the rest.
//
// As a heuristic, the filter only compares the String fields of Defs, since
// that's a very common way for values to differ in practice. It doesn't rule
// out anything else.
type valueFilter struct {
	def Def // The value to filter against
}

// newValueFilter returns a filter for values that may unify with w, or nil if
// it can't filter anything.
func newValueFilter(w *Value) *valueFilter {
	if debugNoFastPaths {
		return nil
	}
	d, ok := w.Domain.(Def)
	if !ok || len(d.fields) == 0 {
		// We only filter against Defs with fields.
		return nil
	}
	return &valueFilter{def: d}
}

// mayUnify returns false if v definitely does not unify with the value f
// filters against. If f is nil, it always returns true.
//
// It returns false only if unifying the two Defs would visit a pair of String
// fields that don't match before visiting any other fields that could report an
// error. Hence, if mayUnify returns false, unification would fail without an
// error.
func (f *valueFilter) mayUnify(v *Value) bool {
	if f == nil {
		return true
	}
	d, ok := v.Domain.(Def)
	if !ok {
		return true
	}
	// Visit the keys in the same order as Def.unify.
	for _, key := range f.def.keys {
		dv, ok := d.fields[key]
		if !ok {
			continue
		}
		fv := f.def.fields[key]
		_, dTop := dv.Domain.(Top)
		_, fTop := fv.Domain.(Top)
		if dTop || fTop {
			// Unifying with Top always succeeds.
			continue
		}
		ds, ok1 := dv.Domain.(String)
		fs, ok2 := fv.Domain.(String)
		if !ok1 || !ok2 {
			// Unifying these fields could report an error, which we
			// mustn't hide.
			return true
		}
		if ds.kind == stringRegex && fs.kind == stringRegex {
			// String.unify doesn't check if regexps intersect.
			continue
		}
		// Both fields are Strings, so we can unify just these two fields
		// cheaply, without touching the environment. If they don't match,
		// neither do the Defs.
		if ds.unify(fs) == nil {
			return false
		}
	}
	return true
}
