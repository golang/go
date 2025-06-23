// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements support functions for error messages.

package types2

// lookupError returns a case-specific error when a lookup of selector sel in the
// given type fails but an object with alternative spelling (case folding) is found.
// If structLit is set, the error message is specifically for struct literal fields.
func (check *Checker) lookupError(typ Type, sel string, obj Object, structLit bool) string {
	// Provide more detail if there is an unexported object, or one with different capitalization.
	// If selector and object are in the same package (==), export doesn't matter, otherwise (!=) it does.
	// Messages depend on whether it's a general lookup or a field lookup in a struct literal.
	//
	// case           sel     pkg   have   message (examples for general lookup)
	// ---------------------------------------------------------------------------------------------------------
	// ok             x.Foo   ==    Foo
	// misspelled     x.Foo   ==    FoO    type X has no field or method Foo, but does have field FoO
	// misspelled     x.Foo   ==    foo    type X has no field or method Foo, but does have field foo
	// misspelled     x.Foo   ==    foO    type X has no field or method Foo, but does have field foO
	//
	// misspelled     x.foo   ==    Foo    type X has no field or method foo, but does have field Foo
	// misspelled     x.foo   ==    FoO    type X has no field or method foo, but does have field FoO
	// ok             x.foo   ==    foo
	// misspelled     x.foo   ==    foO    type X has no field or method foo, but does have field foO
	//
	// ok             x.Foo   !=    Foo
	// misspelled     x.Foo   !=    FoO    type X has no field or method Foo, but does have field FoO
	// unexported     x.Foo   !=    foo    type X has no field or method Foo, but does have unexported field foo
	// missing        x.Foo   !=    foO    type X has no field or method Foo
	//
	// misspelled     x.foo   !=    Foo    type X has no field or method foo, but does have field Foo
	// missing        x.foo   !=    FoO    type X has no field or method foo
	// inaccessible   x.foo   !=    foo    cannot refer to unexported field foo
	// missing        x.foo   !=    foO    type X has no field or method foo

	const (
		ok           = iota
		missing      // no object found
		misspelled   // found object with different spelling
		unexported   // found object with name differing only in first letter
		inaccessible // found object with matching name but inaccessible from the current package
	)

	// determine case
	e := missing
	var alt string // alternative spelling of selector; if any
	if obj != nil {
		alt = obj.Name()
		if obj.Pkg() == check.pkg {
			assert(alt != sel) // otherwise there is no lookup error
			e = misspelled
		} else if isExported(sel) {
			if isExported(alt) {
				e = misspelled
			} else if tail(sel) == tail(alt) {
				e = unexported
			}
		} else if isExported(alt) {
			if tail(sel) == tail(alt) {
				e = misspelled
			}
		} else if sel == alt {
			e = inaccessible
		}
	}

	if structLit {
		switch e {
		case missing:
			return check.sprintf("unknown field %s in struct literal of type %s", sel, typ)
		case misspelled:
			return check.sprintf("unknown field %s in struct literal of type %s, but does have %s", sel, typ, alt)
		case unexported:
			return check.sprintf("unknown field %s in struct literal of type %s, but does have unexported %s", sel, typ, alt)
		case inaccessible:
			return check.sprintf("cannot refer to unexported field %s in struct literal of type %s", alt, typ)
		}
	} else {
		what := "object"
		switch obj.(type) {
		case *Var:
			what = "field"
		case *Func:
			what = "method"
		}
		switch e {
		case missing:
			return check.sprintf("type %s has no field or method %s", typ, sel)
		case misspelled:
			return check.sprintf("type %s has no field or method %s, but does have %s %s", typ, sel, what, alt)
		case unexported:
			return check.sprintf("type %s has no field or method %s, but does have unexported %s %s", typ, sel, what, alt)
		case inaccessible:
			return check.sprintf("cannot refer to unexported %s %s", what, alt)
		}
	}

	panic("unreachable")
}

// tail returns the string s without its first (UTF-8) character.
// If len(s) == 0, the result is s.
func tail(s string) string {
	for i, _ := range s {
		if i > 0 {
			return s[i:]
		}
	}
	return s
}
