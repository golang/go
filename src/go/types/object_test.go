// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "testing"

func TestIsAlias(t *testing.T) {
	check := func(obj *TypeName, want bool) {
		if got := obj.IsAlias(); got != want {
			t.Errorf("%v: got IsAlias = %v; want %v", obj, got, want)
		}
	}

	// predeclared types
	for _, name := range Universe.Names() {
		if obj, _ := Universe.Lookup(name).(*TypeName); obj != nil {
			check(obj, name == "byte" || name == "rune")
		}
	}

	// various other types
	t0 := NewTypeName(0, nil, "t0", nil)
	check(t0, false) // no type yet

	t1 := NewTypeName(0, nil, "t1", nil)
	n1 := NewNamed(t1, new(Struct), nil)
	check(t1, false) // type name refers to named type and vice versa

	t2 := NewTypeName(0, nil, "t2", new(Interface))
	check(t2, true) // type name refers to unnamed type

	t3 := NewTypeName(0, nil, "t3", n1)
	check(t3, true) // type name refers to named type with different type name (true alias)
}
