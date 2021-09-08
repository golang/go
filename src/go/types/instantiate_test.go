// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	. "go/types"
	"testing"
)

func TestInstantiateEquality(t *testing.T) {
	const src = genericPkg + "p; type T[P any] int"

	pkg, err := pkgFor(".", src, nil)
	if err != nil {
		t.Fatal(err)
	}

	T := pkg.Scope().Lookup("T").Type().(*Named)

	// Instantiating the same type twice should result in pointer-equivalent
	// instances.
	env := NewEnvironment()
	res1, err := Instantiate(env, T, []Type{Typ[Int]}, false)
	if err != nil {
		t.Fatal(err)
	}
	res2, err := Instantiate(env, T, []Type{Typ[Int]}, false)
	if err != nil {
		t.Fatal(err)
	}

	if res1 != res2 {
		t.Errorf("first instance (%s) not pointer-equivalent to second instance (%s)", res1, res2)
	}
}

func TestInstantiateNonEquality(t *testing.T) {
	const src = genericPkg + "p; type T[P any] int"

	pkg1, err := pkgFor(".", src, nil)
	if err != nil {
		t.Fatal(err)
	}
	pkg2, err := pkgFor(".", src, nil)
	if err != nil {
		t.Fatal(err)
	}

	// We consider T1 and T2 to be distinct types, so their instances should not
	// be deduplicated by the environment.
	T1 := pkg1.Scope().Lookup("T").Type().(*Named)
	T2 := pkg2.Scope().Lookup("T").Type().(*Named)

	env := NewEnvironment()
	res1, err := Instantiate(env, T1, []Type{Typ[Int]}, false)
	if err != nil {
		t.Fatal(err)
	}
	res2, err := Instantiate(env, T2, []Type{Typ[Int]}, false)
	if err != nil {
		t.Fatal(err)
	}

	if res1 == res2 {
		t.Errorf("instance from pkg1 (%s) is pointer-equivalent to instance from pkg2 (%s)", res1, res2)
	}
	if Identical(res1, res2) {
		t.Errorf("instance from pkg1 (%s) is identical to instance from pkg2 (%s)", res1, res2)
	}
}
