// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"strings"
	"testing"
)

var testTerms = map[string]*term{
	"∅":       nil,
	"⊤":       {},
	"int":     {false, Typ[Int]},
	"~int":    {true, Typ[Int]},
	"string":  {false, Typ[String]},
	"~string": {true, Typ[String]},
	// TODO(gri) add a defined type
}

func TestTermString(t *testing.T) {
	for want, x := range testTerms {
		if got := x.String(); got != want {
			t.Errorf("%v.String() == %v; want %v", x, got, want)
		}
	}
}

func split(s string, n int) []string {
	r := strings.Split(s, " ")
	if len(r) != n {
		panic("invalid test case: " + s)
	}
	return r
}

func testTerm(name string) *term {
	r, ok := testTerms[name]
	if !ok {
		panic("invalid test argument: " + name)
	}
	return r
}

func TestTermEqual(t *testing.T) {
	for _, test := range []string{
		"∅ ∅ T",
		"⊤ ⊤ T",
		"int int T",
		"~int ~int T",
		"∅ ⊤ F",
		"∅ int F",
		"∅ ~int F",
		"⊤ int F",
		"⊤ ~int F",
		"int ~int F",
	} {
		args := split(test, 3)
		x := testTerm(args[0])
		y := testTerm(args[1])
		want := args[2] == "T"
		if got := x.equal(y); got != want {
			t.Errorf("%v.equal(%v) = %v; want %v", x, y, got, want)
		}
		// equal is symmetric
		x, y = y, x
		if got := x.equal(y); got != want {
			t.Errorf("%v.equal(%v) = %v; want %v", x, y, got, want)
		}
	}
}

func TestTermUnion(t *testing.T) {
	for _, test := range []string{
		"∅ ∅ ∅ ∅",
		"∅ ⊤ ⊤ ∅",
		"∅ int int ∅",
		"∅ ~int ~int ∅",
		"⊤ ⊤ ⊤ ∅",
		"⊤ int ⊤ ∅",
		"⊤ ~int ⊤ ∅",
		"int int int ∅",
		"int ~int ~int ∅",
		"int string int string",
		"int ~string int ~string",
		"~int ~string ~int ~string",

		// union is symmetric, but the result order isn't - repeat symmetric cases explictly
		"⊤ ∅ ⊤ ∅",
		"int ∅ int ∅",
		"~int ∅ ~int ∅",
		"int ⊤ ⊤ ∅",
		"~int ⊤ ⊤ ∅",
		"~int int ~int ∅",
		"string int string int",
		"~string int ~string int",
		"~string ~int ~string ~int",
	} {
		args := split(test, 4)
		x := testTerm(args[0])
		y := testTerm(args[1])
		want1 := testTerm(args[2])
		want2 := testTerm(args[3])
		if got1, got2 := x.union(y); !got1.equal(want1) || !got2.equal(want2) {
			t.Errorf("%v.union(%v) = %v, %v; want %v, %v", x, y, got1, got2, want1, want2)
		}
	}
}

func TestTermIntersection(t *testing.T) {
	for _, test := range []string{
		"∅ ∅ ∅",
		"∅ ⊤ ∅",
		"∅ int ∅",
		"∅ ~int ∅",
		"⊤ ⊤ ⊤",
		"⊤ int int",
		"⊤ ~int ~int",
		"int int int",
		"int ~int int",
		"int string ∅",
		"int ~string ∅",
		"~int ~string ∅",
	} {
		args := split(test, 3)
		x := testTerm(args[0])
		y := testTerm(args[1])
		want := testTerm(args[2])
		if got := x.intersect(y); !got.equal(want) {
			t.Errorf("%v.intersect(%v) = %v; want %v", x, y, got, want)
		}
		// intersect is symmetric
		x, y = y, x
		if got := x.intersect(y); !got.equal(want) {
			t.Errorf("%v.intersect(%v) = %v; want %v", x, y, got, want)
		}
	}
}

func TestTermIncludes(t *testing.T) {
	for _, test := range []string{
		"∅ int F",
		"⊤ int T",
		"int int T",
		"~int int T",
		"string int F",
		"~string int F",
	} {
		args := split(test, 3)
		x := testTerm(args[0])
		y := testTerm(args[1]).typ
		want := args[2] == "T"
		if got := x.includes(y); got != want {
			t.Errorf("%v.includes(%v) = %v; want %v", x, y, got, want)
		}
	}
}

func TestTermSubsetOf(t *testing.T) {
	for _, test := range []string{
		"∅ ∅ T",
		"⊤ ⊤ T",
		"int int T",
		"~int ~int T",
		"∅ ⊤ T",
		"∅ int T",
		"∅ ~int T",
		"⊤ int F",
		"⊤ ~int F",
		"int ~int T",
	} {
		args := split(test, 3)
		x := testTerm(args[0])
		y := testTerm(args[1])
		want := args[2] == "T"
		if got := x.subsetOf(y); got != want {
			t.Errorf("%v.subsetOf(%v) = %v; want %v", x, y, got, want)
		}
	}
}

func TestTermDisjoint(t *testing.T) {
	for _, test := range []string{
		"int int F",
		"~int ~int F",
		"int ~int F",
		"int string T",
		"int ~string T",
		"~int ~string T",
	} {
		args := split(test, 3)
		x := testTerm(args[0])
		y := testTerm(args[1])
		want := args[2] == "T"
		if got := x.disjoint(y); got != want {
			t.Errorf("%v.disjoint(%v) = %v; want %v", x, y, got, want)
		}
		// disjoint is symmetric
		x, y = y, x
		if got := x.disjoint(y); got != want {
			t.Errorf("%v.disjoint(%v) = %v; want %v", x, y, got, want)
		}
	}
}
