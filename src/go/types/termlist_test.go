// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"strings"
	"testing"
)

// maketl makes a term list from a string of the term list.
func maketl(s string) termlist {
	s = strings.Replace(s, " ", "", -1)
	names := strings.Split(s, "∪")
	r := make(termlist, len(names))
	for i, n := range names {
		r[i] = testTerm(n)
	}
	return r
}

func TestTermlistTop(t *testing.T) {
	if !topTermlist.isTop() {
		t.Errorf("topTermlist is not top")
	}
}

func TestTermlistString(t *testing.T) {
	for _, want := range []string{
		"∅",
		"⊤",
		"int",
		"~int",
		"∅ ∪ ∅",
		"⊤ ∪ ⊤",
		"∅ ∪ ⊤ ∪ int",
	} {
		if got := maketl(want).String(); got != want {
			t.Errorf("(%v).String() == %v", want, got)
		}
	}
}

func TestTermlistIsEmpty(t *testing.T) {
	for test, want := range map[string]bool{
		"∅":         true,
		"∅ ∪ ∅":     true,
		"∅ ∪ ∅ ∪ ⊤": false,
		"⊤":         false,
		"⊤ ∪ int":   false,
	} {
		xl := maketl(test)
		got := xl.isEmpty()
		if got != want {
			t.Errorf("(%v).isEmpty() == %v; want %v", test, got, want)
		}
	}
}

func TestTermlistIsTop(t *testing.T) {
	for test, want := range map[string]bool{
		"∅":             false,
		"∅ ∪ ∅":         false,
		"int ∪ ~string": false,
		"∅ ∪ ∅ ∪ ⊤":     true,
		"⊤":             true,
		"⊤ ∪ int":       true,
	} {
		xl := maketl(test)
		got := xl.isTop()
		if got != want {
			t.Errorf("(%v).isTop() == %v; want %v", test, got, want)
		}
	}
}

func TestTermlistNorm(t *testing.T) {
	for _, test := range []struct {
		xl, want string
	}{
		{"∅", "∅"},
		{"∅ ∪ ∅", "∅"},
		{"∅ ∪ int", "int"},
		{"⊤ ∪ int", "⊤"},
		{"~int ∪ int", "~int"},
		{"int ∪ ~string ∪ int", "int ∪ ~string"},
		{"~int ∪ string ∪ ⊤ ∪ ~string ∪ int", "⊤"},
	} {
		xl := maketl(test.xl)
		got := maketl(test.xl).norm()
		if got.String() != test.want {
			t.Errorf("(%v).norm() = %v; want %v", xl, got, test.want)
		}
	}
}

func TestTermlistStructuralType(t *testing.T) {
	// helper to deal with nil types
	tstring := func(typ Type) string {
		if typ == nil {
			return "nil"
		}
		return typ.String()
	}

	for test, want := range map[string]string{
		"∅":                 "nil",
		"⊤":                 "nil",
		"int":               "int",
		"~int":              "int",
		"~int ∪ string":     "nil",
		"∅ ∪ int":           "int",
		"∅ ∪ ~int":          "int",
		"∅ ∪ ~int ∪ string": "nil",
	} {
		xl := maketl(test)
		got := tstring(xl.structuralType())
		if got != want {
			t.Errorf("(%v).structuralType() == %v; want %v", test, got, want)
		}
	}
}

func TestTermlistUnion(t *testing.T) {
	for _, test := range []struct {
		xl, yl, want string
	}{

		{"∅", "∅", "∅"},
		{"∅", "⊤", "⊤"},
		{"∅", "int", "int"},
		{"⊤", "~int", "⊤"},
		{"int", "~int", "~int"},
		{"int", "string", "int ∪ string"},
		{"int ∪ string", "~string", "int ∪ ~string"},
		{"~int ∪ string", "~string ∪ int", "~int ∪ ~string"},
		{"~int ∪ string ∪ ∅", "~string ∪ int", "~int ∪ ~string"},
		{"~int ∪ string ∪ ⊤", "~string ∪ int", "⊤"},
	} {
		xl := maketl(test.xl)
		yl := maketl(test.yl)
		got := xl.union(yl).String()
		if got != test.want {
			t.Errorf("(%v).union(%v) = %v; want %v", test.xl, test.yl, got, test.want)
		}
	}
}

func TestTermlistIntersect(t *testing.T) {
	for _, test := range []struct {
		xl, yl, want string
	}{

		{"∅", "∅", "∅"},
		{"∅", "⊤", "∅"},
		{"∅", "int", "∅"},
		{"⊤", "~int", "~int"},
		{"int", "~int", "int"},
		{"int", "string", "∅"},
		{"int ∪ string", "~string", "string"},
		{"~int ∪ string", "~string ∪ int", "int ∪ string"},
		{"~int ∪ string ∪ ∅", "~string ∪ int", "int ∪ string"},
		{"~int ∪ string ∪ ⊤", "~string ∪ int", "int ∪ ~string"},
	} {
		xl := maketl(test.xl)
		yl := maketl(test.yl)
		got := xl.intersect(yl).String()
		if got != test.want {
			t.Errorf("(%v).intersect(%v) = %v; want %v", test.xl, test.yl, got, test.want)
		}
	}
}

func TestTermlistEqual(t *testing.T) {
	for _, test := range []struct {
		xl, yl string
		want   bool
	}{
		{"∅", "∅", true},
		{"∅", "⊤", false},
		{"⊤", "⊤", true},
		{"⊤ ∪ int", "⊤", true},
		{"⊤ ∪ int", "string ∪ ⊤", true},
		{"int ∪ ~string", "string ∪ int", false},
		{"int ∪ ~string ∪ ∅", "string ∪ int ∪ ~string", true},
	} {
		xl := maketl(test.xl)
		yl := maketl(test.yl)
		got := xl.equal(yl)
		if got != test.want {
			t.Errorf("(%v).equal(%v) = %v; want %v", test.xl, test.yl, got, test.want)
		}
	}
}

func TestTermlistIncludes(t *testing.T) {
	for _, test := range []struct {
		xl, typ string
		want    bool
	}{
		{"∅", "int", false},
		{"⊤", "int", true},
		{"~int", "int", true},
		{"int", "string", false},
		{"~int", "string", false},
		{"int ∪ string", "string", true},
		{"~int ∪ string", "int", true},
		{"~int ∪ string ∪ ∅", "string", true},
		{"~string ∪ ∅ ∪ ⊤", "int", true},
	} {
		xl := maketl(test.xl)
		yl := testTerm(test.typ).typ
		got := xl.includes(yl)
		if got != test.want {
			t.Errorf("(%v).includes(%v) = %v; want %v", test.xl, yl, got, test.want)
		}
	}
}

func TestTermlistSupersetOf(t *testing.T) {
	for _, test := range []struct {
		xl, typ string
		want    bool
	}{
		{"∅", "∅", true},
		{"∅", "⊤", false},
		{"∅", "int", false},
		{"⊤", "∅", true},
		{"⊤", "⊤", true},
		{"⊤", "int", true},
		{"⊤", "~int", true},
		{"~int", "int", true},
		{"~int", "~int", true},
		{"int", "~int", false},
		{"int", "string", false},
		{"~int", "string", false},
		{"int ∪ string", "string", true},
		{"int ∪ string", "~string", false},
		{"~int ∪ string", "int", true},
		{"~int ∪ string ∪ ∅", "string", true},
		{"~string ∪ ∅ ∪ ⊤", "int", true},
	} {
		xl := maketl(test.xl)
		y := testTerm(test.typ)
		got := xl.supersetOf(y)
		if got != test.want {
			t.Errorf("(%v).supersetOf(%v) = %v; want %v", test.xl, y, got, test.want)
		}
	}
}

func TestTermlistSubsetOf(t *testing.T) {
	for _, test := range []struct {
		xl, yl string
		want   bool
	}{
		{"∅", "∅", true},
		{"∅", "⊤", true},
		{"⊤", "∅", false},
		{"⊤", "⊤", true},
		{"int", "int ∪ string", true},
		{"~int", "int ∪ string", false},
		{"~int", "string ∪ string ∪ int ∪ ~int", true},
		{"int ∪ string", "string", false},
		{"int ∪ string", "string ∪ int", true},
		{"int ∪ ~string", "string ∪ int", false},
		{"int ∪ ~string", "string ∪ int ∪ ⊤", true},
		{"int ∪ ~string", "string ∪ int ∪ ∅ ∪ string", false},
	} {
		xl := maketl(test.xl)
		yl := maketl(test.yl)
		got := xl.subsetOf(yl)
		if got != test.want {
			t.Errorf("(%v).subsetOf(%v) = %v; want %v", test.xl, test.yl, got, test.want)
		}
	}
}
