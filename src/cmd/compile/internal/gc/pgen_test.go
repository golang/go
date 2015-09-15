// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "testing"

// Test all code paths for cmpstackvarlt.
func TestCmpstackvar(t *testing.T) {
	testdata := []struct {
		a, b Node
		lt   bool
	}{
		{
			Node{Class: PAUTO},
			Node{Class: PFUNC},
			false,
		},
		{
			Node{Class: PFUNC},
			Node{Class: PAUTO},
			true,
		},
		{
			Node{Class: PFUNC, Xoffset: 0},
			Node{Class: PFUNC, Xoffset: 10},
			true,
		},
		{
			Node{Class: PFUNC, Xoffset: 20},
			Node{Class: PFUNC, Xoffset: 10},
			false,
		},
		{
			Node{Class: PFUNC, Xoffset: 10},
			Node{Class: PFUNC, Xoffset: 10},
			false,
		},
		{
			Node{Class: PAUTO, Used: true},
			Node{Class: PAUTO, Used: false},
			true,
		},
		{
			Node{Class: PAUTO, Used: false},
			Node{Class: PAUTO, Used: true},
			false,
		},
		{
			Node{Class: PAUTO, Type: &Type{Haspointers: 1}}, // haspointers -> false
			Node{Class: PAUTO, Type: &Type{Haspointers: 2}}, // haspointers -> true
			false,
		},
		{
			Node{Class: PAUTO, Type: &Type{Haspointers: 2}}, // haspointers -> true
			Node{Class: PAUTO, Type: &Type{Haspointers: 1}}, // haspointers -> false
			true,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: true}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: false}},
			true,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: false}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: true}},
			false,
		},
		{
			Node{Class: PAUTO, Type: &Type{Width: 1}, Name: &Name{}},
			Node{Class: PAUTO, Type: &Type{Width: 2}, Name: &Name{}},
			false,
		},
		{
			Node{Class: PAUTO, Type: &Type{Width: 2}, Name: &Name{}},
			Node{Class: PAUTO, Type: &Type{Width: 1}, Name: &Name{}},
			true,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "xyz"}},
			true,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
			false,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "xyz"}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
			false,
		},
	}
	for _, d := range testdata {
		got := cmpstackvarlt(&d.a, &d.b)
		if got != d.lt {
			t.Errorf("want %#v < %#v", d.a, d.b)
		}
	}
}
