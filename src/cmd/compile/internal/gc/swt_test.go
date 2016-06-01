// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/big"
	"testing"
)

func TestExprcmp(t *testing.T) {
	testdata := []struct {
		a, b caseClause
		want int
	}{
		// Non-constants.
		{
			caseClause{node: Nod(OXXX, nil, nil)},
			caseClause{node: Nod(OXXX, nil, nil), isconst: true},
			+1,
		},
		{
			caseClause{node: Nod(OXXX, nil, nil), isconst: true},
			caseClause{node: Nod(OXXX, nil, nil)},
			-1,
		},
		// Type switches
		{
			caseClause{node: Nod(OXXX, Nodintconst(0), nil), isconst: true},
			caseClause{node: Nod(OXXX, Nodbool(true), nil), isconst: true},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, Nodbool(true), nil), isconst: true},
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), isconst: true},
			+1,
		},
		{
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TBOOL, Vargen: 1}}, nil), isconst: true},
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TINT, Vargen: 0}}, nil), isconst: true},
			+1,
		},
		{
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TBOOL, Vargen: 1}}, nil), isconst: true},
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TINT, Vargen: 1}}, nil), isconst: true},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TBOOL, Vargen: 0}}, nil), isconst: true},
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TINT, Vargen: 1}}, nil), isconst: true},
			-1,
		},
		// Constant values.
		// CTFLT
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.1)}}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.2)}}), nil), isconst: true},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.1)}}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.1)}}), nil), isconst: true},
			0,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.2)}}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.1)}}), nil), isconst: true},
			+1,
		},
		// CTINT
		{
			caseClause{node: Nod(OXXX, Nodintconst(0), nil), isconst: true},
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), isconst: true},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), isconst: true},
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), isconst: true},
			0,
		},
		{
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), isconst: true},
			caseClause{node: Nod(OXXX, Nodintconst(0), nil), isconst: true},
			+1,
		},
		// CTRUNE
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('a'), Rune: true}}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('b'), Rune: true}}), nil), isconst: true},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('b'), Rune: true}}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('b'), Rune: true}}), nil), isconst: true},
			0,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('b'), Rune: true}}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('a'), Rune: true}}), nil), isconst: true},
			+1,
		},
		// CTSTR
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"ab"}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), isconst: true},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{"xyz"}), nil), isconst: true},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), isconst: true},
			0,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{"ab"}), nil), isconst: true},
			+1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"xyz"}), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), isconst: true},
			+1,
		},
		// Everything else should compare equal.
		{
			caseClause{node: Nod(OXXX, nodnil(), nil), isconst: true},
			caseClause{node: Nod(OXXX, nodnil(), nil), isconst: true},
			0,
		},
	}
	for i, d := range testdata {
		got := exprcmp(d.a, d.b)
		if d.want != got {
			t.Errorf("%d: exprcmp(a, b) = %d; want %d", i, got, d.want)
			t.Logf("\ta = caseClause{node: %#v, isconst: %v}", d.a.node, d.a.isconst)
			t.Logf("\tb = caseClause{node: %#v, isconst: %v}", d.b.node, d.b.isconst)
		}
	}
}
