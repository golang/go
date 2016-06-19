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
			caseClause{node: Nod(OXXX, nil, nil), typ: caseKindExprVar},
			caseClause{node: Nod(OXXX, nil, nil), typ: caseKindExprConst},
			+1,
		},
		{
			caseClause{node: Nod(OXXX, nil, nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nil, nil), typ: caseKindExprVar},
			-1,
		},
		// Type switches
		{
			caseClause{node: Nod(OXXX, Nodintconst(0), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, Nodbool(true), nil), typ: caseKindExprConst},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, Nodbool(true), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), typ: caseKindExprConst},
			+1,
		},
		{
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TBOOL, Vargen: 1}}, nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TINT, Vargen: 0}}, nil), typ: caseKindExprConst},
			+1,
		},
		{
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TBOOL, Vargen: 1}}, nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TINT, Vargen: 1}}, nil), typ: caseKindExprConst},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TBOOL, Vargen: 0}}, nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, &Node{Type: &Type{Etype: TINT, Vargen: 1}}, nil), typ: caseKindExprConst},
			-1,
		},
		// Constant values.
		// CTFLT
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.1)}}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.2)}}), nil), typ: caseKindExprConst},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.1)}}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.1)}}), nil), typ: caseKindExprConst},
			0,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.2)}}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpflt{Val: *big.NewFloat(0.1)}}), nil), typ: caseKindExprConst},
			+1,
		},
		// CTINT
		{
			caseClause{node: Nod(OXXX, Nodintconst(0), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), typ: caseKindExprConst},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), typ: caseKindExprConst},
			0,
		},
		{
			caseClause{node: Nod(OXXX, Nodintconst(1), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, Nodintconst(0), nil), typ: caseKindExprConst},
			+1,
		},
		// CTRUNE
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('a'), Rune: true}}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('b'), Rune: true}}), nil), typ: caseKindExprConst},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('b'), Rune: true}}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('b'), Rune: true}}), nil), typ: caseKindExprConst},
			0,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('b'), Rune: true}}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{&Mpint{Val: *big.NewInt('a'), Rune: true}}), nil), typ: caseKindExprConst},
			+1,
		},
		// CTSTR
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"ab"}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), typ: caseKindExprConst},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{"xyz"}), nil), typ: caseKindExprConst},
			-1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), typ: caseKindExprConst},
			0,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{"ab"}), nil), typ: caseKindExprConst},
			+1,
		},
		{
			caseClause{node: Nod(OXXX, nodlit(Val{"xyz"}), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodlit(Val{"abc"}), nil), typ: caseKindExprConst},
			+1,
		},
		// Everything else should compare equal.
		{
			caseClause{node: Nod(OXXX, nodnil(), nil), typ: caseKindExprConst},
			caseClause{node: Nod(OXXX, nodnil(), nil), typ: caseKindExprConst},
			0,
		},
	}
	for i, d := range testdata {
		got := exprcmp(&d.a, &d.b)
		if d.want != got {
			t.Errorf("%d: exprcmp(a, b) = %d; want %d", i, got, d.want)
			t.Logf("\ta = caseClause{node: %#v, typ: %#v}", d.a.node, d.a.typ)
			t.Logf("\tb = caseClause{node: %#v, typ: %#v}", d.b.node, d.b.typ)
		}
	}
}
