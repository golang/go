// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"math/big"
	"testing"
)

func nodrune(r rune) *Node {
	return nodlit(Val{&Mpint{Val: *big.NewInt(int64(r)), Rune: true}})
}

func nodflt(f float64) *Node {
	return nodlit(Val{&Mpflt{Val: *big.NewFloat(f)}})
}

func TestCaseClauseByConstVal(t *testing.T) {
	tests := []struct {
		a, b *Node
	}{
		// CTFLT
		{nodflt(0.1), nodflt(0.2)},
		// CTINT
		{nodintconst(0), nodintconst(1)},
		// CTRUNE
		{nodrune('a'), nodrune('b')},
		// CTSTR
		{nodlit(Val{"ab"}), nodlit(Val{"abc"})},
		{nodlit(Val{"ab"}), nodlit(Val{"xyz"})},
		{nodlit(Val{"abc"}), nodlit(Val{"xyz"})},
	}
	for i, test := range tests {
		a := caseClause{node: nod(OXXX, test.a, nil)}
		b := caseClause{node: nod(OXXX, test.b, nil)}
		s := caseClauseByConstVal{a, b}
		if less := s.Less(0, 1); !less {
			t.Errorf("%d: caseClauseByConstVal(%v, %v) = false", i, test.a, test.b)
		}
		if less := s.Less(1, 0); less {
			t.Errorf("%d: caseClauseByConstVal(%v, %v) = true", i, test.a, test.b)
		}
	}
}
