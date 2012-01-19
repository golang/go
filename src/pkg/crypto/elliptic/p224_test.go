// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"math/big"
	"testing"
)

var toFromBigTests = []string{
	"0",
	"1",
	"23",
	"b70e0cb46bb4bf7f321390b94a03c1d356c01122343280d6105c1d21",
	"706a46d476dcb76798e6046d89474788d164c18032d268fd10704fa6",
}

func p224AlternativeToBig(in *p224FieldElement) *big.Int {
	ret := new(big.Int)
	tmp := new(big.Int)

	for i := uint(0); i < 8; i++ {
		tmp.SetInt64(int64(in[i]))
		tmp.Lsh(tmp, 28*i)
		ret.Add(ret, tmp)
	}
	ret.Mod(ret, p224.P)
	return ret
}

func TestToFromBig(t *testing.T) {
	for i, test := range toFromBigTests {
		n, _ := new(big.Int).SetString(test, 16)
		var x p224FieldElement
		p224FromBig(&x, n)
		m := p224ToBig(&x)
		if n.Cmp(m) != 0 {
			t.Errorf("#%d: %x != %x", i, n, m)
		}
		q := p224AlternativeToBig(&x)
		if n.Cmp(q) != 0 {
			t.Errorf("#%d: %x != %x (alternative)", i, n, m)
		}
	}
}
