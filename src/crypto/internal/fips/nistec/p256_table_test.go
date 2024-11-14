// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && !arm64 && !ppc64le && !s390x) || purego

package nistec

import (
	"bytes"
	"crypto/internal/fips/nistec/fiat"
	"fmt"
	"testing"
)

func TestP256PrecomputedTable(t *testing.T) {
	base := NewP256Point().SetGenerator()

	for i := 0; i < 43; i++ {
		t.Run(fmt.Sprintf("table[%d]", i), func(t *testing.T) {
			testP256AffineTable(t, base, &p256GeneratorTables[i])
		})

		for k := 0; k < 6; k++ {
			base.Double(base)
		}
	}
}

func testP256AffineTable(t *testing.T, base *P256Point, table *p256AffineTable) {
	p := NewP256Point()
	zInv := new(fiat.P256Element)

	for j := 0; j < 32; j++ {
		p.Add(p, base)

		// Convert p to affine coordinates.
		zInv.Invert(&p.z)
		p.x.Mul(&p.x, zInv)
		p.y.Mul(&p.y, zInv)
		p.z.One()

		if !bytes.Equal(table[j].x.Bytes(), p.x.Bytes()) ||
			!bytes.Equal(table[j].y.Bytes(), p.y.Bytes()) {
			t.Fatalf("incorrect table entry at index %d", j)
		}
	}
}
