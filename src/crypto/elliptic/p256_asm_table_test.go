// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64
// +build amd64 arm64

package elliptic

import (
	"reflect"
	"testing"
)

func TestP256PrecomputedTable(t *testing.T) {

	basePoint := []uint64{
		0x79e730d418a9143c, 0x75ba95fc5fedb601, 0x79fb732b77622510, 0x18905f76a53755c6,
		0xddf25357ce95560a, 0x8b4ab8e4ba19e45c, 0xd2e88688dd21f325, 0x8571ff1825885d85,
		0x0000000000000001, 0xffffffff00000000, 0xffffffffffffffff, 0x00000000fffffffe,
	}
	t1 := make([]uint64, 12)
	t2 := make([]uint64, 12)
	copy(t2, basePoint)

	zInv := make([]uint64, 4)
	zInvSq := make([]uint64, 4)
	for j := 0; j < 32; j++ {
		copy(t1, t2)
		for i := 0; i < 43; i++ {
			// The window size is 6 so we need to double 6 times.
			if i != 0 {
				for k := 0; k < 6; k++ {
					p256PointDoubleAsm(t1, t1)
				}
			}
			// Convert the point to affine form. (Its values are
			// still in Montgomery form however.)
			p256Inverse(zInv, t1[8:12])
			p256Sqr(zInvSq, zInv, 1)
			p256Mul(zInv, zInv, zInvSq)

			p256Mul(t1[:4], t1[:4], zInvSq)
			p256Mul(t1[4:8], t1[4:8], zInv)

			copy(t1[8:12], basePoint[8:12])

			if got, want := p256Precomputed[i][j*8:(j*8)+8], t1[:8]; !reflect.DeepEqual(got, want) {
				t.Fatalf("Unexpected table entry at [%d][%d:%d]: got %v, want %v", i, j*8, (j*8)+8, got, want)
			}
		}
		if j == 0 {
			p256PointDoubleAsm(t2, basePoint)
		} else {
			p256PointAddAsm(t2, t2, basePoint)
		}
	}

}
