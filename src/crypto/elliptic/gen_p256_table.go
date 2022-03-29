// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"crypto/elliptic"
	"encoding/binary"
	"log"
	"os"
)

func main() {
	// Generate precomputed p256 tables.
	var pre [43][32 * 8]uint64
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
					elliptic.P256PointDoubleAsm(t1, t1)
				}
			}
			// Convert the point to affine form. (Its values are
			// still in Montgomery form however.)
			elliptic.P256Inverse(zInv, t1[8:12])
			elliptic.P256Sqr(zInvSq, zInv, 1)
			elliptic.P256Mul(zInv, zInv, zInvSq)
			elliptic.P256Mul(t1[:4], t1[:4], zInvSq)
			elliptic.P256Mul(t1[4:8], t1[4:8], zInv)
			copy(t1[8:12], basePoint[8:12])
			// Update the table entry
			copy(pre[i][j*8:], t1[:8])
		}
		if j == 0 {
			elliptic.P256PointDoubleAsm(t2, basePoint)
		} else {
			elliptic.P256PointAddAsm(t2, t2, basePoint)
		}
	}

	var bin []byte

	// Dump the precomputed tables, flattened, little-endian.
	// These tables are used directly by assembly on little-endian platforms.
	// go:embedding the data into a string lets it be stored readonly.
	for i := range &pre {
		for _, v := range &pre[i] {
			var u8 [8]byte
			binary.LittleEndian.PutUint64(u8[:], v)
			bin = append(bin, u8[:]...)
		}
	}

	err := os.WriteFile("p256_asm_table.bin", bin, 0644)
	if err != nil {
		log.Fatal(err)
	}
}
