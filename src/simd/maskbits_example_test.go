// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"fmt"
	"math/bits"
	"simd"
)

func ExampleMask8s_ToBits() {
	values, _ := simd.LoadInt8sPart([]int8{-1, 0, -1})
	mask := values.ToMask()
	var storage [simd.MaxMaskBitsLen]uint
	n := mask.ToBits(storage[:])
	var selected []int
	// A mask always has at least one word. Consume it directly so a
	// single-word bitmap needs only the loop over its selected bits.
	word := storage[0]
	for word != 0 {
		selected = append(selected, bits.TrailingZeros(word))
		word &= word - 1
	}
	for j, word := range storage[1:n] {
		for word != 0 {
			selected = append(selected, (j+1)*bits.UintSize+bits.TrailingZeros(word))
			word &= word - 1
		}
	}
	fmt.Println(selected)
	// Output: [0 2]
}
