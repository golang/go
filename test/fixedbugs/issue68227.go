// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type someType []uint64

func (s *someType) push(v uint64) {
	*s = append(*s, v)
}

func (s *someType) problematicFn(x1Lo, x1Hi, x2Lo, x2Hi uint64) {
	r1 := int32(int16(x1Lo>>0)) * int32(int16(x2Lo>>0))
	g()
	r3 := int32(int16(x1Lo>>32)) * int32(int16(x2Lo>>32))
	r4 := int32(int16(x1Lo>>48)) * int32(int16(x2Lo>>48))
	r5 := int32(int16(x1Hi>>0)) * int32(int16(x2Hi>>0))
	r7 := int32(int16(x1Hi>>32)) * int32(int16(x2Hi>>32))
	r8 := int32(int16(x1Hi>>48)) * int32(int16(x2Hi>>48))
	s.push(uint64(uint32(r1)) | (uint64(uint32(r3+r4)) << 32))
	s.push(uint64(uint32(r5)) | (uint64(uint32(r7+r8)) << 32))
}

//go:noinline
func g() {
}

func main() {
	s := &someType{}
	s.problematicFn(0x1000100010001, 0x1000100010001, 0xffffffffffffffff, 0xffffffffffffffff)
	for i := 0; i < 2; i++ {
		if got, want := (*s)[i], uint64(0xfffffffeffffffff); got != want {
			fmt.Printf("s[%d]=%x, want %x\n", i, got, want)
		}
	}
}
