// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

type bitset8 uint8

func (f *bitset8) set(mask uint8, b bool) {
	if b {
		*(*uint8)(f) |= mask
	} else {
		*(*uint8)(f) &^= mask
	}
}

func (f bitset8) get2(shift uint8) uint8 {
	return uint8(f>>shift) & 3
}

// set2 sets two bits in f using the bottom two bits of b.
func (f *bitset8) set2(shift uint8, b uint8) {
	// Clear old bits.
	*(*uint8)(f) &^= 3 << shift
	// Set new bits.
	*(*uint8)(f) |= (b & 3) << shift
}

type bitset16 uint16

func (f *bitset16) set(mask uint16, b bool) {
	if b {
		*(*uint16)(f) |= mask
	} else {
		*(*uint16)(f) &^= mask
	}
}
