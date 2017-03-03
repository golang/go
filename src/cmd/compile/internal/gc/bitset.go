// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

type bitset8 uint8

func (f *bitset8) set(mask uint8, b bool) {
	if b {
		*(*uint8)(f) |= mask
	} else {
		*(*uint8)(f) &^= mask
	}
}

type bitset16 uint16

func (f *bitset16) set(mask uint16, b bool) {
	if b {
		*(*uint16)(f) |= mask
	} else {
		*(*uint16)(f) &^= mask
	}
}

type bitset32 uint32

func (f *bitset32) set(mask uint32, b bool) {
	if b {
		*(*uint32)(f) |= mask
	} else {
		*(*uint32)(f) &^= mask
	}
}
