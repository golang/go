// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

const BADWIDTH = -1000000000

type bitset8 uint8

func (f *bitset8) set(mask uint8, b bool) {
	if b {
		*(*uint8)(f) |= mask
	} else {
		*(*uint8)(f) &^= mask
	}
}
