// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bytes operations implements byte operations which work
// over sequences.  This package favors efficiency.

package bytes

// ShiftLeft function does bit shifts along a slice.  When shiftBits is positive, the bits
// shift to the left, and when shiftBits is negative, the bits shift to the right.  As
// this operates directly on the slice, memory is conserved.  For I/O conservation, each
// element is read once and written once.
func ShiftLeft(dst []byte, shiftBits int) {
	if shiftBits < 0 {
		rsh(dst, -shiftBits)
	} else {
		lsh(dst, shiftBits)
	}
}

// Right shift a number of bits
func rsh(dst []byte, shiftBits int) {
	lenDst := len(dst)
	if shiftBits == 0 || lenDst == 0 {
		return
	} else if shiftBits < 0 {
		panic("bytes error: negative shift amount")
	}

	// determine the pad bytes
	pad := shiftBits / 8

	if shiftBits < lenDst*8 {

		// preset shift registers
		shift := shiftBits % 8
		trunc := 8 - shift

		// do the shift
		cur := byte(0)
		next := byte(dst[lenDst-1-pad])
		for i := lenDst - 1; i > pad; i-- {
			cur = next // encourage compilers to optimize
			next = dst[i-pad-1]
			dst[i] = (cur >> shift) | (next << trunc)
		}
		dst[pad] = dst[0] >> shift
	} else {
		pad = lenDst
	}

	// pad out the rest
	for i := 0; i < pad; i++ {
		dst[i] = 0
	}
}

// Left shift a number of bits
func lsh(dst []byte, shiftBits int) {
	lenDst := len(dst)
	if shiftBits == 0 || lenDst == 0 {
		return
	} else if shiftBits < 0 {
		panic("bytes error: negative shift amount")
	}

	// determine the pad bytes
	pad := shiftBits / 8

	if shiftBits < lenDst*8 {

		// preset shift registers
		shift := shiftBits % 8
		trunc := 8 - shift

		// do the shift
		cur := byte(0)
		next := byte(dst[pad])
		for i := 0; i < lenDst-pad-1; i++ {
			cur = next // encourage compilers to optimize
			next = dst[i+pad+1]
			dst[i] = (cur << shift) | (next >> trunc)
		}
		dst[lenDst-pad-1] = dst[lenDst-1] << shift
	} else {
		pad = lenDst
	}

	// pad out the rest
	for i := lenDst - pad; i < lenDst; i++ {
		dst[i] = 0
	}
}
