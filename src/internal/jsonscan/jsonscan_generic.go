// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && !arm64

package jsonscan

import (
	"internal/byteorder"
	"math/bits"
	"unsafe"
)

const (
	wordLen  = 8
	lowBits  = 0x0101010101010101
	highBits = 0x8080808080808080
)

func indexStringSpecial(p *byte, n int) int {
	s := unsafe.Slice(p, n)
	i := 0
	for ; n-i >= wordLen; i += wordLen {
		if m := stringSpecialMask(byteorder.LEUint64(s[i:])); m != 0 {
			return i + firstMarkedByte(m)
		}
	}
	return i + indexStringSpecialRef(s[i:])
}

func indexEscapeHTML(p *byte, n int) int {
	s := unsafe.Slice(p, n)
	i := 0
	for ; n-i >= wordLen; i += wordLen {
		if m := escapeHTMLMask(byteorder.LEUint64(s[i:])); m != 0 {
			return i + firstMarkedByte(m)
		}
	}
	return i + indexEscapeHTMLRef(s[i:])
}

func firstMarkedByte(m uint64) int {
	return bits.TrailingZeros64(m) / 8
}

func controlOrHigh(w uint64) uint64 {
	return ((w - 0x20*lowBits) | w) & highBits
}

func equalTo(w, splat uint64) uint64 {
	x := w ^ splat
	return (x - lowBits) &^ x & highBits
}

func stringSpecialMask(w uint64) uint64 {
	return controlOrHigh(w) | equalTo(w, '"'*lowBits) | equalTo(w, '\\'*lowBits)
}

func escapeHTMLMask(w uint64) uint64 {
	return stringSpecialMask(w) | equalTo(w, '<'*lowBits) | equalTo(w, '>'*lowBits) | equalTo(w, '&'*lowBits)
}
