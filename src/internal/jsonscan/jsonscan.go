// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jsonscan finds the end of a run of plain ASCII bytes inside a JSON
// string body. It uses SSE2 or AVX2 on amd64, NEON on arm64, and SWAR over
// uint64 words on every other architecture. No function reads outside its
// argument.
package jsonscan

import (
	"internal/abi"
	"unsafe"
)

const minKernelLen = 16

// IndexStringSpecial returns the index of the first byte c of b with
// c == '"' || c == '\\' || c < 0x20 || c >= 0x80, or len(b) if b has none.
func IndexStringSpecial[B ~[]byte | ~string](b B) int {
	if len(b) < minKernelLen {
		return indexStringSpecialRef(b)
	}
	return indexStringSpecial(view(b))
}

// IndexEscapeHTML returns the index of the first byte c of b with
// c < 0x20 || c == '"' || c == '\\' || c == '<' || c == '>' || c == '&' || c >= 0x80,
// or len(b) if b has none.
func IndexEscapeHTML[B ~[]byte | ~string](b B) int {
	if len(b) < minKernelLen {
		return indexEscapeHTMLRef(b)
	}
	return indexEscapeHTML(view(b))
}

// IndexEscape returns the index of the first byte c of b with
// c < 0x20 || c == '"' || c == '\\' || c >= 0x80, or len(b) if b has none.
func IndexEscape[B ~[]byte | ~string](b B) int {
	return IndexStringSpecial(b)
}

func view[B ~[]byte | ~string](b B) (*byte, int) {
	s := *(*string)(unsafe.Pointer(&b))
	// SAFETY: the kernels only read s[0:len(s)] and keep no pointer, and the
	// returned pointer keeps the bytes alive as a call argument. NoEscape
	// stops the compiler from treating b as written by the bodyless assembly
	// function, which would force callers to copy a string converted to []byte.
	return (*byte)(abi.NoEscape(unsafe.Pointer(unsafe.StringData(s)))), len(s)
}

func indexStringSpecialRef[B ~[]byte | ~string](b B) int {
	for i := range len(b) {
		if isStringSpecial(b[i]) {
			return i
		}
	}
	return len(b)
}

func indexEscapeHTMLRef[B ~[]byte | ~string](b B) int {
	for i := range len(b) {
		if isEscapeHTML(b[i]) {
			return i
		}
	}
	return len(b)
}

func isStringSpecial(c byte) bool {
	return c < 0x20 || c == '"' || c == '\\' || c >= 0x80
}

func isEscapeHTML(c byte) bool {
	return isStringSpecial(c) || c == '<' || c == '>' || c == '&'
}
