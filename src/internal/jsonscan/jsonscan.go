// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jsonscan finds the end of a run of plain ASCII bytes inside a JSON
// string body. It uses SSE2 or AVX2 on amd64, NEON on arm64, and SWAR over
// uint64 words on every other architecture. No function reads outside its
// argument.
package jsonscan

import "unsafe"

const minKernelLen = 16

// IndexStringSpecial returns the index of the first byte c of b with
// c == '"' || c == '\\' || c < 0x20 || c >= 0x80, or len(b) if b has none.
func IndexStringSpecial[B ~[]byte | ~string](b B) int {
	if len(b) < minKernelLen {
		return indexStringSpecialRef(b)
	}
	s := view(b)
	return indexStringSpecial(unsafe.StringData(s), len(s))
}

// IndexEscapeHTML returns the index of the first byte c of b with
// c < 0x20 || c == '"' || c == '\\' || c == '<' || c == '>' || c == '&' || c >= 0x80,
// or len(b) if b has none.
func IndexEscapeHTML[B ~[]byte | ~string](b B) int {
	if len(b) < minKernelLen {
		return indexEscapeHTMLRef(b)
	}
	s := view(b)
	return indexEscapeHTML(unsafe.StringData(s), len(s))
}

// IndexEscape returns the index of the first byte c of b with
// c < 0x20 || c == '"' || c == '\\' || c >= 0x80, or len(b) if b has none.
func IndexEscape[B ~[]byte | ~string](b B) int {
	return IndexStringSpecial(b)
}

func view[B ~[]byte | ~string](b B) string {
	return *(*string)(unsafe.Pointer(&b))
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
