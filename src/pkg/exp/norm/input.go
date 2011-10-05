// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import "utf8"

type input interface {
	skipASCII(p int) int
	skipNonStarter() int
	appendSlice(buf []byte, s, e int) []byte
	copySlice(buf []byte, s, e int)
	charinfo(p int) (uint16, int)
	decomposeNFC(p int) uint16
	decomposeNFKC(p int) uint16
	hangul(p int) uint32
}

type inputString string

func (s inputString) skipASCII(p int) int {
	for ; p < len(s) && s[p] < utf8.RuneSelf; p++ {
	}
	return p
}

func (s inputString) skipNonStarter() int {
	p := 0
	for ; p < len(s) && !utf8.RuneStart(s[p]); p++ {
	}
	return p
}

func (s inputString) appendSlice(buf []byte, b, e int) []byte {
	for i := b; i < e; i++ {
		buf = append(buf, s[i])
	}
	return buf
}

func (s inputString) copySlice(buf []byte, b, e int) {
	copy(buf, s[b:e])
}

func (s inputString) charinfo(p int) (uint16, int) {
	return charInfoTrie.lookupString(string(s[p:]))
}

func (s inputString) decomposeNFC(p int) uint16 {
	return nfcDecompTrie.lookupStringUnsafe(string(s[p:]))
}

func (s inputString) decomposeNFKC(p int) uint16 {
	return nfkcDecompTrie.lookupStringUnsafe(string(s[p:]))
}

func (s inputString) hangul(p int) uint32 {
	if !isHangulString(string(s[p:])) {
		return 0
	}
	rune, _ := utf8.DecodeRuneInString(string(s[p:]))
	return uint32(rune)
}

type inputBytes []byte

func (s inputBytes) skipASCII(p int) int {
	for ; p < len(s) && s[p] < utf8.RuneSelf; p++ {
	}
	return p
}

func (s inputBytes) skipNonStarter() int {
	p := 0
	for ; p < len(s) && !utf8.RuneStart(s[p]); p++ {
	}
	return p
}

func (s inputBytes) appendSlice(buf []byte, b, e int) []byte {
	return append(buf, s[b:e]...)
}

func (s inputBytes) copySlice(buf []byte, b, e int) {
	copy(buf, s[b:e])
}

func (s inputBytes) charinfo(p int) (uint16, int) {
	return charInfoTrie.lookup(s[p:])
}

func (s inputBytes) decomposeNFC(p int) uint16 {
	return nfcDecompTrie.lookupUnsafe(s[p:])
}

func (s inputBytes) decomposeNFKC(p int) uint16 {
	return nfkcDecompTrie.lookupUnsafe(s[p:])
}

func (s inputBytes) hangul(p int) uint32 {
	if !isHangul(s[p:]) {
		return 0
	}
	rune, _ := utf8.DecodeRune(s[p:])
	return uint32(rune)
}
