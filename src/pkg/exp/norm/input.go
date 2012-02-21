// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import "unicode/utf8"

type input interface {
	skipASCII(p, max int) int
	skipNonStarter(p int) int
	appendSlice(buf []byte, s, e int) []byte
	copySlice(buf []byte, s, e int)
	charinfoNFC(p int) (uint16, int)
	charinfoNFKC(p int) (uint16, int)
	hangul(p int) rune
}

type inputString string

func (s inputString) skipASCII(p, max int) int {
	for ; p < max && s[p] < utf8.RuneSelf; p++ {
	}
	return p
}

func (s inputString) skipNonStarter(p int) int {
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

func (s inputString) charinfoNFC(p int) (uint16, int) {
	return nfcTrie.lookupString(string(s[p:]))
}

func (s inputString) charinfoNFKC(p int) (uint16, int) {
	return nfkcTrie.lookupString(string(s[p:]))
}

func (s inputString) hangul(p int) rune {
	if !isHangulString(string(s[p:])) {
		return 0
	}
	rune, _ := utf8.DecodeRuneInString(string(s[p:]))
	return rune
}

type inputBytes []byte

func (s inputBytes) skipASCII(p, max int) int {
	for ; p < max && s[p] < utf8.RuneSelf; p++ {
	}
	return p
}

func (s inputBytes) skipNonStarter(p int) int {
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

func (s inputBytes) charinfoNFC(p int) (uint16, int) {
	return nfcTrie.lookup(s[p:])
}

func (s inputBytes) charinfoNFKC(p int) (uint16, int) {
	return nfkcTrie.lookup(s[p:])
}

func (s inputBytes) hangul(p int) rune {
	if !isHangul(s[p:]) {
		return 0
	}
	rune, _ := utf8.DecodeRune(s[p:])
	return rune
}
