// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import "unicode/utf8"

type input struct {
	str   string
	bytes []byte
}

func inputBytes(str []byte) input {
	return input{bytes: str}
}

func inputString(str string) input {
	return input{str: str}
}

func (in *input) setBytes(str []byte) {
	in.str = ""
	in.bytes = str
}

func (in *input) setString(str string) {
	in.str = str
	in.bytes = nil
}

func (in *input) _byte(p int) byte {
	if in.bytes == nil {
		return in.str[p]
	}
	return in.bytes[p]
}

func (in *input) skipASCII(p, max int) int {
	if in.bytes == nil {
		for ; p < max && in.str[p] < utf8.RuneSelf; p++ {
		}
	} else {
		for ; p < max && in.bytes[p] < utf8.RuneSelf; p++ {
		}
	}
	return p
}

func (in *input) skipNonStarter(p int) int {
	if in.bytes == nil {
		for ; p < len(in.str) && !utf8.RuneStart(in.str[p]); p++ {
		}
	} else {
		for ; p < len(in.bytes) && !utf8.RuneStart(in.bytes[p]); p++ {
		}
	}
	return p
}

func (in *input) appendSlice(buf []byte, b, e int) []byte {
	if in.bytes != nil {
		return append(buf, in.bytes[b:e]...)
	}
	for i := b; i < e; i++ {
		buf = append(buf, in.str[i])
	}
	return buf
}

func (in *input) copySlice(buf []byte, b, e int) int {
	if in.bytes == nil {
		return copy(buf, in.str[b:e])
	}
	return copy(buf, in.bytes[b:e])
}

func (in *input) charinfoNFC(p int) (uint16, int) {
	if in.bytes == nil {
		return nfcTrie.lookupString(in.str[p:])
	}
	return nfcTrie.lookup(in.bytes[p:])
}

func (in *input) charinfoNFKC(p int) (uint16, int) {
	if in.bytes == nil {
		return nfkcTrie.lookupString(in.str[p:])
	}
	return nfkcTrie.lookup(in.bytes[p:])
}

func (in *input) hangul(p int) (r rune) {
	if in.bytes == nil {
		if !isHangulString(in.str[p:]) {
			return 0
		}
		r, _ = utf8.DecodeRuneInString(in.str[p:])
	} else {
		if !isHangul(in.bytes[p:]) {
			return 0
		}
		r, _ = utf8.DecodeRune(in.bytes[p:])
	}
	return r
}
