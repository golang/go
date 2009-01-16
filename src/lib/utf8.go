// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// UTF-8 support.

package utf8

export const (
	RuneError = 0xFFFD;
	RuneSelf = 0x80;
	RuneMax = 0x10FFFF;
	UTFMax = 4;
)

const (
	_T1 = 0x00;	// 0000 0000
	_Tx = 0x80;	// 1000 0000
	_T2 = 0xC0;	// 1100 0000
	_T3 = 0xE0;	// 1110 0000
	_T4 = 0xF0;	// 1111 0000
	_T5 = 0xF8;	// 1111 1000

	_Maskx = 0x3F;	// 0011 1111
	_Mask2 = 0x1F;	// 0001 1111
	_Mask3 = 0x0F;	// 0000 1111
	_Mask4 = 0x07;	// 0000 0111

	_Rune1Max = 1<<7 - 1;
	_Rune2Max = 1<<11 - 1;
	_Rune3Max = 1<<16 - 1;
	_Rune4Max = 1<<21 - 1;
)

func decodeRuneInternal(p []byte) (rune, size int, short bool) {
	n := len(p);
	if n < 1 {
		return RuneError, 0, true;
	}
	c0 := p[0];

	// 1-byte, 7-bit sequence?
	if c0 < _Tx {
		return int(c0), 1, false
	}

	// unexpected continuation byte?
	if c0 < _T2 {
		return RuneError, 1, false
	}

	// need first continuation byte
	if n < 2 {
		return RuneError, 1, true
	}
	c1 := p[1];
	if c1 < _Tx || _T2 <= c1 {
		return RuneError, 1, false
	}

	// 2-byte, 11-bit sequence?
	if c0 < _T3 {
		rune = int(c0&_Mask2)<<6 | int(c1&_Maskx);
		if rune <= _Rune1Max {
			return RuneError, 1, false
		}
		return rune, 2, false
	}

	// need second continuation byte
	if n < 3 {
		return RuneError, 1, true
	}
	c2 := p[2];
	if c2 < _Tx || _T2 <= c2 {
		return RuneError, 1, false
	}

	// 3-byte, 16-bit sequence?
	if c0 < _T4 {
		rune = int(c0&_Mask3)<<12 | int(c1&_Maskx)<<6 | int(c2&_Maskx);
		if rune <= _Rune2Max {
			return RuneError, 1, false
		}
		return rune, 3, false
	}

	// need third continuation byte
	if n < 4 {
		return RuneError, 1, true
	}
	c3 := p[3];
	if c3 < _Tx || _T2 <= c3 {
		return RuneError, 1, false
	}

	// 4-byte, 21-bit sequence?
	if c0 < _T5 {
		rune = int(c0&_Mask4)<<18 | int(c1&_Maskx)<<12 | int(c2&_Maskx)<<6 | int(c3&_Maskx);
		if rune <= _Rune3Max {
			return RuneError, 1, false
		}
		return rune, 4, false
	}

	// error
	return RuneError, 1, false
}

func decodeRuneInStringInternal(s string, i int, n int) (rune, size int, short bool) {
	if n < 1 {
		return RuneError, 0, true;
	}
	c0 := s[i];

	// 1-byte, 7-bit sequence?
	if c0 < _Tx {
		return int(c0), 1, false
	}

	// unexpected continuation byte?
	if c0 < _T2 {
		return RuneError, 1, false
	}

	// need first continuation byte
	if n < 2 {
		return RuneError, 1, true
	}
	c1 := s[i+1];
	if c1 < _Tx || _T2 <= c1 {
		return RuneError, 1, false
	}

	// 2-byte, 11-bit sequence?
	if c0 < _T3 {
		rune = int(c0&_Mask2)<<6 | int(c1&_Maskx);
		if rune <= _Rune1Max {
			return RuneError, 1, false
		}
		return rune, 2, false
	}

	// need second continuation byte
	if n < 3 {
		return RuneError, 1, true
	}
	c2 := s[i+2];
	if c2 < _Tx || _T2 <= c2 {
		return RuneError, 1, false
	}

	// 3-byte, 16-bit sequence?
	if c0 < _T4 {
		rune = int(c0&_Mask3)<<12 | int(c1&_Maskx)<<6 | int(c2&_Maskx);
		if rune <= _Rune2Max {
			return RuneError, 1, false
		}
		return rune, 3, false
	}

	// need third continuation byte
	if n < 4 {
		return RuneError, 1, true
	}
	c3 := s[i+3];
	if c3 < _Tx || _T2 <= c3 {
		return RuneError, 1, false
	}

	// 4-byte, 21-bit sequence?
	if c0 < _T5 {
		rune = int(c0&_Mask4)<<18 | int(c1&_Maskx)<<12 | int(c2&_Maskx)<<6 | int(c3&_Maskx);
		if rune <= _Rune3Max {
			return RuneError, 1, false
		}
		return rune, 4, false
	}

	// error
	return RuneError, 1, false
}

export func FullRune(p []byte) bool {
	rune, size, short := decodeRuneInternal(p);
	return !short
}

export func FullRuneInString(s string, i int) bool {
	rune, size, short := decodeRuneInStringInternal(s, i, len(s) - i);
	return !short
}

export func DecodeRune(p []byte) (rune, size int) {
	var short bool;
	rune, size, short = decodeRuneInternal(p);
	return;
}

export func DecodeRuneInString(s string, i int) (rune, size int) {
	var short bool;
	rune, size, short = decodeRuneInStringInternal(s, i, len(s) - i);
	return;
}

export func RuneLen(rune int) int {
	switch {
	case rune <= _Rune1Max:
		return 1;
	case rune <= _Rune2Max:
		return 2;
	case rune <= _Rune3Max:
		return 3;
	case rune <= _Rune4Max:
		return 4;
	}
	return -1;
}

export func EncodeRune(rune int, p []byte) int {
	if rune <= _Rune1Max {
		p[0] = byte(rune);
		return 1;
	}

	if rune <= _Rune2Max {
		p[0] = _T2 | byte(rune>>6);
		p[1] = _Tx | byte(rune)&_Maskx;
		return 2;
	}

	if rune > RuneMax {
		rune = RuneError
	}

	if rune <= _Rune3Max {
		p[0] = _T3 | byte(rune>>12);
		p[1] = _Tx | byte(rune>>6)&_Maskx;
		p[2] = _Tx | byte(rune)&_Maskx;
		return 3;
	}

	p[0] = _T4 | byte(rune>>18);
	p[1] = _Tx | byte(rune>>12)&_Maskx;
	p[2] = _Tx | byte(rune>>6)&_Maskx;
	p[3] = _Tx | byte(rune)&_Maskx;
	return 4;
}

export func RuneCount(p []byte) int {
	i := 0;
	var n int;
	for n = 0; i < len(p); n++ {
		if p[i] < RuneSelf {
			i++;
		} else {
			rune, size := DecodeRune(p[i:len(p)]);
			i += size;
		}
	}
	return n;
}

export func RuneCountInString(s string, i int, l int) int {
	ei := i + l;
	n := 0;
	for n = 0; i < ei; n++ {
		if s[i] < RuneSelf {
			i++;
		} else {
			rune, size, short := decodeRuneInStringInternal(s, i, ei - i);
			i += size;
		}
	}
	return n;
}

