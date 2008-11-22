// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// UTF-8 support.

package utf8

export const (
	RuneError = 0xFFFD;
	RuneSelf = 0x80;
	RuneMax = 1<<21 - 1;
)

const (
	T1 = 0x00;	// 0000 0000
	Tx = 0x80;	// 1000 0000
	T2 = 0xC0;	// 1100 0000
	T3 = 0xE0;	// 1110 0000
	T4 = 0xF0;	// 1111 0000
	T5 = 0xF8;	// 1111 1000

	Maskx = 0x3F;	// 0011 1111
	Mask2 = 0x1F;	// 0001 1111
	Mask3 = 0x0F;	// 0000 1111
	Mask4 = 0x07;	// 0000 0111

	Rune1Max = 1<<7 - 1;
	Rune2Max = 1<<11 - 1;
	Rune3Max = 1<<16 - 1;
	Rune4Max = 1<<21 - 1;
)

func DecodeRuneInternal(p *[]byte) (rune, size int, short bool) {
	if len(p) < 1 {
		return RuneError, 0, true;
	}
	c0 := p[0];

	// 1-byte, 7-bit sequence?
	if c0 < Tx {
		return int(c0), 1, false
	}

	// unexpected continuation byte?
	if c0 < T2 {
		return RuneError, 1, false
	}

	// need first continuation byte
	if len(p) < 2 {
		return RuneError, 1, true
	}
	c1 := p[1];
	if c1 < Tx || T2 <= c1 {
		return RuneError, 1, false
	}

	// 2-byte, 11-bit sequence?
	if c0 < T3 {
		rune = int(c0&Mask2)<<6 | int(c1&Maskx);
		if rune <= Rune1Max {
			return RuneError, 1, false
		}
		return rune, 2, false
	}

	// need second continuation byte
	if len(p) < 3 {
		return RuneError, 1, true
	}
	c2 := p[2];
	if c2 < Tx || T2 <= c2 {
		return RuneError, 1, false
	}

	// 3-byte, 16-bit sequence?
	if c0 < T4 {
		rune = int(c0&Mask3)<<12 | int(c1&Maskx)<<6 | int(c2&Maskx);
		if rune <= Rune2Max {
			return RuneError, 1, false
		}
		return rune, 3, false
	}

	// need third continuation byte
	if len(p) < 4 {
		return RuneError, 1, true
	}
	c3 := p[3];
	if c3 < Tx || T2 <= c3 {
		return RuneError, 1, false
	}

	// 4-byte, 21-bit sequence?
	if c0 < T5 {
		rune = int(c0&Mask4)<<18 | int(c1&Maskx)<<12 | int(c2&Maskx)<<6 | int(c3&Maskx);
		if rune <= Rune3Max {
			return RuneError, 1, false
		}
		return rune, 4, false
	}

	// error
	return RuneError, 1, false
}

export func FullRune(p *[]byte) bool {
	rune, size, short := DecodeRuneInternal(p);
	return !short
}

export func DecodeRune(p *[]byte) (rune, size int) {
	var short bool;
	rune, size, short = DecodeRuneInternal(p);
	return;
}

export func RuneLen(rune int) int {
	switch {
	case rune <= Rune1Max:
		return 1;
	case rune <= Rune2Max:
		return 2;
	case rune <= Rune3Max:
		return 3;
	case rune <= Rune4Max:
		return 4;
	}
	return -1;
}

export func EncodeRune(rune int, p *[]byte) int {
	if rune <= Rune1Max {
		p[0] = byte(rune);
		return 1;
	}

	if rune <= Rune2Max {
		p[0] = T2 | byte(rune>>6);
		p[1] = Tx | byte(rune)&Maskx;
		return 2;
	}

	if rune > RuneMax {
		rune = RuneError
	}

	if rune <= Rune3Max {
		p[0] = T3 | byte(rune>>12);
		p[1] = Tx | byte(rune>>6)&Maskx;
		p[2] = Tx | byte(rune)&Maskx;
		return 3;
	}

	p[0] = T4 | byte(rune>>18);
	p[1] = Tx | byte(rune>>12)&Maskx;
	p[2] = Tx | byte(rune>>6)&Maskx;
	p[3] = Tx | byte(rune)&Maskx;
	return 4;
}

