/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 2002 by Lucent Technologies.
 *              Portions Copyright 2009 The Go Authors. All rights reserved.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */

/*
 * This code is copied, with slight editing due to type differences,
 * from a subset of ../lib9/utf/rune.c
 */

package runtime

const (
	bit1 = 7
	bitx = 6
	bit2 = 5
	bit3 = 4
	bit4 = 3
	bit5 = 2

	t1 = ((1 << (bit1 + 1)) - 1) ^ 0xFF /* 0000 0000 */
	tx = ((1 << (bitx + 1)) - 1) ^ 0xFF /* 1000 0000 */
	t2 = ((1 << (bit2 + 1)) - 1) ^ 0xFF /* 1100 0000 */
	t3 = ((1 << (bit3 + 1)) - 1) ^ 0xFF /* 1110 0000 */
	t4 = ((1 << (bit4 + 1)) - 1) ^ 0xFF /* 1111 0000 */
	t5 = ((1 << (bit5 + 1)) - 1) ^ 0xFF /* 1111 1000 */

	rune1 = (1 << (bit1 + 0*bitx)) - 1 /* 0000 0000 0111 1111 */
	rune2 = (1 << (bit2 + 1*bitx)) - 1 /* 0000 0111 1111 1111 */
	rune3 = (1 << (bit3 + 2*bitx)) - 1 /* 1111 1111 1111 1111 */
	rune4 = (1 << (bit4 + 3*bitx)) - 1 /* 0001 1111 1111 1111 1111 1111 */

	maskx = (1 << bitx) - 1 /* 0011 1111 */
	testx = maskx ^ 0xFF    /* 1100 0000 */

	runeerror = 0xFFFD
	runeself  = 0x80

	surrogateMin = 0xD800
	surrogateMax = 0xDFFF

	bad = runeerror

	runemax = 0x10FFFF /* maximum rune value */
)

/*
 * Modified by Wei-Hwa Huang, Google Inc., on 2004-09-24
 * This is a slower but "safe" version of the old chartorune
 * that works on strings that are not necessarily null-terminated.
 *
 * If you know for sure that your string is null-terminated,
 * chartorune will be a bit faster.
 *
 * It is guaranteed not to attempt to access "length"
 * past the incoming pointer.  This is to avoid
 * possible access violations.  If the string appears to be
 * well-formed but incomplete (i.e., to get the whole Rune
 * we'd need to read past str+length) then we'll set the Rune
 * to Bad and return 0.
 *
 * Note that if we have decoding problems for other
 * reasons, we return 1 instead of 0.
 */
func charntorune(s string) (rune, int) {
	/* When we're not allowed to read anything */
	if len(s) <= 0 {
		return bad, 1
	}

	/*
	 * one character sequence (7-bit value)
	 *	00000-0007F => T1
	 */
	c := s[0]
	if c < tx {
		return rune(c), 1
	}

	// If we can't read more than one character we must stop
	if len(s) <= 1 {
		return bad, 1
	}

	/*
	 * two character sequence (11-bit value)
	 *	0080-07FF => t2 tx
	 */
	c1 := s[1] ^ tx
	if (c1 & testx) != 0 {
		return bad, 1
	}
	if c < t3 {
		if c < t2 {
			return bad, 1
		}
		l := ((rune(c) << bitx) | rune(c1)) & rune2
		if l <= rune1 {
			return bad, 1
		}
		return l, 2
	}

	// If we can't read more than two characters we must stop
	if len(s) <= 2 {
		return bad, 1
	}

	/*
	 * three character sequence (16-bit value)
	 *	0800-FFFF => t3 tx tx
	 */
	c2 := s[2] ^ tx
	if (c2 & testx) != 0 {
		return bad, 1
	}
	if c < t4 {
		l := ((((rune(c) << bitx) | rune(c1)) << bitx) | rune(c2)) & rune3
		if l <= rune2 {
			return bad, 1
		}
		if surrogateMin <= l && l <= surrogateMax {
			return bad, 1
		}
		return l, 3
	}

	if len(s) <= 3 {
		return bad, 1
	}

	/*
	 * four character sequence (21-bit value)
	 *	10000-1FFFFF => t4 tx tx tx
	 */
	c3 := s[3] ^ tx
	if (c3 & testx) != 0 {
		return bad, 1
	}
	if c < t5 {
		l := ((((((rune(c) << bitx) | rune(c1)) << bitx) | rune(c2)) << bitx) | rune(c3)) & rune4
		if l <= rune3 || l > runemax {
			return bad, 1
		}
		return l, 4
	}

	// Support for 5-byte or longer UTF-8 would go here, but
	// since we don't have that, we'll just return bad.
	return bad, 1
}

// runetochar converts r to bytes and writes the result to str.
// returns the number of bytes generated.
func runetochar(str []byte, r rune) int {
	/* runes are signed, so convert to unsigned for range check. */
	c := uint32(r)
	/*
	 * one character sequence
	 *	00000-0007F => 00-7F
	 */
	if c <= rune1 {
		str[0] = byte(c)
		return 1
	}
	/*
	 * two character sequence
	 *	0080-07FF => t2 tx
	 */
	if c <= rune2 {
		str[0] = byte(t2 | (c >> (1 * bitx)))
		str[1] = byte(tx | (c & maskx))
		return 2
	}

	/*
	 * If the rune is out of range or a surrogate half, convert it to the error rune.
	 * Do this test here because the error rune encodes to three bytes.
	 * Doing it earlier would duplicate work, since an out of range
	 * rune wouldn't have fit in one or two bytes.
	 */
	if c > runemax {
		c = runeerror
	}
	if surrogateMin <= c && c <= surrogateMax {
		c = runeerror
	}

	/*
	 * three character sequence
	 *	0800-FFFF => t3 tx tx
	 */
	if c <= rune3 {
		str[0] = byte(t3 | (c >> (2 * bitx)))
		str[1] = byte(tx | ((c >> (1 * bitx)) & maskx))
		str[2] = byte(tx | (c & maskx))
		return 3
	}

	/*
	 * four character sequence (21-bit value)
	 *     10000-1FFFFF => t4 tx tx tx
	 */
	str[0] = byte(t4 | (c >> (3 * bitx)))
	str[1] = byte(tx | ((c >> (2 * bitx)) & maskx))
	str[2] = byte(tx | ((c >> (1 * bitx)) & maskx))
	str[3] = byte(tx | (c & maskx))
	return 4
}
