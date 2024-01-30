// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cases

import "golang.org/x/text/transform"

// A context is used for iterating over source bytes, fetching case info and
// writing to a destination buffer.
//
// Casing operations may need more than one rune of context to decide how a rune
// should be cased. Casing implementations should call checkpoint on context
// whenever it is known to be safe to return the runes processed so far.
//
// It is recommended for implementations to not allow for more than 30 case
// ignorables as lookahead (analogous to the limit in norm) and to use state if
// unbounded lookahead is needed for cased runes.
type context struct {
	dst, src []byte
	atEOF    bool

	pDst int // pDst points past the last written rune in dst.
	pSrc int // pSrc points to the start of the currently scanned rune.

	// checkpoints safe to return in Transform, where nDst <= pDst and nSrc <= pSrc.
	nDst, nSrc int
	err        error

	sz   int  // size of current rune
	info info // case information of currently scanned rune

	// State preserved across calls to Transform.
	isMidWord bool // false if next cased letter needs to be title-cased.
}

func (c *context) Reset() {
	c.isMidWord = false
}

// ret returns the return values for the Transform method. It checks whether
// there were insufficient bytes in src to complete and introduces an error
// accordingly, if necessary.
func (c *context) ret() (nDst, nSrc int, err error) {
	if c.err != nil || c.nSrc == len(c.src) {
		return c.nDst, c.nSrc, c.err
	}
	// This point is only reached by mappers if there was no short destination
	// buffer. This means that the source buffer was exhausted and that c.sz was
	// set to 0 by next.
	if c.atEOF && c.pSrc == len(c.src) {
		return c.pDst, c.pSrc, nil
	}
	return c.nDst, c.nSrc, transform.ErrShortSrc
}

// retSpan returns the return values for the Span method. It checks whether
// there were insufficient bytes in src to complete and introduces an error
// accordingly, if necessary.
func (c *context) retSpan() (n int, err error) {
	_, nSrc, err := c.ret()
	return nSrc, err
}

// checkpoint sets the return value buffer points for Transform to the current
// positions.
func (c *context) checkpoint() {
	if c.err == nil {
		c.nDst, c.nSrc = c.pDst, c.pSrc+c.sz
	}
}

// unreadRune causes the last rune read by next to be reread on the next
// invocation of next. Only one unreadRune may be called after a call to next.
func (c *context) unreadRune() {
	c.sz = 0
}

func (c *context) next() bool {
	c.pSrc += c.sz
	if c.pSrc == len(c.src) || c.err != nil {
		c.info, c.sz = 0, 0
		return false
	}
	v, sz := trie.lookup(c.src[c.pSrc:])
	c.info, c.sz = info(v), sz
	if c.sz == 0 {
		if c.atEOF {
			// A zero size means we have an incomplete rune. If we are atEOF,
			// this means it is an illegal rune, which we will consume one
			// byte at a time.
			c.sz = 1
		} else {
			c.err = transform.ErrShortSrc
			return false
		}
	}
	return true
}

// writeBytes adds bytes to dst.
func (c *context) writeBytes(b []byte) bool {
	if len(c.dst)-c.pDst < len(b) {
		c.err = transform.ErrShortDst
		return false
	}
	// This loop is faster than using copy.
	for _, ch := range b {
		c.dst[c.pDst] = ch
		c.pDst++
	}
	return true
}

// writeString writes the given string to dst.
func (c *context) writeString(s string) bool {
	if len(c.dst)-c.pDst < len(s) {
		c.err = transform.ErrShortDst
		return false
	}
	// This loop is faster than using copy.
	for i := 0; i < len(s); i++ {
		c.dst[c.pDst] = s[i]
		c.pDst++
	}
	return true
}

// copy writes the current rune to dst.
func (c *context) copy() bool {
	return c.writeBytes(c.src[c.pSrc : c.pSrc+c.sz])
}

// copyXOR copies the current rune to dst and modifies it by applying the XOR
// pattern of the case info. It is the responsibility of the caller to ensure
// that this is a rune with a XOR pattern defined.
func (c *context) copyXOR() bool {
	if !c.copy() {
		return false
	}
	if c.info&xorIndexBit == 0 {
		// Fast path for 6-bit XOR pattern, which covers most cases.
		c.dst[c.pDst-1] ^= byte(c.info >> xorShift)
	} else {
		// Interpret XOR bits as an index.
		// TODO: test performance for unrolling this loop. Verify that we have
		// at least two bytes and at most three.
		idx := c.info >> xorShift
		for p := c.pDst - 1; ; p-- {
			c.dst[p] ^= xorData[idx]
			idx--
			if xorData[idx] == 0 {
				break
			}
		}
	}
	return true
}

// hasPrefix returns true if src[pSrc:] starts with the given string.
func (c *context) hasPrefix(s string) bool {
	b := c.src[c.pSrc:]
	if len(b) < len(s) {
		return false
	}
	for i, c := range b[:len(s)] {
		if c != s[i] {
			return false
		}
	}
	return true
}

// caseType returns an info with only the case bits, normalized to either
// cLower, cUpper, cTitle or cUncased.
func (c *context) caseType() info {
	cm := c.info & 0x7
	if cm < 4 {
		return cm
	}
	if cm >= cXORCase {
		// xor the last bit of the rune with the case type bits.
		b := c.src[c.pSrc+c.sz-1]
		return info(b&1) ^ cm&0x3
	}
	if cm == cIgnorableCased {
		return cLower
	}
	return cUncased
}

// lower writes the lowercase version of the current rune to dst.
func lower(c *context) bool {
	ct := c.caseType()
	if c.info&hasMappingMask == 0 || ct == cLower {
		return c.copy()
	}
	if c.info&exceptionBit == 0 {
		return c.copyXOR()
	}
	e := exceptions[c.info>>exceptionShift:]
	offset := 2 + e[0]&lengthMask // size of header + fold string
	if nLower := (e[1] >> lengthBits) & lengthMask; nLower != noChange {
		return c.writeString(e[offset : offset+nLower])
	}
	return c.copy()
}

func isLower(c *context) bool {
	ct := c.caseType()
	if c.info&hasMappingMask == 0 || ct == cLower {
		return true
	}
	if c.info&exceptionBit == 0 {
		c.err = transform.ErrEndOfSpan
		return false
	}
	e := exceptions[c.info>>exceptionShift:]
	if nLower := (e[1] >> lengthBits) & lengthMask; nLower != noChange {
		c.err = transform.ErrEndOfSpan
		return false
	}
	return true
}

// upper writes the uppercase version of the current rune to dst.
func upper(c *context) bool {
	ct := c.caseType()
	if c.info&hasMappingMask == 0 || ct == cUpper {
		return c.copy()
	}
	if c.info&exceptionBit == 0 {
		return c.copyXOR()
	}
	e := exceptions[c.info>>exceptionShift:]
	offset := 2 + e[0]&lengthMask // size of header + fold string
	// Get length of first special case mapping.
	n := (e[1] >> lengthBits) & lengthMask
	if ct == cTitle {
		// The first special case mapping is for lower. Set n to the second.
		if n == noChange {
			n = 0
		}
		n, e = e[1]&lengthMask, e[n:]
	}
	if n != noChange {
		return c.writeString(e[offset : offset+n])
	}
	return c.copy()
}

// isUpper writes the isUppercase version of the current rune to dst.
func isUpper(c *context) bool {
	ct := c.caseType()
	if c.info&hasMappingMask == 0 || ct == cUpper {
		return true
	}
	if c.info&exceptionBit == 0 {
		c.err = transform.ErrEndOfSpan
		return false
	}
	e := exceptions[c.info>>exceptionShift:]
	// Get length of first special case mapping.
	n := (e[1] >> lengthBits) & lengthMask
	if ct == cTitle {
		n = e[1] & lengthMask
	}
	if n != noChange {
		c.err = transform.ErrEndOfSpan
		return false
	}
	return true
}

// title writes the title case version of the current rune to dst.
func title(c *context) bool {
	ct := c.caseType()
	if c.info&hasMappingMask == 0 || ct == cTitle {
		return c.copy()
	}
	if c.info&exceptionBit == 0 {
		if ct == cLower {
			return c.copyXOR()
		}
		return c.copy()
	}
	// Get the exception data.
	e := exceptions[c.info>>exceptionShift:]
	offset := 2 + e[0]&lengthMask // size of header + fold string

	nFirst := (e[1] >> lengthBits) & lengthMask
	if nTitle := e[1] & lengthMask; nTitle != noChange {
		if nFirst != noChange {
			e = e[nFirst:]
		}
		return c.writeString(e[offset : offset+nTitle])
	}
	if ct == cLower && nFirst != noChange {
		// Use the uppercase version instead.
		return c.writeString(e[offset : offset+nFirst])
	}
	// Already in correct case.
	return c.copy()
}

// isTitle reports whether the current rune is in title case.
func isTitle(c *context) bool {
	ct := c.caseType()
	if c.info&hasMappingMask == 0 || ct == cTitle {
		return true
	}
	if c.info&exceptionBit == 0 {
		if ct == cLower {
			c.err = transform.ErrEndOfSpan
			return false
		}
		return true
	}
	// Get the exception data.
	e := exceptions[c.info>>exceptionShift:]
	if nTitle := e[1] & lengthMask; nTitle != noChange {
		c.err = transform.ErrEndOfSpan
		return false
	}
	nFirst := (e[1] >> lengthBits) & lengthMask
	if ct == cLower && nFirst != noChange {
		c.err = transform.ErrEndOfSpan
		return false
	}
	return true
}

// foldFull writes the foldFull version of the current rune to dst.
func foldFull(c *context) bool {
	if c.info&hasMappingMask == 0 {
		return c.copy()
	}
	ct := c.caseType()
	if c.info&exceptionBit == 0 {
		if ct != cLower || c.info&inverseFoldBit != 0 {
			return c.copyXOR()
		}
		return c.copy()
	}
	e := exceptions[c.info>>exceptionShift:]
	n := e[0] & lengthMask
	if n == 0 {
		if ct == cLower {
			return c.copy()
		}
		n = (e[1] >> lengthBits) & lengthMask
	}
	return c.writeString(e[2 : 2+n])
}

// isFoldFull reports whether the current run is mapped to foldFull
func isFoldFull(c *context) bool {
	if c.info&hasMappingMask == 0 {
		return true
	}
	ct := c.caseType()
	if c.info&exceptionBit == 0 {
		if ct != cLower || c.info&inverseFoldBit != 0 {
			c.err = transform.ErrEndOfSpan
			return false
		}
		return true
	}
	e := exceptions[c.info>>exceptionShift:]
	n := e[0] & lengthMask
	if n == 0 && ct == cLower {
		return true
	}
	c.err = transform.ErrEndOfSpan
	return false
}
