// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package collate contains types for comparing and sorting Unicode strings
// according to a given collation order.  Package locale provides a high-level
// interface to collation. Users should typically use that package instead.
package collate

import (
	"bytes"
	"exp/norm"
)

// Level identifies the collation comparison level.
// The primary level corresponds to the basic sorting of text.
// The secondary level corresponds to accents and related linguistic elements.
// The tertiary level corresponds to casing and related concepts.
// The quaternary level is derived from the other levels by the
// various algorithms for handling variable elements.
type Level int

const (
	Primary Level = iota
	Secondary
	Tertiary
	Quaternary
	Identity
)

// AlternateHandling identifies the various ways in which variables are handled.
// A rune with a primary weight lower than the variable top is considered a
// variable. 
// See http://www.unicode.org/reports/tr10/#Variable_Weighting for details.
type AlternateHandling int

const (
	// AltNonIgnorable turns off special handling of variables.
	AltNonIgnorable AlternateHandling = iota

	// AltBlanked sets variables and all subsequent primary ignorables to be
	// ignorable at all levels. This is identical to removing all variables
	// and subsequent primary ignorables from the input.
	AltBlanked

	// AltShifted sets variables to be ignorable for levels one through three and
	// adds a fourth level based on the values of the ignored levels.
	AltShifted

	// AltShiftTrimmed is a slight variant of AltShifted that is used to
	// emulate POSIX.
	AltShiftTrimmed
)

// Collator provides functionality for comparing strings for a given
// collation order.
type Collator struct {
	// Strength sets the maximum level to use in comparison.
	Strength Level

	// Alternate specifies an alternative handling of variables.
	Alternate AlternateHandling

	// Backwards specifies the order of sorting at the secondary level.
	// This option exists predominantly to support reverse sorting of accents in French.
	Backwards bool

	// TODO: implement:
	// With HiraganaQuaternary enabled, Hiragana codepoints will get lower values
	// than all the other non-variable code points. Strength must be greater or
	// equal to Quaternary for this to take effect.
	HiraganaQuaternary bool

	// If CaseLevel is true, a level consisting only of case characteristics will
	// be inserted in front of the tertiary level.  To ignore accents but take
	// cases into account, set Strength to Primary and CaseLevel to true.
	CaseLevel bool

	// If Numeric is true, any sequence of decimal digits (category is Nd) is sorted
	// at a primary level with its numeric value.  For example, "A-21" < "A-123".
	Numeric bool

	f norm.Form

	t *table
}

// SetVariableTop sets all runes with primary strength less than the primary
// strength of r to be variable and thus affected by alternate handling.
func (c *Collator) SetVariableTop(r rune) {
	// TODO: implement
}

// Buffer holds reusable buffers that can be used during collation.
// Reusing a Buffer for the various calls that accept it may avoid
// unnecessary memory allocations.
type Buffer struct {
	// TODO: try various parameters and techniques, such as using
	// a chan of buffers for a pool.
	ba  [4096]byte
	wa  [512]weights
	key []byte
	ce  []weights
}

func (b *Buffer) init() {
	if b.ce == nil {
		b.ce = b.wa[:0]
		b.key = b.ba[:0]
	} else {
		b.ce = b.ce[:0]
	}
}

// ResetKeys clears the buffer used for generated keys. Calling ResetKeys
// invalidates keys previously obtained from Key or KeyFromString.
func (b *Buffer) ResetKeys() {
	b.ce = b.ce[:0]
	b.key = b.key[:0]
}

// Compare returns an integer comparing the two byte slices.
// The result will be 0 if a==b, -1 if a < b, and +1 if a > b.
// Compare calls ResetKeys, thereby invalidating keys
// previously generated using Key or KeyFromString using buf.
func (c *Collator) Compare(buf *Buffer, a, b []byte) int {
	// TODO: for now we simply compute keys and compare.  Once we
	// have good benchmarks, move to an implementation that works
	// incrementally for the majority of cases.
	// - Benchmark with long strings that only vary in modifiers.
	buf.ResetKeys()
	ka := c.Key(buf, a)
	kb := c.Key(buf, b)
	defer buf.ResetKeys()
	return bytes.Compare(ka, kb)
}

// CompareString returns an integer comparing the two strings.
// The result will be 0 if a==b, -1 if a < b, and +1 if a > b.
// CompareString calls ResetKeys, thereby invalidating keys
// previously generated using Key or KeyFromString using buf.
func (c *Collator) CompareString(buf *Buffer, a, b string) int {
	buf.ResetKeys()
	ka := c.KeyFromString(buf, a)
	kb := c.KeyFromString(buf, b)
	defer buf.ResetKeys()
	return bytes.Compare(ka, kb)
}

func (c *Collator) Prefix(buf *Buffer, s, prefix []byte) int {
	// iterate over s, track bytes consumed.
	return 0
}

// Key returns the collation key for str.
// Passing the buffer buf may avoid memory allocations.
// The returned slice will point to an allocation in Buffer and will remain
// valid until the next call to buf.ResetKeys().
func (c *Collator) Key(buf *Buffer, str []byte) []byte {
	// See http://www.unicode.org/reports/tr10/#Main_Algorithm for more details.
	buf.init()
	c.getColElems(buf, str)
	return c.key(buf, buf.ce)
}

// KeyFromString returns the collation key for str.
// Passing the buffer buf may avoid memory allocations.
// The returned slice will point to an allocation in Buffer and will retain
// valid until the next call to buf.ResetKeys().
func (c *Collator) KeyFromString(buf *Buffer, str string) []byte {
	// See http://www.unicode.org/reports/tr10/#Main_Algorithm for more details.
	buf.init()
	c.getColElemsString(buf, str)
	return c.key(buf, buf.ce)
}

func (c *Collator) key(buf *Buffer, w []weights) []byte {
	processWeights(c.Alternate, c.t.variableTop, w)
	kn := len(buf.key)
	c.keyFromElems(buf, w)
	return buf.key[kn:]
}

func (c *Collator) getColElems(buf *Buffer, str []byte) {
	i := c.iter()
	i.src.SetInput(c.f, str)
	for !i.done() {
		buf.ce = i.next(buf.ce)
	}
}

func (c *Collator) getColElemsString(buf *Buffer, str string) {
	i := c.iter()
	i.src.SetInputString(c.f, str)
	for !i.done() {
		buf.ce = i.next(buf.ce)
	}
}

type iter struct {
	src        norm.Iter
	ba         [1024]byte
	buf        []byte
	t          *table
	p          int
	minBufSize int
	_done, eof bool
}

func (c *Collator) iter() iter {
	i := iter{t: c.t, minBufSize: c.t.maxContractLen}
	i.buf = i.ba[:0]
	return i
}

func (i *iter) done() bool {
	return i._done
}

func (i *iter) next(ce []weights) []weights {
	if !i.eof && len(i.buf)-i.p < i.minBufSize {
		// replenish buffer
		n := copy(i.buf, i.buf[i.p:])
		n += i.src.Next(i.buf[n:cap(i.buf)])
		i.buf = i.buf[:n]
		i.p = 0
		i.eof = i.src.Done()
	}
	if i.p == len(i.buf) {
		i._done = true
		return ce
	}
	ce, sz := i.t.appendNext(ce, i.buf[i.p:])
	i.p += sz
	return ce
}

func appendPrimary(key []byte, p uint32) []byte {
	// Convert to variable length encoding; supports up to 23 bits.
	if p <= 0x7FFF {
		key = append(key, uint8(p>>8), uint8(p))
	} else {
		key = append(key, uint8(p>>16)|0x80, uint8(p>>8), uint8(p))
	}
	return key
}

// keyFromElems converts the weights ws to a compact sequence of bytes.
// The result will be appended to the byte buffer in buf.
func (c *Collator) keyFromElems(buf *Buffer, ws []weights) {
	for _, v := range ws {
		if w := v.primary; w > 0 {
			buf.key = appendPrimary(buf.key, w)
		}
	}
	if Secondary <= c.Strength {
		buf.key = append(buf.key, 0, 0)
		// TODO: we can use one 0 if we can guarantee that all non-zero weights are > 0xFF.
		if !c.Backwards {
			for _, v := range ws {
				if w := v.secondary; w > 0 {
					buf.key = append(buf.key, uint8(w>>8), uint8(w))
				}
			}
		} else {
			for i := len(ws) - 1; i >= 0; i-- {
				if w := ws[i].secondary; w > 0 {
					buf.key = append(buf.key, uint8(w>>8), uint8(w))
				}
			}
		}
	} else if c.CaseLevel {
		buf.key = append(buf.key, 0, 0)
	}
	if Tertiary <= c.Strength || c.CaseLevel {
		buf.key = append(buf.key, 0, 0)
		for _, v := range ws {
			if w := v.tertiary; w > 0 {
				buf.key = append(buf.key, w)
			}
		}
		// Derive the quaternary weights from the options and other levels.
		// Note that we represent maxQuaternary as 0xFF. The first byte of the
		// representation of a a primary weight is always smaller than 0xFF,
		// so using this single byte value will compare correctly.
		if Quaternary <= c.Strength {
			if c.Alternate == AltShiftTrimmed {
				lastNonFFFF := len(buf.key)
				buf.key = append(buf.key, 0)
				for _, v := range ws {
					if w := v.quaternary; w == maxQuaternary {
						buf.key = append(buf.key, 0xFF)
					} else if w > 0 {
						buf.key = appendPrimary(buf.key, w)
						lastNonFFFF = len(buf.key)
					}
				}
				buf.key = buf.key[:lastNonFFFF]
			} else {
				buf.key = append(buf.key, 0)
				for _, v := range ws {
					if w := v.quaternary; w == maxQuaternary {
						buf.key = append(buf.key, 0xFF)
					} else if w > 0 {
						buf.key = appendPrimary(buf.key, w)
					}
				}
			}
		}
	}
}

func processWeights(vw AlternateHandling, top uint32, wa []weights) {
	ignore := false
	switch vw {
	case AltShifted, AltShiftTrimmed:
		for i := range wa {
			if p := wa[i].primary; p <= top && p != 0 {
				wa[i] = weights{quaternary: p}
				ignore = true
			} else if p == 0 {
				if ignore {
					wa[i] = weights{}
				} else if wa[i].tertiary != 0 {
					wa[i].quaternary = maxQuaternary
				}
			} else {
				wa[i].quaternary = maxQuaternary
				ignore = false
			}
		}
	case AltBlanked:
		for i := range wa {
			if p := wa[i].primary; p <= top && (ignore || p != 0) {
				wa[i] = weights{}
				ignore = true
			} else {
				ignore = false
			}
		}
	}
}
