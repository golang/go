// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bidirule implements the Bidi Rule defined by RFC 5893.
//
// This package is under development. The API may change without notice and
// without preserving backward compatibility.
package bidirule

import (
	"errors"
	"unicode/utf8"

	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/bidi"
)

// This file contains an implementation of RFC 5893: Right-to-Left Scripts for
// Internationalized Domain Names for Applications (IDNA)
//
// A label is an individual component of a domain name.  Labels are usually
// shown separated by dots; for example, the domain name "www.example.com" is
// composed of three labels: "www", "example", and "com".
//
// An RTL label is a label that contains at least one character of class R, AL,
// or AN. An LTR label is any label that is not an RTL label.
//
// A "Bidi domain name" is a domain name that contains at least one RTL label.
//
//  The following guarantees can be made based on the above:
//
//  o  In a domain name consisting of only labels that satisfy the rule,
//     the requirements of Section 3 are satisfied.  Note that even LTR
//     labels and pure ASCII labels have to be tested.
//
//  o  In a domain name consisting of only LDH labels (as defined in the
//     Definitions document [RFC5890]) and labels that satisfy the rule,
//     the requirements of Section 3 are satisfied as long as a label
//     that starts with an ASCII digit does not come after a
//     right-to-left label.
//
//  No guarantee is given for other combinations.

// ErrInvalid indicates a label is invalid according to the Bidi Rule.
var ErrInvalid = errors.New("bidirule: failed Bidi Rule")

type ruleState uint8

const (
	ruleInitial ruleState = iota
	ruleLTR
	ruleLTRFinal
	ruleRTL
	ruleRTLFinal
	ruleInvalid
)

type ruleTransition struct {
	next ruleState
	mask uint16
}

var transitions = [...][2]ruleTransition{
	// [2.1] The first character must be a character with Bidi property L, R, or
	// AL. If it has the R or AL property, it is an RTL label; if it has the L
	// property, it is an LTR label.
	ruleInitial: {
		{ruleLTRFinal, 1 << bidi.L},
		{ruleRTLFinal, 1<<bidi.R | 1<<bidi.AL},
	},
	ruleRTL: {
		// [2.3] In an RTL label, the end of the label must be a character with
		// Bidi property R, AL, EN, or AN, followed by zero or more characters
		// with Bidi property NSM.
		{ruleRTLFinal, 1<<bidi.R | 1<<bidi.AL | 1<<bidi.EN | 1<<bidi.AN},

		// [2.2] In an RTL label, only characters with the Bidi properties R,
		// AL, AN, EN, ES, CS, ET, ON, BN, or NSM are allowed.
		// We exclude the entries from [2.3]
		{ruleRTL, 1<<bidi.ES | 1<<bidi.CS | 1<<bidi.ET | 1<<bidi.ON | 1<<bidi.BN | 1<<bidi.NSM},
	},
	ruleRTLFinal: {
		// [2.3] In an RTL label, the end of the label must be a character with
		// Bidi property R, AL, EN, or AN, followed by zero or more characters
		// with Bidi property NSM.
		{ruleRTLFinal, 1<<bidi.R | 1<<bidi.AL | 1<<bidi.EN | 1<<bidi.AN | 1<<bidi.NSM},

		// [2.2] In an RTL label, only characters with the Bidi properties R,
		// AL, AN, EN, ES, CS, ET, ON, BN, or NSM are allowed.
		// We exclude the entries from [2.3] and NSM.
		{ruleRTL, 1<<bidi.ES | 1<<bidi.CS | 1<<bidi.ET | 1<<bidi.ON | 1<<bidi.BN},
	},
	ruleLTR: {
		// [2.6] In an LTR label, the end of the label must be a character with
		// Bidi property L or EN, followed by zero or more characters with Bidi
		// property NSM.
		{ruleLTRFinal, 1<<bidi.L | 1<<bidi.EN},

		// [2.5] In an LTR label, only characters with the Bidi properties L,
		// EN, ES, CS, ET, ON, BN, or NSM are allowed.
		// We exclude the entries from [2.6].
		{ruleLTR, 1<<bidi.ES | 1<<bidi.CS | 1<<bidi.ET | 1<<bidi.ON | 1<<bidi.BN | 1<<bidi.NSM},
	},
	ruleLTRFinal: {
		// [2.6] In an LTR label, the end of the label must be a character with
		// Bidi property L or EN, followed by zero or more characters with Bidi
		// property NSM.
		{ruleLTRFinal, 1<<bidi.L | 1<<bidi.EN | 1<<bidi.NSM},

		// [2.5] In an LTR label, only characters with the Bidi properties L,
		// EN, ES, CS, ET, ON, BN, or NSM are allowed.
		// We exclude the entries from [2.6].
		{ruleLTR, 1<<bidi.ES | 1<<bidi.CS | 1<<bidi.ET | 1<<bidi.ON | 1<<bidi.BN},
	},
	ruleInvalid: {
		{ruleInvalid, 0},
		{ruleInvalid, 0},
	},
}

// [2.4] In an RTL label, if an EN is present, no AN may be present, and
// vice versa.
const exclusiveRTL = uint16(1<<bidi.EN | 1<<bidi.AN)

// From RFC 5893
// An RTL label is a label that contains at least one character of type
// R, AL, or AN.
//
// An LTR label is any label that is not an RTL label.

// Direction reports the direction of the given label as defined by RFC 5893.
// The Bidi Rule does not have to be applied to labels of the category
// LeftToRight.
func Direction(b []byte) bidi.Direction {
	for i := 0; i < len(b); {
		e, sz := bidi.Lookup(b[i:])
		if sz == 0 {
			i++
		}
		c := e.Class()
		if c == bidi.R || c == bidi.AL || c == bidi.AN {
			return bidi.RightToLeft
		}
		i += sz
	}
	return bidi.LeftToRight
}

// DirectionString reports the direction of the given label as defined by RFC
// 5893. The Bidi Rule does not have to be applied to labels of the category
// LeftToRight.
func DirectionString(s string) bidi.Direction {
	for i := 0; i < len(s); {
		e, sz := bidi.LookupString(s[i:])
		if sz == 0 {
			i++
			continue
		}
		c := e.Class()
		if c == bidi.R || c == bidi.AL || c == bidi.AN {
			return bidi.RightToLeft
		}
		i += sz
	}
	return bidi.LeftToRight
}

// Valid reports whether b conforms to the BiDi rule.
func Valid(b []byte) bool {
	var t Transformer
	if n, ok := t.advance(b); !ok || n < len(b) {
		return false
	}
	return t.isFinal()
}

// ValidString reports whether s conforms to the BiDi rule.
func ValidString(s string) bool {
	var t Transformer
	if n, ok := t.advanceString(s); !ok || n < len(s) {
		return false
	}
	return t.isFinal()
}

// New returns a Transformer that verifies that input adheres to the Bidi Rule.
func New() *Transformer {
	return &Transformer{}
}

// Transformer implements transform.Transform.
type Transformer struct {
	state  ruleState
	hasRTL bool
	seen   uint16
}

// A rule can only be violated for "Bidi Domain names", meaning if one of the
// following categories has been observed.
func (t *Transformer) isRTL() bool {
	const isRTL = 1<<bidi.R | 1<<bidi.AL | 1<<bidi.AN
	return t.seen&isRTL != 0
}

// Reset implements transform.Transformer.
func (t *Transformer) Reset() { *t = Transformer{} }

// Transform implements transform.Transformer. This Transformer has state and
// needs to be reset between uses.
func (t *Transformer) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	if len(dst) < len(src) {
		src = src[:len(dst)]
		atEOF = false
		err = transform.ErrShortDst
	}
	n, err1 := t.Span(src, atEOF)
	copy(dst, src[:n])
	if err == nil || err1 != nil && err1 != transform.ErrShortSrc {
		err = err1
	}
	return n, n, err
}

// Span returns the first n bytes of src that conform to the Bidi rule.
func (t *Transformer) Span(src []byte, atEOF bool) (n int, err error) {
	if t.state == ruleInvalid && t.isRTL() {
		return 0, ErrInvalid
	}
	n, ok := t.advance(src)
	switch {
	case !ok:
		err = ErrInvalid
	case n < len(src):
		if !atEOF {
			err = transform.ErrShortSrc
			break
		}
		err = ErrInvalid
	case !t.isFinal():
		err = ErrInvalid
	}
	return n, err
}

// Precomputing the ASCII values decreases running time for the ASCII fast path
// by about 30%.
var asciiTable [128]bidi.Properties

func init() {
	for i := range asciiTable {
		p, _ := bidi.LookupRune(rune(i))
		asciiTable[i] = p
	}
}

func (t *Transformer) advance(s []byte) (n int, ok bool) {
	var e bidi.Properties
	var sz int
	for n < len(s) {
		if s[n] < utf8.RuneSelf {
			e, sz = asciiTable[s[n]], 1
		} else {
			e, sz = bidi.Lookup(s[n:])
			if sz <= 1 {
				if sz == 1 {
					// We always consider invalid UTF-8 to be invalid, even if
					// the string has not yet been determined to be RTL.
					// TODO: is this correct?
					return n, false
				}
				return n, true // incomplete UTF-8 encoding
			}
		}
		// TODO: using CompactClass would result in noticeable speedup.
		// See unicode/bidi/prop.go:Properties.CompactClass.
		c := uint16(1 << e.Class())
		t.seen |= c
		if t.seen&exclusiveRTL == exclusiveRTL {
			t.state = ruleInvalid
			return n, false
		}
		switch tr := transitions[t.state]; {
		case tr[0].mask&c != 0:
			t.state = tr[0].next
		case tr[1].mask&c != 0:
			t.state = tr[1].next
		default:
			t.state = ruleInvalid
			if t.isRTL() {
				return n, false
			}
		}
		n += sz
	}
	return n, true
}

func (t *Transformer) advanceString(s string) (n int, ok bool) {
	var e bidi.Properties
	var sz int
	for n < len(s) {
		if s[n] < utf8.RuneSelf {
			e, sz = asciiTable[s[n]], 1
		} else {
			e, sz = bidi.LookupString(s[n:])
			if sz <= 1 {
				if sz == 1 {
					return n, false // invalid UTF-8
				}
				return n, true // incomplete UTF-8 encoding
			}
		}
		// TODO: using CompactClass results in noticeable speedup.
		// See unicode/bidi/prop.go:Properties.CompactClass.
		c := uint16(1 << e.Class())
		t.seen |= c
		if t.seen&exclusiveRTL == exclusiveRTL {
			t.state = ruleInvalid
			return n, false
		}
		switch tr := transitions[t.state]; {
		case tr[0].mask&c != 0:
			t.state = tr[0].next
		case tr[1].mask&c != 0:
			t.state = tr[1].next
		default:
			t.state = ruleInvalid
			if t.isRTL() {
				return n, false
			}
		}
		n += sz
	}
	return n, true
}
