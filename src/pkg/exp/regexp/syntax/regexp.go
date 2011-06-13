// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package syntax parses regular expressions into syntax trees.
// WORK IN PROGRESS.
package syntax

// Note to implementers:
// In this package, re is always a *Regexp and r is always a rune.

import (
	"bytes"
	"strconv"
	"strings"
	"unicode"
)

// A Regexp is a node in a regular expression syntax tree.
type Regexp struct {
	Op       Op // operator
	Flags    Flags
	Sub      []*Regexp  // subexpressions, if any
	Sub0     [1]*Regexp // storage for short Sub
	Rune     []int      // matched runes, for OpLiteral, OpCharClass
	Rune0    [2]int     // storage for short Rune
	Min, Max int        // min, max for OpRepeat
	Cap      int        // capturing index, for OpCapture
	Name     string     // capturing name, for OpCapture
}

// An Op is a single regular expression operator.
type Op uint8

// Operators are listed in precedence order, tightest binding to weakest.

const (
	OpNoMatch        Op = 1 + iota // matches no strings
	OpEmptyMatch                   // matches empty string
	OpLiteral                      // matches Runes sequence
	OpCharClass                    // matches Runes interpreted as range pair list
	OpAnyCharNotNL                 // matches any character
	OpAnyChar                      // matches any character
	OpBeginLine                    // matches empty string at beginning of line
	OpEndLine                      // matches empty string at end of line
	OpBeginText                    // matches empty string at beginning of text
	OpEndText                      // matches empty string at end of text
	OpWordBoundary                 // matches word boundary `\b`
	OpNoWordBoundary               // matches word non-boundary `\B`
	OpCapture                      // capturing subexpression with index Cap, optional name Name
	OpStar                         // matches Sub[0] zero or more times
	OpPlus                         // matches Sub[0] one or more times
	OpQuest                        // matches Sub[0] zero or one times
	OpRepeat                       // matches Sub[0] at least Min times, at most Max (Max == -1 is no limit)
	OpConcat                       // matches concatenation of Subs
	OpAlternate                    // matches alternation of Subs
)

const opPseudo Op = 128 // where pseudo-ops start

// writeRegexp writes the Perl syntax for the regular expression re to b.
func writeRegexp(b *bytes.Buffer, re *Regexp) {
	switch re.Op {
	default:
		b.WriteString("<invalid op" + strconv.Itoa(int(re.Op)) + ">")
	case OpNoMatch:
		b.WriteString(`[^\x00-\x{10FFFF}]`)
	case OpEmptyMatch:
		b.WriteString(`(?:)`)
	case OpLiteral:
		for _, r := range re.Rune {
			escape(b, r, false)
		}
	case OpCharClass:
		if len(re.Rune)%2 != 0 {
			b.WriteString(`[invalid char class]`)
			break
		}
		b.WriteRune('[')
		if len(re.Rune) > 0 && re.Rune[0] == 0 && re.Rune[len(re.Rune)-1] == unicode.MaxRune {
			// Contains 0 and MaxRune.  Probably a negated class.
			// Print the gaps.
			b.WriteRune('^')
			for i := 1; i < len(re.Rune)-1; i += 2 {
				lo, hi := re.Rune[i]+1, re.Rune[i+1]-1
				escape(b, lo, lo == '-')
				if lo != hi {
					b.WriteRune('-')
					escape(b, hi, hi == '-')
				}
			}
		} else {
			for i := 0; i < len(re.Rune); i += 2 {
				lo, hi := re.Rune[i], re.Rune[i+1]
				escape(b, lo, lo == '-')
				if lo != hi {
					b.WriteRune('-')
					escape(b, hi, hi == '-')
				}
			}
		}
		b.WriteRune(']')
	case OpAnyCharNotNL:
		b.WriteString(`[^\n]`)
	case OpAnyChar:
		b.WriteRune('.')
	case OpBeginLine:
		b.WriteRune('^')
	case OpEndLine:
		b.WriteRune('$')
	case OpBeginText:
		b.WriteString(`\A`)
	case OpEndText:
		b.WriteString(`\z`)
	case OpWordBoundary:
		b.WriteString(`\b`)
	case OpNoWordBoundary:
		b.WriteString(`\B`)
	case OpCapture:
		if re.Name != "" {
			b.WriteString(`(?P<`)
			b.WriteString(re.Name)
			b.WriteRune('>')
		} else {
			b.WriteRune('(')
		}
		writeRegexp(b, re.Sub[0])
		b.WriteRune(')')
	case OpStar, OpPlus, OpQuest, OpRepeat:
		if sub := re.Sub[0]; sub.Op > OpCapture {
			b.WriteString(`(?:`)
			writeRegexp(b, sub)
			b.WriteString(`)`)
		} else {
			writeRegexp(b, sub)
		}
		switch re.Op {
		case OpStar:
			b.WriteRune('*')
		case OpPlus:
			b.WriteRune('+')
		case OpQuest:
			b.WriteRune('?')
		case OpRepeat:
			b.WriteRune('{')
			b.WriteString(strconv.Itoa(re.Min))
			if re.Max != re.Min {
				b.WriteRune(',')
				if re.Max >= 0 {
					b.WriteString(strconv.Itoa(re.Max))
				}
			}
			b.WriteRune('}')
		}
	case OpConcat:
		for _, sub := range re.Sub {
			if sub.Op == OpAlternate {
				b.WriteString(`(?:`)
				writeRegexp(b, sub)
				b.WriteString(`)`)
			} else {
				writeRegexp(b, sub)
			}
		}
	case OpAlternate:
		for i, sub := range re.Sub {
			if i > 0 {
				b.WriteRune('|')
			}
			writeRegexp(b, sub)
		}
	}
}

func (re *Regexp) String() string {
	var b bytes.Buffer
	writeRegexp(&b, re)
	return b.String()
}

const meta = `\.+*?()|[]{}^$`

func escape(b *bytes.Buffer, r int, force bool) {
	if unicode.IsPrint(r) {
		if strings.IndexRune(meta, r) >= 0 || force {
			b.WriteRune('\\')
		}
		b.WriteRune(r)
		return
	}

	switch r {
	case '\a':
		b.WriteString(`\a`)
	case '\f':
		b.WriteString(`\f`)
	case '\n':
		b.WriteString(`\n`)
	case '\r':
		b.WriteString(`\r`)
	case '\t':
		b.WriteString(`\t`)
	case '\v':
		b.WriteString(`\v`)
	default:
		b.WriteString(`\x{`)
		b.WriteString(strconv.Itob(r, 16))
		b.WriteString(`}`)
	}
}
