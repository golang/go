// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// Note to implementers:
// In this package, re is always a *Regexp and r is always a rune.

import (
	"slices"
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
	Rune     []rune     // matched runes, for OpLiteral, OpCharClass
	Rune0    [2]rune    // storage for short Rune
	Min, Max int        // min, max for OpRepeat
	Cap      int        // capturing index, for OpCapture
	Name     string     // capturing name, for OpCapture
}

//go:generate stringer -type Op -trimprefix Op

// An Op is a single regular expression operator.
type Op uint8

// Operators are listed in precedence order, tightest binding to weakest.
// Character class operators are listed simplest to most complex
// (OpLiteral, OpCharClass, OpAnyCharNotNL, OpAnyChar).

const (
	OpNoMatch        Op = 1 + iota // matches no strings
	OpEmptyMatch                   // matches empty string
	OpLiteral                      // matches Runes sequence
	OpCharClass                    // matches Runes interpreted as range pair list
	OpAnyCharNotNL                 // matches any character except newline
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

// Equal reports whether x and y have identical structure.
func (x *Regexp) Equal(y *Regexp) bool {
	if x == nil || y == nil {
		return x == y
	}
	if x.Op != y.Op {
		return false
	}
	switch x.Op {
	case OpEndText:
		// The parse flags remember whether this is \z or \Z.
		if x.Flags&WasDollar != y.Flags&WasDollar {
			return false
		}

	case OpLiteral, OpCharClass:
		return slices.Equal(x.Rune, y.Rune)

	case OpAlternate, OpConcat:
		return slices.EqualFunc(x.Sub, y.Sub, (*Regexp).Equal)

	case OpStar, OpPlus, OpQuest:
		if x.Flags&NonGreedy != y.Flags&NonGreedy || !x.Sub[0].Equal(y.Sub[0]) {
			return false
		}

	case OpRepeat:
		if x.Flags&NonGreedy != y.Flags&NonGreedy || x.Min != y.Min || x.Max != y.Max || !x.Sub[0].Equal(y.Sub[0]) {
			return false
		}

	case OpCapture:
		if x.Cap != y.Cap || x.Name != y.Name || !x.Sub[0].Equal(y.Sub[0]) {
			return false
		}
	}
	return true
}

// printFlags is a bit set indicating which flags (including non-capturing parens) to print around a regexp.
type printFlags uint8

const (
	flagI    printFlags = 1 << iota // (?i:
	flagM                           // (?m:
	flagS                           // (?s:
	flagOff                         // )
	flagPrec                        // (?: )
	negShift = 5                    // flagI<<negShift is (?-i:
)

// addSpan enables the flags f around start..last,
// by setting flags[start] = f and flags[last] = flagOff.
func addSpan(start, last *Regexp, f printFlags, flags *map[*Regexp]printFlags) {
	if *flags == nil {
		*flags = make(map[*Regexp]printFlags)
	}
	(*flags)[start] = f
	(*flags)[last] |= flagOff // maybe start==last
}

// calcFlags calculates the flags to print around each subexpression in re,
// storing that information in (*flags)[sub] for each affected subexpression.
// The first time an entry needs to be written to *flags, calcFlags allocates the map.
// calcFlags also calculates the flags that must be active or can't be active
// around re and returns those flags.
func calcFlags(re *Regexp, flags *map[*Regexp]printFlags) (must, cant printFlags) {
	switch re.Op {
	default:
		return 0, 0

	case OpLiteral:
		// If literal is fold-sensitive, return (flagI, 0) or (0, flagI)
		// according to whether (?i) is active.
		// If literal is not fold-sensitive, return 0, 0.
		for _, r := range re.Rune {
			if minFold <= r && r <= maxFold && unicode.SimpleFold(r) != r {
				if re.Flags&FoldCase != 0 {
					return flagI, 0
				} else {
					return 0, flagI
				}
			}
		}
		return 0, 0

	case OpCharClass:
		// If literal is fold-sensitive, return 0, flagI - (?i) has been compiled out.
		// If literal is not fold-sensitive, return 0, 0.
		return calcFlagsI(re)

	case OpAnyCharNotNL: // (?-s).
		return 0, flagS

	case OpAnyChar: // (?s).
		return flagS, 0

	case OpBeginLine, OpEndLine: // (?m)^ (?m)$
		return flagM, 0

	case OpEndText:
		if re.Flags&WasDollar != 0 { // (?-m)$
			return 0, flagM
		}
		return 0, 0

	case OpCapture, OpStar, OpPlus, OpQuest, OpRepeat:
		return calcFlags(re.Sub[0], flags)

	case OpConcat, OpAlternate:
		// Gather the must and cant for each subexpression.
		// When we find a conflicting subexpression, insert the necessary
		// flags around the previously identified span and start over.
		var must, cant, allCant printFlags
		start := 0
		last := 0
		did := false
		for i, sub := range re.Sub {
			subMust, subCant := calcFlags(sub, flags)
			if must&subCant != 0 || subMust&cant != 0 {
				if must != 0 {
					addSpan(re.Sub[start], re.Sub[last], must, flags)
				}
				must = 0
				cant = 0
				start = i
				did = true
			}
			must |= subMust
			cant |= subCant
			allCant |= subCant
			if subMust != 0 {
				last = i
			}
			if must == 0 && start == i {
				start++
			}
		}
		if !did {
			// No conflicts: pass the accumulated must and cant upward.
			return must, cant
		}
		if must != 0 {
			// Conflicts found; need to finish final span.
			addSpan(re.Sub[start], re.Sub[last], must, flags)
		}
		return 0, allCant
	}
}

func calcFlagsI(re *Regexp) (must, cant printFlags) {
	if len(re.Rune) < 2 {
		return 0, 0
	}

	maxRange := min(maxFold, re.Rune[len(re.Rune)-1])
	pre := rune(minFold)
	inside, outside := 0, int(maxFold-maxRange)
	checkInRange := true

	// If last range dominates the whole range,
	// we can check outside the range.
	lastRange := max(0, maxRange-re.Rune[len(re.Rune)-2])
	if lastRange > rune(outside)+re.Rune[len(re.Rune)-2] {
		checkInRange = false
		goto check
	}

	// If the range from last rune to maxFold dominates,
	// we can check inside the range.
	if outside > int(re.Rune[len(re.Rune)-1]) {
		goto check
	}

	// 2 conditions above should catch most cases.
	// If not, do a slow calculation
	for i := 0; i < len(re.Rune); i += 2 {
		lo := max(minFold, re.Rune[i])
		hi := min(maxFold, re.Rune[i+1])
		if lo > hi {
			continue
		}

		inside += int(hi - lo + 1)
		outside += int(lo - pre)
		pre = max(minFold, hi+1)
	}

	checkInRange = inside < outside
check:
	if checkInRange {
		for i := 0; i < len(re.Rune); i += 2 {
			lo := max(minFold, re.Rune[i])
			hi := min(maxFold, re.Rune[i+1])
			for r := lo; r <= hi; r++ {
				for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
					if !(lo <= f && f <= hi) && !inCharClass(f, re.Rune) {
						return 0, flagI
					}
				}
			}
		}

		return 0, 0
	}

	// Check characters outside the defined range
	pre = minFold
	for i := 0; i < len(re.Rune); i += 2 {
		lo := re.Rune[i]
		hi := min(maxFold, re.Rune[i+1])
		// Between `pre` and `lo` (exclusive)
		for r := pre; r < lo; r++ {
			for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
				if inCharClass(f, re.Rune) {
					return 0, flagI
				}
			}
		}
		pre = max(minFold, hi+1)
	}

	// Check characters between `pre` and `maxFold`
	for r := pre; r <= maxFold; r++ {
		for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
			if inCharClass(f, re.Rune) {
				return 0, flagI
			}
		}
	}

	return 0, 0
}

// writeRegexp writes the Perl syntax for the regular expression re to b.
func writeRegexp(b *strings.Builder, re *Regexp, f printFlags, flags map[*Regexp]printFlags) {
	f |= flags[re]
	if f&flagPrec != 0 && f&^(flagOff|flagPrec) != 0 && f&flagOff != 0 {
		// flagPrec is redundant with other flags being added and terminated
		f &^= flagPrec
	}
	if f&^(flagOff|flagPrec) != 0 {
		b.WriteString(`(?`)
		if f&flagI != 0 {
			b.WriteString(`i`)
		}
		if f&flagM != 0 {
			b.WriteString(`m`)
		}
		if f&flagS != 0 {
			b.WriteString(`s`)
		}
		if f&((flagM|flagS)<<negShift) != 0 {
			b.WriteString(`-`)
			if f&(flagM<<negShift) != 0 {
				b.WriteString(`m`)
			}
			if f&(flagS<<negShift) != 0 {
				b.WriteString(`s`)
			}
		}
		b.WriteString(`:`)
	}
	if f&flagOff != 0 {
		defer b.WriteString(`)`)
	}
	if f&flagPrec != 0 {
		b.WriteString(`(?:`)
		defer b.WriteString(`)`)
	}

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
		if len(re.Rune) == 0 {
			b.WriteString(`^\x00-\x{10FFFF}`)
		} else if re.Rune[0] == 0 && re.Rune[len(re.Rune)-1] == unicode.MaxRune && len(re.Rune) > 2 {
			// Contains 0 and MaxRune. Probably a negated class.
			// Print the gaps.
			b.WriteRune('^')
			for i := 1; i < len(re.Rune)-1; i += 2 {
				lo, hi := re.Rune[i]+1, re.Rune[i+1]-1
				escape(b, lo, lo == '-')
				if lo != hi {
					if hi != lo+1 {
						b.WriteRune('-')
					}
					escape(b, hi, hi == '-')
				}
			}
		} else {
			for i := 0; i < len(re.Rune); i += 2 {
				lo, hi := re.Rune[i], re.Rune[i+1]
				escape(b, lo, lo == '-')
				if lo != hi {
					if hi != lo+1 {
						b.WriteRune('-')
					}
					escape(b, hi, hi == '-')
				}
			}
		}
		b.WriteRune(']')
	case OpAnyCharNotNL, OpAnyChar:
		b.WriteString(`.`)
	case OpBeginLine:
		b.WriteString(`^`)
	case OpEndLine:
		b.WriteString(`$`)
	case OpBeginText:
		b.WriteString(`\A`)
	case OpEndText:
		if re.Flags&WasDollar != 0 {
			b.WriteString(`$`)
		} else {
			b.WriteString(`\z`)
		}
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
		if re.Sub[0].Op != OpEmptyMatch {
			writeRegexp(b, re.Sub[0], flags[re.Sub[0]], flags)
		}
		b.WriteRune(')')
	case OpStar, OpPlus, OpQuest, OpRepeat:
		p := printFlags(0)
		sub := re.Sub[0]
		if sub.Op > OpCapture || sub.Op == OpLiteral && len(sub.Rune) > 1 {
			p = flagPrec
		}
		writeRegexp(b, sub, p, flags)

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
		if re.Flags&NonGreedy != 0 {
			b.WriteRune('?')
		}
	case OpConcat:
		for _, sub := range re.Sub {
			p := printFlags(0)
			if sub.Op == OpAlternate {
				p = flagPrec
			}
			writeRegexp(b, sub, p, flags)
		}
	case OpAlternate:
		for i, sub := range re.Sub {
			if i > 0 {
				b.WriteRune('|')
			}
			writeRegexp(b, sub, 0, flags)
		}
	}
}

func (re *Regexp) String() string {
	var b strings.Builder
	var flags map[*Regexp]printFlags
	must, cant := calcFlags(re, &flags)
	must |= (cant &^ flagI) << negShift
	if must != 0 {
		must |= flagOff
	}
	writeRegexp(&b, re, must, flags)
	return b.String()
}

const meta = `\.+*?()|[]{}^$`

func escape(b *strings.Builder, r rune, force bool) {
	if unicode.IsPrint(r) {
		if strings.ContainsRune(meta, r) || force {
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
		if r < 0x100 {
			b.WriteString(`\x`)
			s := strconv.FormatInt(int64(r), 16)
			if len(s) == 1 {
				b.WriteRune('0')
			}
			b.WriteString(s)
			break
		}
		b.WriteString(`\x{`)
		b.WriteString(strconv.FormatInt(int64(r), 16))
		b.WriteString(`}`)
	}
}

// MaxCap walks the regexp to find the maximum capture index.
func (re *Regexp) MaxCap() int {
	m := 0
	if re.Op == OpCapture {
		m = re.Cap
	}
	for _, sub := range re.Sub {
		if n := sub.MaxCap(); m < n {
			m = n
		}
	}
	return m
}

// CapNames walks the regexp to find the names of capturing groups.
func (re *Regexp) CapNames() []string {
	names := make([]string, re.MaxCap()+1)
	re.capNames(names)
	return names
}

func (re *Regexp) capNames(names []string) {
	if re.Op == OpCapture {
		names[re.Cap] = re.Name
	}
	for _, sub := range re.Sub {
		sub.capNames(names)
	}
}
