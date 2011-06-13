// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"os"
	"sort"
	"unicode"
	"utf8"
)

// An Error describes a failure to parse a regular expression
// and gives the offending expression.
type Error struct {
	Code ErrorCode
	Expr string
}

func (e *Error) String() string {
	return "error parsing regexp: " + e.Code.String() + ": `" + e.Expr + "`"
}

// An ErrorCode describes a failure to parse a regular expression.
type ErrorCode string

const (
	// Unexpected error
	ErrInternalError ErrorCode = "regexp/syntax: internal error"

	// Parse errors
	ErrInvalidCharClass      ErrorCode = "invalid character class"
	ErrInvalidCharRange      ErrorCode = "invalid character class range"
	ErrInvalidEscape         ErrorCode = "invalid escape sequence"
	ErrInvalidNamedCapture   ErrorCode = "invalid named capture"
	ErrInvalidPerlOp         ErrorCode = "invalid or unsupported Perl syntax"
	ErrInvalidRepeatOp       ErrorCode = "invalid nested repetition operator"
	ErrInvalidRepeatSize     ErrorCode = "invalid repeat count"
	ErrInvalidUTF8           ErrorCode = "invalid UTF-8"
	ErrMissingBracket        ErrorCode = "missing closing ]"
	ErrMissingParen          ErrorCode = "missing closing )"
	ErrMissingRepeatArgument ErrorCode = "missing argument to repetition operator"
	ErrTrailingBackslash     ErrorCode = "trailing backslash at end of expression"
)

func (e ErrorCode) String() string {
	return string(e)
}

// Flags control the behavior of the parser and record information about regexp context.
type Flags uint16

const (
	FoldCase      Flags = 1 << iota // case-insensitive match
	Literal                         // treat pattern as literal string
	ClassNL                         // allow character classes like [^a-z] and [[:space:]] to match newline
	DotNL                           // allow . to match newline
	OneLine                         // treat ^ and $ as only matching at beginning and end of text
	NonGreedy                       // make repetition operators default to non-greedy
	PerlX                           // allow Perl extensions
	UnicodeGroups                   // allow \p{Han}, \P{Han} for Unicode group and negation
	WasDollar                       // regexp OpEndText was $, not \z
	Simple                          // regexp contains no counted repetition

	MatchNL = ClassNL | DotNL

	Perl        = ClassNL | OneLine | PerlX | UnicodeGroups // as close to Perl as possible
	POSIX Flags = 0                                         // POSIX syntax
)

// Pseudo-ops for parsing stack.
const (
	opLeftParen = opPseudo + iota
	opVerticalBar
)

type parser struct {
	flags       Flags     // parse mode flags
	stack       []*Regexp // stack of parsed expressions
	numCap      int       // number of capturing groups seen
	wholeRegexp string
}

// Parse stack manipulation.

// push pushes the regexp re onto the parse stack and returns the regexp.
func (p *parser) push(re *Regexp) *Regexp {
	// TODO: automatic concatenation
	// TODO: turn character class into literal
	// TODO: compute simple

	p.stack = append(p.stack, re)
	return re
}

// newLiteral returns a new OpLiteral Regexp with the given flags
func newLiteral(r int, flags Flags) *Regexp {
	re := &Regexp{
		Op:    OpLiteral,
		Flags: flags,
	}
	re.Rune0[0] = r
	re.Rune = re.Rune0[:1]
	return re
}

// literal pushes a literal regexp for the rune r on the stack
// and returns that regexp.
func (p *parser) literal(r int) *Regexp {
	return p.push(newLiteral(r, p.flags))
}

// op pushes a regexp with the given op onto the stack
// and returns that regexp.
func (p *parser) op(op Op) *Regexp {
	return p.push(&Regexp{Op: op, Flags: p.flags})
}

// repeat replaces the top stack element with itself repeated
// according to op.
func (p *parser) repeat(op Op, opstr string) os.Error {
	n := len(p.stack)
	if n == 0 {
		return &Error{ErrMissingRepeatArgument, opstr}
	}
	sub := p.stack[n-1]
	re := &Regexp{
		Op: op,
	}
	re.Sub = re.Sub0[:1]
	re.Sub[0] = sub
	p.stack[n-1] = re
	return nil
}

// concat replaces the top of the stack (above the topmost '|' or '(') with its concatenation.
func (p *parser) concat() *Regexp {
	// TODO: Flatten concats.

	// Scan down to find pseudo-operator | or (.
	i := len(p.stack)
	for i > 0 && p.stack[i-1].Op < opPseudo {
		i--
	}
	sub := p.stack[i:]
	p.stack = p.stack[:i]

	var re *Regexp
	switch len(sub) {
	case 0:
		re = &Regexp{Op: OpEmptyMatch}
	case 1:
		re = sub[0]
	default:
		re = &Regexp{Op: OpConcat}
		re.Sub = append(re.Sub0[:0], sub...)
	}
	return p.push(re)
}

// alternate replaces the top of the stack (above the topmost '(') with its alternation.
func (p *parser) alternate() *Regexp {
	// TODO: Flatten alternates.

	// Scan down to find pseudo-operator (.
	// There are no | above (.
	i := len(p.stack)
	for i > 0 && p.stack[i-1].Op < opPseudo {
		i--
	}
	sub := p.stack[i:]
	p.stack = p.stack[:i]

	var re *Regexp
	switch len(sub) {
	case 0:
		re = &Regexp{Op: OpNoMatch}
	case 1:
		re = sub[0]
	default:
		re = &Regexp{Op: OpAlternate}
		re.Sub = append(re.Sub0[:0], sub...)
	}
	return p.push(re)
}

// Parsing.

func Parse(s string, flags Flags) (*Regexp, os.Error) {
	if flags&Literal != 0 {
		// Trivial parser for literal string.
		if err := checkUTF8(s); err != nil {
			return nil, err
		}
		re := &Regexp{
			Op:    OpLiteral,
			Flags: flags,
		}
		re.Rune = re.Rune0[:0] // use local storage for small strings
		for _, c := range s {
			if len(re.Rune) >= cap(re.Rune) {
				// string is too long to fit in Rune0.  let Go handle it
				re.Rune = []int(s)
				break
			}
			re.Rune = append(re.Rune, c)
		}
		return re, nil
	}

	// Otherwise, must do real work.
	var (
		p   parser
		err os.Error
		c   int
		op  Op
	)
	p.flags = flags
	p.wholeRegexp = s
	t := s
	for t != "" {
		switch t[0] {
		default:
			if c, t, err = nextRune(t); err != nil {
				return nil, err
			}
			p.literal(c)

		case '(':
			// TODO: Actual Perl flag parsing.
			if len(t) >= 3 && t[1] == '?' && t[2] == ':' {
				// non-capturing paren
				p.op(opLeftParen)
				t = t[3:]
				break
			}
			p.numCap++
			p.op(opLeftParen).Cap = p.numCap
			t = t[1:]
		case '|':
			p.concat()
			if err = p.parseVerticalBar(); err != nil {
				return nil, err
			}
			t = t[1:]
		case ')':
			if err = p.parseRightParen(); err != nil {
				return nil, err
			}
			t = t[1:]
		case '^':
			if p.flags&OneLine != 0 {
				p.op(OpBeginText)
			} else {
				p.op(OpBeginLine)
			}
			t = t[1:]
		case '$':
			if p.flags&OneLine != 0 {
				p.op(OpEndText).Flags |= WasDollar
			} else {
				p.op(OpEndLine)
			}
			t = t[1:]
		case '.':
			if p.flags&DotNL != 0 {
				p.op(OpAnyChar)
			} else {
				p.op(OpAnyCharNotNL)
			}
			t = t[1:]
		case '[':
			if t, err = p.parseClass(t); err != nil {
				return nil, err
			}
		case '*', '+', '?':
			switch t[0] {
			case '*':
				op = OpStar
			case '+':
				op = OpPlus
			case '?':
				op = OpQuest
			}
			// TODO: greedy
			if err = p.repeat(op, t[0:1]); err != nil {
				return nil, err
			}
			t = t[1:]
		case '{':
			return nil, os.NewError("repeat not implemented")
		case '\\':
			return nil, os.NewError("escape not implemented")
		}
	}

	p.concat()
	if p.swapVerticalBar() {
		// pop vertical bar
		p.stack = p.stack[:len(p.stack)-1]
	}
	p.alternate()

	n := len(p.stack)
	if n != 1 {
		return nil, &Error{ErrMissingParen, s}
	}
	return p.stack[0], nil
}

// parseVerticalBar handles a | in the input.
func (p *parser) parseVerticalBar() os.Error {
	p.concat()

	// The concatenation we just parsed is on top of the stack.
	// If it sits above an opVerticalBar, swap it below
	// (things below an opVerticalBar become an alternation).
	// Otherwise, push a new vertical bar.
	if !p.swapVerticalBar() {
		p.op(opVerticalBar)
	}

	return nil
}

// If the top of the stack is an element followed by an opVerticalBar
// swapVerticalBar swaps the two and returns true.
// Otherwise it returns false.
func (p *parser) swapVerticalBar() bool {
	if n := len(p.stack); n >= 2 {
		re1 := p.stack[n-1]
		re2 := p.stack[n-2]
		if re2.Op == opVerticalBar {
			p.stack[n-2] = re1
			p.stack[n-1] = re2
			return true
		}
	}
	return false
}

// parseRightParen handles a ) in the input.
func (p *parser) parseRightParen() os.Error {
	p.concat()
	if p.swapVerticalBar() {
		// pop vertical bar
		p.stack = p.stack[:len(p.stack)-1]
	}
	p.alternate()

	n := len(p.stack)
	if n < 2 {
		return &Error{ErrInternalError, ""}
	}
	re1 := p.stack[n-1]
	re2 := p.stack[n-2]
	p.stack = p.stack[:n-2]
	if re2.Op != opLeftParen {
		return &Error{ErrMissingParen, p.wholeRegexp}
	}
	if re2.Cap == 0 {
		// Just for grouping.
		p.push(re1)
	} else {
		re2.Op = OpCapture
		re2.Sub = re2.Sub0[:1]
		re2.Sub[0] = re1
		p.push(re2)
	}
	return nil
}

// parseClassChar parses a character class character at the beginning of s
// and returns it.
func (p *parser) parseClassChar(s, wholeClass string) (r int, rest string, err os.Error) {
	if s == "" {
		return 0, "", &Error{Code: ErrMissingBracket, Expr: wholeClass}
	}

	// TODO: Escapes

	return nextRune(s)
}

// parseClass parses a character class at the beginning of s
// and pushes it onto the parse stack.
func (p *parser) parseClass(s string) (rest string, err os.Error) {
	t := s[1:] // chop [
	re := &Regexp{Op: OpCharClass, Flags: p.flags}
	re.Rune = re.Rune0[:0]

	sign := +1
	if t != "" && t[0] == '^' {
		sign = -1
		t = t[1:]

		// If character class does not match \n, add it here,
		// so that negation later will do the right thing.
		if p.flags&ClassNL == 0 {
			re.Rune = append(re.Rune, '\n', '\n')
		}
	}

	class := re.Rune
	first := true // ] and - are okay as first char in class
	for t == "" || t[0] != ']' || first {
		// POSIX: - is only okay unescaped as first or last in class.
		// Perl: - is okay anywhere.
		if t != "" && t[0] == '-' && p.flags&PerlX == 0 && !first && (len(t) == 1 || t[1] != ']') {
			_, size := utf8.DecodeRuneInString(t[1:])
			return "", &Error{Code: ErrInvalidCharRange, Expr: t[:1+size]}
		}
		first = false

		// TODO: Look for [:alnum:]
		// TODO: Look for Unicode group.
		// TODO: Look for Perl group.

		// Single character or simple range.
		rng := t
		var lo, hi int
		if lo, t, err = p.parseClassChar(t, s); err != nil {
			return "", err
		}
		hi = lo
		// [a-] means (a|-) so check for final ].
		if len(t) >= 2 && t[0] == '-' && t[1] != ']' {
			t = t[1:]
			if hi, t, err = p.parseClassChar(t, s); err != nil {
				return "", err
			}
			if hi < lo {
				rng = rng[:len(rng)-len(t)]
				return "", &Error{Code: ErrInvalidCharRange, Expr: rng}
			}
		}

		// Expand last range if overlaps or abuts.
		if n := len(class); n > 0 {
			clo, chi := class[n-2], class[n-1]
			if lo <= chi+1 && clo <= hi+1 {
				if lo < clo {
					class[n-2] = lo
				}
				if hi > chi {
					class[n-1] = hi
				}
				continue
			}
		}

		class = append(class, lo, hi)
	}
	t = t[1:] // chop ]

	// Use &re.Rune instead of &class to avoid allocation.
	re.Rune = class
	class = cleanClass(&re.Rune)
	if sign < 0 {
		class = negateClass(class)
	}
	re.Rune = class
	p.push(re)
	return t, nil
}

// cleanClass sorts the ranges (pairs of elements of r),
// merges them, and eliminates duplicates.
func cleanClass(rp *[]int) []int {
	// Sort by lo increasing, hi decreasing to break ties.
	sort.Sort(ranges{rp})

	r := *rp
	// Merge abutting, overlapping.
	w := 2 // write index
	for i := 2; i < len(r); i += 2 {
		lo, hi := r[i], r[i+1]
		if lo <= r[w-1]+1 {
			// merge with previous range
			if hi > r[w-1] {
				r[w-1] = hi
			}
			continue
		}
		// new disjoint range
		r[w] = lo
		r[w+1] = hi
		w += 2
	}

	return r[:w]
}

// negateClass overwrites r and returns r's negation.
// It assumes the class r is already clean.
func negateClass(r []int) []int {
	nextLo := 0 // lo end of next class to add
	w := 0      // write index
	for i := 0; i < len(r); i += 2 {
		lo, hi := r[i], r[i+1]
		if nextLo <= lo-1 {
			r[w] = nextLo
			r[w+1] = lo - 1
			w += 2
		}
		nextLo = hi + 1
	}
	if nextLo <= unicode.MaxRune {
		// It's possible for the negation to have one more
		// range - this one - than the original class, so use append.
		r = append(r[:w], nextLo, unicode.MaxRune)
	}
	return r
}

// ranges implements sort.Interface on a []rune.
// The choice of receiver type definition is strange
// but avoids an allocation since we already have
// a *[]int.
type ranges struct {
	p *[]int
}

func (ra ranges) Less(i, j int) bool {
	p := *ra.p
	i *= 2
	j *= 2
	return p[i] < p[j] || p[i] == p[j] && p[i+1] > p[j+1]
}

func (ra ranges) Len() int {
	return len(*ra.p) / 2
}

func (ra ranges) Swap(i, j int) {
	p := *ra.p
	i *= 2
	j *= 2
	p[i], p[i+1], p[j], p[j+1] = p[j], p[j+1], p[i], p[i+1]
}


func checkUTF8(s string) os.Error {
	for s != "" {
		rune, size := utf8.DecodeRuneInString(s)
		if rune == utf8.RuneError && size == 1 {
			return &Error{Code: ErrInvalidUTF8, Expr: s}
		}
		s = s[size:]
	}
	return nil
}

func nextRune(s string) (c int, t string, err os.Error) {
	c, size := utf8.DecodeRuneInString(s)
	if c == utf8.RuneError && size == 1 {
		return 0, "", &Error{Code: ErrInvalidUTF8, Expr: s}
	}
	return c, s[size:], nil
}
