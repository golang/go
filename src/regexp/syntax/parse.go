// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// An Error describes a failure to parse a regular expression
// and gives the offending expression.
type Error struct {
	Code ErrorCode
	Expr string
}

func (e *Error) Error() string {
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
	ErrUnexpectedParen       ErrorCode = "unexpected )"
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
	free        *Regexp
	numCap      int // number of capturing groups seen
	wholeRegexp string
	tmpClass    []rune // temporary char class work space
}

func (p *parser) newRegexp(op Op) *Regexp {
	re := p.free
	if re != nil {
		p.free = re.Sub0[0]
		*re = Regexp{}
	} else {
		re = new(Regexp)
	}
	re.Op = op
	return re
}

func (p *parser) reuse(re *Regexp) {
	re.Sub0[0] = p.free
	p.free = re
}

// Parse stack manipulation.

// push pushes the regexp re onto the parse stack and returns the regexp.
func (p *parser) push(re *Regexp) *Regexp {
	if re.Op == OpCharClass && len(re.Rune) == 2 && re.Rune[0] == re.Rune[1] {
		// Single rune.
		if p.maybeConcat(re.Rune[0], p.flags&^FoldCase) {
			return nil
		}
		re.Op = OpLiteral
		re.Rune = re.Rune[:1]
		re.Flags = p.flags &^ FoldCase
	} else if re.Op == OpCharClass && len(re.Rune) == 4 &&
		re.Rune[0] == re.Rune[1] && re.Rune[2] == re.Rune[3] &&
		unicode.SimpleFold(re.Rune[0]) == re.Rune[2] &&
		unicode.SimpleFold(re.Rune[2]) == re.Rune[0] ||
		re.Op == OpCharClass && len(re.Rune) == 2 &&
			re.Rune[0]+1 == re.Rune[1] &&
			unicode.SimpleFold(re.Rune[0]) == re.Rune[1] &&
			unicode.SimpleFold(re.Rune[1]) == re.Rune[0] {
		// Case-insensitive rune like [Aa] or [Δδ].
		if p.maybeConcat(re.Rune[0], p.flags|FoldCase) {
			return nil
		}

		// Rewrite as (case-insensitive) literal.
		re.Op = OpLiteral
		re.Rune = re.Rune[:1]
		re.Flags = p.flags | FoldCase
	} else {
		// Incremental concatenation.
		p.maybeConcat(-1, 0)
	}

	p.stack = append(p.stack, re)
	return re
}

// maybeConcat implements incremental concatenation
// of literal runes into string nodes. The parser calls this
// before each push, so only the top fragment of the stack
// might need processing. Since this is called before a push,
// the topmost literal is no longer subject to operators like *
// (Otherwise ab* would turn into (ab)*.)
// If r >= 0 and there's a node left over, maybeConcat uses it
// to push r with the given flags.
// maybeConcat reports whether r was pushed.
func (p *parser) maybeConcat(r rune, flags Flags) bool {
	n := len(p.stack)
	if n < 2 {
		return false
	}

	re1 := p.stack[n-1]
	re2 := p.stack[n-2]
	if re1.Op != OpLiteral || re2.Op != OpLiteral || re1.Flags&FoldCase != re2.Flags&FoldCase {
		return false
	}

	// Push re1 into re2.
	re2.Rune = append(re2.Rune, re1.Rune...)

	// Reuse re1 if possible.
	if r >= 0 {
		re1.Rune = re1.Rune0[:1]
		re1.Rune[0] = r
		re1.Flags = flags
		return true
	}

	p.stack = p.stack[:n-1]
	p.reuse(re1)
	return false // did not push r
}

// newLiteral returns a new OpLiteral Regexp with the given flags
func (p *parser) newLiteral(r rune, flags Flags) *Regexp {
	re := p.newRegexp(OpLiteral)
	re.Flags = flags
	if flags&FoldCase != 0 {
		r = minFoldRune(r)
	}
	re.Rune0[0] = r
	re.Rune = re.Rune0[:1]
	return re
}

// minFoldRune returns the minimum rune fold-equivalent to r.
func minFoldRune(r rune) rune {
	if r < minFold || r > maxFold {
		return r
	}
	min := r
	r0 := r
	for r = unicode.SimpleFold(r); r != r0; r = unicode.SimpleFold(r) {
		if min > r {
			min = r
		}
	}
	return min
}

// literal pushes a literal regexp for the rune r on the stack
// and returns that regexp.
func (p *parser) literal(r rune) {
	p.push(p.newLiteral(r, p.flags))
}

// op pushes a regexp with the given op onto the stack
// and returns that regexp.
func (p *parser) op(op Op) *Regexp {
	re := p.newRegexp(op)
	re.Flags = p.flags
	return p.push(re)
}

// repeat replaces the top stack element with itself repeated according to op, min, max.
// before is the regexp suffix starting at the repetition operator.
// after is the regexp suffix following after the repetition operator.
// repeat returns an updated 'after' and an error, if any.
func (p *parser) repeat(op Op, min, max int, before, after, lastRepeat string) (string, error) {
	flags := p.flags
	if p.flags&PerlX != 0 {
		if len(after) > 0 && after[0] == '?' {
			after = after[1:]
			flags ^= NonGreedy
		}
		if lastRepeat != "" {
			// In Perl it is not allowed to stack repetition operators:
			// a** is a syntax error, not a doubled star, and a++ means
			// something else entirely, which we don't support!
			return "", &Error{ErrInvalidRepeatOp, lastRepeat[:len(lastRepeat)-len(after)]}
		}
	}
	n := len(p.stack)
	if n == 0 {
		return "", &Error{ErrMissingRepeatArgument, before[:len(before)-len(after)]}
	}
	sub := p.stack[n-1]
	if sub.Op >= opPseudo {
		return "", &Error{ErrMissingRepeatArgument, before[:len(before)-len(after)]}
	}

	re := p.newRegexp(op)
	re.Min = min
	re.Max = max
	re.Flags = flags
	re.Sub = re.Sub0[:1]
	re.Sub[0] = sub
	p.stack[n-1] = re

	if op == OpRepeat && (min >= 2 || max >= 2) && !repeatIsValid(re, 1000) {
		return "", &Error{ErrInvalidRepeatSize, before[:len(before)-len(after)]}
	}

	return after, nil
}

// repeatIsValid reports whether the repetition re is valid.
// Valid means that the combination of the top-level repetition
// and any inner repetitions does not exceed n copies of the
// innermost thing.
// This function rewalks the regexp tree and is called for every repetition,
// so we have to worry about inducing quadratic behavior in the parser.
// We avoid this by only calling repeatIsValid when min or max >= 2.
// In that case the depth of any >= 2 nesting can only get to 9 without
// triggering a parse error, so each subtree can only be rewalked 9 times.
func repeatIsValid(re *Regexp, n int) bool {
	if re.Op == OpRepeat {
		m := re.Max
		if m == 0 {
			return true
		}
		if m < 0 {
			m = re.Min
		}
		if m > n {
			return false
		}
		if m > 0 {
			n /= m
		}
	}
	for _, sub := range re.Sub {
		if !repeatIsValid(sub, n) {
			return false
		}
	}
	return true
}

// concat replaces the top of the stack (above the topmost '|' or '(') with its concatenation.
func (p *parser) concat() *Regexp {
	p.maybeConcat(-1, 0)

	// Scan down to find pseudo-operator | or (.
	i := len(p.stack)
	for i > 0 && p.stack[i-1].Op < opPseudo {
		i--
	}
	subs := p.stack[i:]
	p.stack = p.stack[:i]

	// Empty concatenation is special case.
	if len(subs) == 0 {
		return p.push(p.newRegexp(OpEmptyMatch))
	}

	return p.push(p.collapse(subs, OpConcat))
}

// alternate replaces the top of the stack (above the topmost '(') with its alternation.
func (p *parser) alternate() *Regexp {
	// Scan down to find pseudo-operator (.
	// There are no | above (.
	i := len(p.stack)
	for i > 0 && p.stack[i-1].Op < opPseudo {
		i--
	}
	subs := p.stack[i:]
	p.stack = p.stack[:i]

	// Make sure top class is clean.
	// All the others already are (see swapVerticalBar).
	if len(subs) > 0 {
		cleanAlt(subs[len(subs)-1])
	}

	// Empty alternate is special case
	// (shouldn't happen but easy to handle).
	if len(subs) == 0 {
		return p.push(p.newRegexp(OpNoMatch))
	}

	return p.push(p.collapse(subs, OpAlternate))
}

// cleanAlt cleans re for eventual inclusion in an alternation.
func cleanAlt(re *Regexp) {
	switch re.Op {
	case OpCharClass:
		re.Rune = cleanClass(&re.Rune)
		if len(re.Rune) == 2 && re.Rune[0] == 0 && re.Rune[1] == unicode.MaxRune {
			re.Rune = nil
			re.Op = OpAnyChar
			return
		}
		if len(re.Rune) == 4 && re.Rune[0] == 0 && re.Rune[1] == '\n'-1 && re.Rune[2] == '\n'+1 && re.Rune[3] == unicode.MaxRune {
			re.Rune = nil
			re.Op = OpAnyCharNotNL
			return
		}
		if cap(re.Rune)-len(re.Rune) > 100 {
			// re.Rune will not grow any more.
			// Make a copy or inline to reclaim storage.
			re.Rune = append(re.Rune0[:0], re.Rune...)
		}
	}
}

// collapse returns the result of applying op to sub.
// If sub contains op nodes, they all get hoisted up
// so that there is never a concat of a concat or an
// alternate of an alternate.
func (p *parser) collapse(subs []*Regexp, op Op) *Regexp {
	if len(subs) == 1 {
		return subs[0]
	}
	re := p.newRegexp(op)
	re.Sub = re.Sub0[:0]
	for _, sub := range subs {
		if sub.Op == op {
			re.Sub = append(re.Sub, sub.Sub...)
			p.reuse(sub)
		} else {
			re.Sub = append(re.Sub, sub)
		}
	}
	if op == OpAlternate {
		re.Sub = p.factor(re.Sub)
		if len(re.Sub) == 1 {
			old := re
			re = re.Sub[0]
			p.reuse(old)
		}
	}
	return re
}

// factor factors common prefixes from the alternation list sub.
// It returns a replacement list that reuses the same storage and
// frees (passes to p.reuse) any removed *Regexps.
//
// For example,
//     ABC|ABD|AEF|BCX|BCY
// simplifies by literal prefix extraction to
//     A(B(C|D)|EF)|BC(X|Y)
// which simplifies by character class introduction to
//     A(B[CD]|EF)|BC[XY]
//
func (p *parser) factor(sub []*Regexp) []*Regexp {
	if len(sub) < 2 {
		return sub
	}

	// Round 1: Factor out common literal prefixes.
	var str []rune
	var strflags Flags
	start := 0
	out := sub[:0]
	for i := 0; i <= len(sub); i++ {
		// Invariant: the Regexps that were in sub[0:start] have been
		// used or marked for reuse, and the slice space has been reused
		// for out (len(out) <= start).
		//
		// Invariant: sub[start:i] consists of regexps that all begin
		// with str as modified by strflags.
		var istr []rune
		var iflags Flags
		if i < len(sub) {
			istr, iflags = p.leadingString(sub[i])
			if iflags == strflags {
				same := 0
				for same < len(str) && same < len(istr) && str[same] == istr[same] {
					same++
				}
				if same > 0 {
					// Matches at least one rune in current range.
					// Keep going around.
					str = str[:same]
					continue
				}
			}
		}

		// Found end of a run with common leading literal string:
		// sub[start:i] all begin with str[0:len(str)], but sub[i]
		// does not even begin with str[0].
		//
		// Factor out common string and append factored expression to out.
		if i == start {
			// Nothing to do - run of length 0.
		} else if i == start+1 {
			// Just one: don't bother factoring.
			out = append(out, sub[start])
		} else {
			// Construct factored form: prefix(suffix1|suffix2|...)
			prefix := p.newRegexp(OpLiteral)
			prefix.Flags = strflags
			prefix.Rune = append(prefix.Rune[:0], str...)

			for j := start; j < i; j++ {
				sub[j] = p.removeLeadingString(sub[j], len(str))
			}
			suffix := p.collapse(sub[start:i], OpAlternate) // recurse

			re := p.newRegexp(OpConcat)
			re.Sub = append(re.Sub[:0], prefix, suffix)
			out = append(out, re)
		}

		// Prepare for next iteration.
		start = i
		str = istr
		strflags = iflags
	}
	sub = out

	// Round 2: Factor out common simple prefixes,
	// just the first piece of each concatenation.
	// This will be good enough a lot of the time.
	//
	// Complex subexpressions (e.g. involving quantifiers)
	// are not safe to factor because that collapses their
	// distinct paths through the automaton, which affects
	// correctness in some cases.
	start = 0
	out = sub[:0]
	var first *Regexp
	for i := 0; i <= len(sub); i++ {
		// Invariant: the Regexps that were in sub[0:start] have been
		// used or marked for reuse, and the slice space has been reused
		// for out (len(out) <= start).
		//
		// Invariant: sub[start:i] consists of regexps that all begin with ifirst.
		var ifirst *Regexp
		if i < len(sub) {
			ifirst = p.leadingRegexp(sub[i])
			if first != nil && first.Equal(ifirst) &&
				// first must be a character class OR a fixed repeat of a character class.
				(isCharClass(first) || (first.Op == OpRepeat && first.Min == first.Max && isCharClass(first.Sub[0]))) {
				continue
			}
		}

		// Found end of a run with common leading regexp:
		// sub[start:i] all begin with first but sub[i] does not.
		//
		// Factor out common regexp and append factored expression to out.
		if i == start {
			// Nothing to do - run of length 0.
		} else if i == start+1 {
			// Just one: don't bother factoring.
			out = append(out, sub[start])
		} else {
			// Construct factored form: prefix(suffix1|suffix2|...)
			prefix := first
			for j := start; j < i; j++ {
				reuse := j != start // prefix came from sub[start]
				sub[j] = p.removeLeadingRegexp(sub[j], reuse)
			}
			suffix := p.collapse(sub[start:i], OpAlternate) // recurse

			re := p.newRegexp(OpConcat)
			re.Sub = append(re.Sub[:0], prefix, suffix)
			out = append(out, re)
		}

		// Prepare for next iteration.
		start = i
		first = ifirst
	}
	sub = out

	// Round 3: Collapse runs of single literals into character classes.
	start = 0
	out = sub[:0]
	for i := 0; i <= len(sub); i++ {
		// Invariant: the Regexps that were in sub[0:start] have been
		// used or marked for reuse, and the slice space has been reused
		// for out (len(out) <= start).
		//
		// Invariant: sub[start:i] consists of regexps that are either
		// literal runes or character classes.
		if i < len(sub) && isCharClass(sub[i]) {
			continue
		}

		// sub[i] is not a char or char class;
		// emit char class for sub[start:i]...
		if i == start {
			// Nothing to do - run of length 0.
		} else if i == start+1 {
			out = append(out, sub[start])
		} else {
			// Make new char class.
			// Start with most complex regexp in sub[start].
			max := start
			for j := start + 1; j < i; j++ {
				if sub[max].Op < sub[j].Op || sub[max].Op == sub[j].Op && len(sub[max].Rune) < len(sub[j].Rune) {
					max = j
				}
			}
			sub[start], sub[max] = sub[max], sub[start]

			for j := start + 1; j < i; j++ {
				mergeCharClass(sub[start], sub[j])
				p.reuse(sub[j])
			}
			cleanAlt(sub[start])
			out = append(out, sub[start])
		}

		// ... and then emit sub[i].
		if i < len(sub) {
			out = append(out, sub[i])
		}
		start = i + 1
	}
	sub = out

	// Round 4: Collapse runs of empty matches into a single empty match.
	start = 0
	out = sub[:0]
	for i := range sub {
		if i+1 < len(sub) && sub[i].Op == OpEmptyMatch && sub[i+1].Op == OpEmptyMatch {
			continue
		}
		out = append(out, sub[i])
	}
	sub = out

	return sub
}

// leadingString returns the leading literal string that re begins with.
// The string refers to storage in re or its children.
func (p *parser) leadingString(re *Regexp) ([]rune, Flags) {
	if re.Op == OpConcat && len(re.Sub) > 0 {
		re = re.Sub[0]
	}
	if re.Op != OpLiteral {
		return nil, 0
	}
	return re.Rune, re.Flags & FoldCase
}

// removeLeadingString removes the first n leading runes
// from the beginning of re. It returns the replacement for re.
func (p *parser) removeLeadingString(re *Regexp, n int) *Regexp {
	if re.Op == OpConcat && len(re.Sub) > 0 {
		// Removing a leading string in a concatenation
		// might simplify the concatenation.
		sub := re.Sub[0]
		sub = p.removeLeadingString(sub, n)
		re.Sub[0] = sub
		if sub.Op == OpEmptyMatch {
			p.reuse(sub)
			switch len(re.Sub) {
			case 0, 1:
				// Impossible but handle.
				re.Op = OpEmptyMatch
				re.Sub = nil
			case 2:
				old := re
				re = re.Sub[1]
				p.reuse(old)
			default:
				copy(re.Sub, re.Sub[1:])
				re.Sub = re.Sub[:len(re.Sub)-1]
			}
		}
		return re
	}

	if re.Op == OpLiteral {
		re.Rune = re.Rune[:copy(re.Rune, re.Rune[n:])]
		if len(re.Rune) == 0 {
			re.Op = OpEmptyMatch
		}
	}
	return re
}

// leadingRegexp returns the leading regexp that re begins with.
// The regexp refers to storage in re or its children.
func (p *parser) leadingRegexp(re *Regexp) *Regexp {
	if re.Op == OpEmptyMatch {
		return nil
	}
	if re.Op == OpConcat && len(re.Sub) > 0 {
		sub := re.Sub[0]
		if sub.Op == OpEmptyMatch {
			return nil
		}
		return sub
	}
	return re
}

// removeLeadingRegexp removes the leading regexp in re.
// It returns the replacement for re.
// If reuse is true, it passes the removed regexp (if no longer needed) to p.reuse.
func (p *parser) removeLeadingRegexp(re *Regexp, reuse bool) *Regexp {
	if re.Op == OpConcat && len(re.Sub) > 0 {
		if reuse {
			p.reuse(re.Sub[0])
		}
		re.Sub = re.Sub[:copy(re.Sub, re.Sub[1:])]
		switch len(re.Sub) {
		case 0:
			re.Op = OpEmptyMatch
			re.Sub = nil
		case 1:
			old := re
			re = re.Sub[0]
			p.reuse(old)
		}
		return re
	}
	if reuse {
		p.reuse(re)
	}
	return p.newRegexp(OpEmptyMatch)
}

func literalRegexp(s string, flags Flags) *Regexp {
	re := &Regexp{Op: OpLiteral}
	re.Flags = flags
	re.Rune = re.Rune0[:0] // use local storage for small strings
	for _, c := range s {
		if len(re.Rune) >= cap(re.Rune) {
			// string is too long to fit in Rune0.  let Go handle it
			re.Rune = []rune(s)
			break
		}
		re.Rune = append(re.Rune, c)
	}
	return re
}

// Parsing.

// Parse parses a regular expression string s, controlled by the specified
// Flags, and returns a regular expression parse tree. The syntax is
// described in the top-level comment.
func Parse(s string, flags Flags) (*Regexp, error) {
	if flags&Literal != 0 {
		// Trivial parser for literal string.
		if err := checkUTF8(s); err != nil {
			return nil, err
		}
		return literalRegexp(s, flags), nil
	}

	// Otherwise, must do real work.
	var (
		p          parser
		err        error
		c          rune
		op         Op
		lastRepeat string
	)
	p.flags = flags
	p.wholeRegexp = s
	t := s
	for t != "" {
		repeat := ""
	BigSwitch:
		switch t[0] {
		default:
			if c, t, err = nextRune(t); err != nil {
				return nil, err
			}
			p.literal(c)

		case '(':
			if p.flags&PerlX != 0 && len(t) >= 2 && t[1] == '?' {
				// Flag changes and non-capturing groups.
				if t, err = p.parsePerlFlags(t); err != nil {
					return nil, err
				}
				break
			}
			p.numCap++
			p.op(opLeftParen).Cap = p.numCap
			t = t[1:]
		case '|':
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
			before := t
			switch t[0] {
			case '*':
				op = OpStar
			case '+':
				op = OpPlus
			case '?':
				op = OpQuest
			}
			after := t[1:]
			if after, err = p.repeat(op, 0, 0, before, after, lastRepeat); err != nil {
				return nil, err
			}
			repeat = before
			t = after
		case '{':
			op = OpRepeat
			before := t
			min, max, after, ok := p.parseRepeat(t)
			if !ok {
				// If the repeat cannot be parsed, { is a literal.
				p.literal('{')
				t = t[1:]
				break
			}
			if min < 0 || min > 1000 || max > 1000 || max >= 0 && min > max {
				// Numbers were too big, or max is present and min > max.
				return nil, &Error{ErrInvalidRepeatSize, before[:len(before)-len(after)]}
			}
			if after, err = p.repeat(op, min, max, before, after, lastRepeat); err != nil {
				return nil, err
			}
			repeat = before
			t = after
		case '\\':
			if p.flags&PerlX != 0 && len(t) >= 2 {
				switch t[1] {
				case 'A':
					p.op(OpBeginText)
					t = t[2:]
					break BigSwitch
				case 'b':
					p.op(OpWordBoundary)
					t = t[2:]
					break BigSwitch
				case 'B':
					p.op(OpNoWordBoundary)
					t = t[2:]
					break BigSwitch
				case 'C':
					// any byte; not supported
					return nil, &Error{ErrInvalidEscape, t[:2]}
				case 'Q':
					// \Q ... \E: the ... is always literals
					var lit string
					if i := strings.Index(t, `\E`); i < 0 {
						lit = t[2:]
						t = ""
					} else {
						lit = t[2:i]
						t = t[i+2:]
					}
					for lit != "" {
						c, rest, err := nextRune(lit)
						if err != nil {
							return nil, err
						}
						p.literal(c)
						lit = rest
					}
					break BigSwitch
				case 'z':
					p.op(OpEndText)
					t = t[2:]
					break BigSwitch
				}
			}

			re := p.newRegexp(OpCharClass)
			re.Flags = p.flags

			// Look for Unicode character group like \p{Han}
			if len(t) >= 2 && (t[1] == 'p' || t[1] == 'P') {
				r, rest, err := p.parseUnicodeClass(t, re.Rune0[:0])
				if err != nil {
					return nil, err
				}
				if r != nil {
					re.Rune = r
					t = rest
					p.push(re)
					break BigSwitch
				}
			}

			// Perl character class escape.
			if r, rest := p.parsePerlClassEscape(t, re.Rune0[:0]); r != nil {
				re.Rune = r
				t = rest
				p.push(re)
				break BigSwitch
			}
			p.reuse(re)

			// Ordinary single-character escape.
			if c, t, err = p.parseEscape(t); err != nil {
				return nil, err
			}
			p.literal(c)
		}
		lastRepeat = repeat
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

// parseRepeat parses {min} (max=min) or {min,} (max=-1) or {min,max}.
// If s is not of that form, it returns ok == false.
// If s has the right form but the values are too big, it returns min == -1, ok == true.
func (p *parser) parseRepeat(s string) (min, max int, rest string, ok bool) {
	if s == "" || s[0] != '{' {
		return
	}
	s = s[1:]
	var ok1 bool
	if min, s, ok1 = p.parseInt(s); !ok1 {
		return
	}
	if s == "" {
		return
	}
	if s[0] != ',' {
		max = min
	} else {
		s = s[1:]
		if s == "" {
			return
		}
		if s[0] == '}' {
			max = -1
		} else if max, s, ok1 = p.parseInt(s); !ok1 {
			return
		} else if max < 0 {
			// parseInt found too big a number
			min = -1
		}
	}
	if s == "" || s[0] != '}' {
		return
	}
	rest = s[1:]
	ok = true
	return
}

// parsePerlFlags parses a Perl flag setting or non-capturing group or both,
// like (?i) or (?: or (?i:.  It removes the prefix from s and updates the parse state.
// The caller must have ensured that s begins with "(?".
func (p *parser) parsePerlFlags(s string) (rest string, err error) {
	t := s

	// Check for named captures, first introduced in Python's regexp library.
	// As usual, there are three slightly different syntaxes:
	//
	//   (?P<name>expr)   the original, introduced by Python
	//   (?<name>expr)    the .NET alteration, adopted by Perl 5.10
	//   (?'name'expr)    another .NET alteration, adopted by Perl 5.10
	//
	// Perl 5.10 gave in and implemented the Python version too,
	// but they claim that the last two are the preferred forms.
	// PCRE and languages based on it (specifically, PHP and Ruby)
	// support all three as well. EcmaScript 4 uses only the Python form.
	//
	// In both the open source world (via Code Search) and the
	// Google source tree, (?P<expr>name) is the dominant form,
	// so that's the one we implement. One is enough.
	if len(t) > 4 && t[2] == 'P' && t[3] == '<' {
		// Pull out name.
		end := strings.IndexRune(t, '>')
		if end < 0 {
			if err = checkUTF8(t); err != nil {
				return "", err
			}
			return "", &Error{ErrInvalidNamedCapture, s}
		}

		capture := t[:end+1] // "(?P<name>"
		name := t[4:end]     // "name"
		if err = checkUTF8(name); err != nil {
			return "", err
		}
		if !isValidCaptureName(name) {
			return "", &Error{ErrInvalidNamedCapture, capture}
		}

		// Like ordinary capture, but named.
		p.numCap++
		re := p.op(opLeftParen)
		re.Cap = p.numCap
		re.Name = name
		return t[end+1:], nil
	}

	// Non-capturing group. Might also twiddle Perl flags.
	var c rune
	t = t[2:] // skip (?
	flags := p.flags
	sign := +1
	sawFlag := false
Loop:
	for t != "" {
		if c, t, err = nextRune(t); err != nil {
			return "", err
		}
		switch c {
		default:
			break Loop

		// Flags.
		case 'i':
			flags |= FoldCase
			sawFlag = true
		case 'm':
			flags &^= OneLine
			sawFlag = true
		case 's':
			flags |= DotNL
			sawFlag = true
		case 'U':
			flags |= NonGreedy
			sawFlag = true

		// Switch to negation.
		case '-':
			if sign < 0 {
				break Loop
			}
			sign = -1
			// Invert flags so that | above turn into &^ and vice versa.
			// We'll invert flags again before using it below.
			flags = ^flags
			sawFlag = false

		// End of flags, starting group or not.
		case ':', ')':
			if sign < 0 {
				if !sawFlag {
					break Loop
				}
				flags = ^flags
			}
			if c == ':' {
				// Open new group
				p.op(opLeftParen)
			}
			p.flags = flags
			return t, nil
		}
	}

	return "", &Error{ErrInvalidPerlOp, s[:len(s)-len(t)]}
}

// isValidCaptureName reports whether name
// is a valid capture name: [A-Za-z0-9_]+.
// PCRE limits names to 32 bytes.
// Python rejects names starting with digits.
// We don't enforce either of those.
func isValidCaptureName(name string) bool {
	if name == "" {
		return false
	}
	for _, c := range name {
		if c != '_' && !isalnum(c) {
			return false
		}
	}
	return true
}

// parseInt parses a decimal integer.
func (p *parser) parseInt(s string) (n int, rest string, ok bool) {
	if s == "" || s[0] < '0' || '9' < s[0] {
		return
	}
	// Disallow leading zeros.
	if len(s) >= 2 && s[0] == '0' && '0' <= s[1] && s[1] <= '9' {
		return
	}
	t := s
	for s != "" && '0' <= s[0] && s[0] <= '9' {
		s = s[1:]
	}
	rest = s
	ok = true
	// Have digits, compute value.
	t = t[:len(t)-len(s)]
	for i := 0; i < len(t); i++ {
		// Avoid overflow.
		if n >= 1e8 {
			n = -1
			break
		}
		n = n*10 + int(t[i]) - '0'
	}
	return
}

// can this be represented as a character class?
// single-rune literal string, char class, ., and .|\n.
func isCharClass(re *Regexp) bool {
	return re.Op == OpLiteral && len(re.Rune) == 1 ||
		re.Op == OpCharClass ||
		re.Op == OpAnyCharNotNL ||
		re.Op == OpAnyChar
}

// does re match r?
func matchRune(re *Regexp, r rune) bool {
	switch re.Op {
	case OpLiteral:
		return len(re.Rune) == 1 && re.Rune[0] == r
	case OpCharClass:
		for i := 0; i < len(re.Rune); i += 2 {
			if re.Rune[i] <= r && r <= re.Rune[i+1] {
				return true
			}
		}
		return false
	case OpAnyCharNotNL:
		return r != '\n'
	case OpAnyChar:
		return true
	}
	return false
}

// parseVerticalBar handles a | in the input.
func (p *parser) parseVerticalBar() error {
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

// mergeCharClass makes dst = dst|src.
// The caller must ensure that dst.Op >= src.Op,
// to reduce the amount of copying.
func mergeCharClass(dst, src *Regexp) {
	switch dst.Op {
	case OpAnyChar:
		// src doesn't add anything.
	case OpAnyCharNotNL:
		// src might add \n
		if matchRune(src, '\n') {
			dst.Op = OpAnyChar
		}
	case OpCharClass:
		// src is simpler, so either literal or char class
		if src.Op == OpLiteral {
			dst.Rune = appendLiteral(dst.Rune, src.Rune[0], src.Flags)
		} else {
			dst.Rune = appendClass(dst.Rune, src.Rune)
		}
	case OpLiteral:
		// both literal
		if src.Rune[0] == dst.Rune[0] && src.Flags == dst.Flags {
			break
		}
		dst.Op = OpCharClass
		dst.Rune = appendLiteral(dst.Rune[:0], dst.Rune[0], dst.Flags)
		dst.Rune = appendLiteral(dst.Rune, src.Rune[0], src.Flags)
	}
}

// If the top of the stack is an element followed by an opVerticalBar
// swapVerticalBar swaps the two and returns true.
// Otherwise it returns false.
func (p *parser) swapVerticalBar() bool {
	// If above and below vertical bar are literal or char class,
	// can merge into a single char class.
	n := len(p.stack)
	if n >= 3 && p.stack[n-2].Op == opVerticalBar && isCharClass(p.stack[n-1]) && isCharClass(p.stack[n-3]) {
		re1 := p.stack[n-1]
		re3 := p.stack[n-3]
		// Make re3 the more complex of the two.
		if re1.Op > re3.Op {
			re1, re3 = re3, re1
			p.stack[n-3] = re3
		}
		mergeCharClass(re3, re1)
		p.reuse(re1)
		p.stack = p.stack[:n-1]
		return true
	}

	if n >= 2 {
		re1 := p.stack[n-1]
		re2 := p.stack[n-2]
		if re2.Op == opVerticalBar {
			if n >= 3 {
				// Now out of reach.
				// Clean opportunistically.
				cleanAlt(p.stack[n-3])
			}
			p.stack[n-2] = re1
			p.stack[n-1] = re2
			return true
		}
	}
	return false
}

// parseRightParen handles a ) in the input.
func (p *parser) parseRightParen() error {
	p.concat()
	if p.swapVerticalBar() {
		// pop vertical bar
		p.stack = p.stack[:len(p.stack)-1]
	}
	p.alternate()

	n := len(p.stack)
	if n < 2 {
		return &Error{ErrUnexpectedParen, p.wholeRegexp}
	}
	re1 := p.stack[n-1]
	re2 := p.stack[n-2]
	p.stack = p.stack[:n-2]
	if re2.Op != opLeftParen {
		return &Error{ErrUnexpectedParen, p.wholeRegexp}
	}
	// Restore flags at time of paren.
	p.flags = re2.Flags
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

// parseEscape parses an escape sequence at the beginning of s
// and returns the rune.
func (p *parser) parseEscape(s string) (r rune, rest string, err error) {
	t := s[1:]
	if t == "" {
		return 0, "", &Error{ErrTrailingBackslash, ""}
	}
	c, t, err := nextRune(t)
	if err != nil {
		return 0, "", err
	}

Switch:
	switch c {
	default:
		if c < utf8.RuneSelf && !isalnum(c) {
			// Escaped non-word characters are always themselves.
			// PCRE is not quite so rigorous: it accepts things like
			// \q, but we don't. We once rejected \_, but too many
			// programs and people insist on using it, so allow \_.
			return c, t, nil
		}

	// Octal escapes.
	case '1', '2', '3', '4', '5', '6', '7':
		// Single non-zero digit is a backreference; not supported
		if t == "" || t[0] < '0' || t[0] > '7' {
			break
		}
		fallthrough
	case '0':
		// Consume up to three octal digits; already have one.
		r = c - '0'
		for i := 1; i < 3; i++ {
			if t == "" || t[0] < '0' || t[0] > '7' {
				break
			}
			r = r*8 + rune(t[0]) - '0'
			t = t[1:]
		}
		return r, t, nil

	// Hexadecimal escapes.
	case 'x':
		if t == "" {
			break
		}
		if c, t, err = nextRune(t); err != nil {
			return 0, "", err
		}
		if c == '{' {
			// Any number of digits in braces.
			// Perl accepts any text at all; it ignores all text
			// after the first non-hex digit. We require only hex digits,
			// and at least one.
			nhex := 0
			r = 0
			for {
				if t == "" {
					break Switch
				}
				if c, t, err = nextRune(t); err != nil {
					return 0, "", err
				}
				if c == '}' {
					break
				}
				v := unhex(c)
				if v < 0 {
					break Switch
				}
				r = r*16 + v
				if r > unicode.MaxRune {
					break Switch
				}
				nhex++
			}
			if nhex == 0 {
				break Switch
			}
			return r, t, nil
		}

		// Easy case: two hex digits.
		x := unhex(c)
		if c, t, err = nextRune(t); err != nil {
			return 0, "", err
		}
		y := unhex(c)
		if x < 0 || y < 0 {
			break
		}
		return x*16 + y, t, nil

	// C escapes. There is no case 'b', to avoid misparsing
	// the Perl word-boundary \b as the C backspace \b
	// when in POSIX mode. In Perl, /\b/ means word-boundary
	// but /[\b]/ means backspace. We don't support that.
	// If you want a backspace, embed a literal backspace
	// character or use \x08.
	case 'a':
		return '\a', t, err
	case 'f':
		return '\f', t, err
	case 'n':
		return '\n', t, err
	case 'r':
		return '\r', t, err
	case 't':
		return '\t', t, err
	case 'v':
		return '\v', t, err
	}
	return 0, "", &Error{ErrInvalidEscape, s[:len(s)-len(t)]}
}

// parseClassChar parses a character class character at the beginning of s
// and returns it.
func (p *parser) parseClassChar(s, wholeClass string) (r rune, rest string, err error) {
	if s == "" {
		return 0, "", &Error{Code: ErrMissingBracket, Expr: wholeClass}
	}

	// Allow regular escape sequences even though
	// many need not be escaped in this context.
	if s[0] == '\\' {
		return p.parseEscape(s)
	}

	return nextRune(s)
}

type charGroup struct {
	sign  int
	class []rune
}

// parsePerlClassEscape parses a leading Perl character class escape like \d
// from the beginning of s. If one is present, it appends the characters to r
// and returns the new slice r and the remainder of the string.
func (p *parser) parsePerlClassEscape(s string, r []rune) (out []rune, rest string) {
	if p.flags&PerlX == 0 || len(s) < 2 || s[0] != '\\' {
		return
	}
	g := perlGroup[s[0:2]]
	if g.sign == 0 {
		return
	}
	return p.appendGroup(r, g), s[2:]
}

// parseNamedClass parses a leading POSIX named character class like [:alnum:]
// from the beginning of s. If one is present, it appends the characters to r
// and returns the new slice r and the remainder of the string.
func (p *parser) parseNamedClass(s string, r []rune) (out []rune, rest string, err error) {
	if len(s) < 2 || s[0] != '[' || s[1] != ':' {
		return
	}

	i := strings.Index(s[2:], ":]")
	if i < 0 {
		return
	}
	i += 2
	name, s := s[0:i+2], s[i+2:]
	g := posixGroup[name]
	if g.sign == 0 {
		return nil, "", &Error{ErrInvalidCharRange, name}
	}
	return p.appendGroup(r, g), s, nil
}

func (p *parser) appendGroup(r []rune, g charGroup) []rune {
	if p.flags&FoldCase == 0 {
		if g.sign < 0 {
			r = appendNegatedClass(r, g.class)
		} else {
			r = appendClass(r, g.class)
		}
	} else {
		tmp := p.tmpClass[:0]
		tmp = appendFoldedClass(tmp, g.class)
		p.tmpClass = tmp
		tmp = cleanClass(&p.tmpClass)
		if g.sign < 0 {
			r = appendNegatedClass(r, tmp)
		} else {
			r = appendClass(r, tmp)
		}
	}
	return r
}

var anyTable = &unicode.RangeTable{
	R16: []unicode.Range16{{Lo: 0, Hi: 1<<16 - 1, Stride: 1}},
	R32: []unicode.Range32{{Lo: 1 << 16, Hi: unicode.MaxRune, Stride: 1}},
}

// unicodeTable returns the unicode.RangeTable identified by name
// and the table of additional fold-equivalent code points.
func unicodeTable(name string) (*unicode.RangeTable, *unicode.RangeTable) {
	// Special case: "Any" means any.
	if name == "Any" {
		return anyTable, anyTable
	}
	if t := unicode.Categories[name]; t != nil {
		return t, unicode.FoldCategory[name]
	}
	if t := unicode.Scripts[name]; t != nil {
		return t, unicode.FoldScript[name]
	}
	return nil, nil
}

// parseUnicodeClass parses a leading Unicode character class like \p{Han}
// from the beginning of s. If one is present, it appends the characters to r
// and returns the new slice r and the remainder of the string.
func (p *parser) parseUnicodeClass(s string, r []rune) (out []rune, rest string, err error) {
	if p.flags&UnicodeGroups == 0 || len(s) < 2 || s[0] != '\\' || s[1] != 'p' && s[1] != 'P' {
		return
	}

	// Committed to parse or return error.
	sign := +1
	if s[1] == 'P' {
		sign = -1
	}
	t := s[2:]
	c, t, err := nextRune(t)
	if err != nil {
		return
	}
	var seq, name string
	if c != '{' {
		// Single-letter name.
		seq = s[:len(s)-len(t)]
		name = seq[2:]
	} else {
		// Name is in braces.
		end := strings.IndexRune(s, '}')
		if end < 0 {
			if err = checkUTF8(s); err != nil {
				return
			}
			return nil, "", &Error{ErrInvalidCharRange, s}
		}
		seq, t = s[:end+1], s[end+1:]
		name = s[3:end]
		if err = checkUTF8(name); err != nil {
			return
		}
	}

	// Group can have leading negation too.  \p{^Han} == \P{Han}, \P{^Han} == \p{Han}.
	if name != "" && name[0] == '^' {
		sign = -sign
		name = name[1:]
	}

	tab, fold := unicodeTable(name)
	if tab == nil {
		return nil, "", &Error{ErrInvalidCharRange, seq}
	}

	if p.flags&FoldCase == 0 || fold == nil {
		if sign > 0 {
			r = appendTable(r, tab)
		} else {
			r = appendNegatedTable(r, tab)
		}
	} else {
		// Merge and clean tab and fold in a temporary buffer.
		// This is necessary for the negative case and just tidy
		// for the positive case.
		tmp := p.tmpClass[:0]
		tmp = appendTable(tmp, tab)
		tmp = appendTable(tmp, fold)
		p.tmpClass = tmp
		tmp = cleanClass(&p.tmpClass)
		if sign > 0 {
			r = appendClass(r, tmp)
		} else {
			r = appendNegatedClass(r, tmp)
		}
	}
	return r, t, nil
}

// parseClass parses a character class at the beginning of s
// and pushes it onto the parse stack.
func (p *parser) parseClass(s string) (rest string, err error) {
	t := s[1:] // chop [
	re := p.newRegexp(OpCharClass)
	re.Flags = p.flags
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

		// Look for POSIX [:alnum:] etc.
		if len(t) > 2 && t[0] == '[' && t[1] == ':' {
			nclass, nt, err := p.parseNamedClass(t, class)
			if err != nil {
				return "", err
			}
			if nclass != nil {
				class, t = nclass, nt
				continue
			}
		}

		// Look for Unicode character group like \p{Han}.
		nclass, nt, err := p.parseUnicodeClass(t, class)
		if err != nil {
			return "", err
		}
		if nclass != nil {
			class, t = nclass, nt
			continue
		}

		// Look for Perl character class symbols (extension).
		if nclass, nt := p.parsePerlClassEscape(t, class); nclass != nil {
			class, t = nclass, nt
			continue
		}

		// Single character or simple range.
		rng := t
		var lo, hi rune
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
		if p.flags&FoldCase == 0 {
			class = appendRange(class, lo, hi)
		} else {
			class = appendFoldedRange(class, lo, hi)
		}
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
func cleanClass(rp *[]rune) []rune {

	// Sort by lo increasing, hi decreasing to break ties.
	sort.Sort(ranges{rp})

	r := *rp
	if len(r) < 2 {
		return r
	}

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

// appendLiteral returns the result of appending the literal x to the class r.
func appendLiteral(r []rune, x rune, flags Flags) []rune {
	if flags&FoldCase != 0 {
		return appendFoldedRange(r, x, x)
	}
	return appendRange(r, x, x)
}

// appendRange returns the result of appending the range lo-hi to the class r.
func appendRange(r []rune, lo, hi rune) []rune {
	// Expand last range or next to last range if it overlaps or abuts.
	// Checking two ranges helps when appending case-folded
	// alphabets, so that one range can be expanding A-Z and the
	// other expanding a-z.
	n := len(r)
	for i := 2; i <= 4; i += 2 { // twice, using i=2, i=4
		if n >= i {
			rlo, rhi := r[n-i], r[n-i+1]
			if lo <= rhi+1 && rlo <= hi+1 {
				if lo < rlo {
					r[n-i] = lo
				}
				if hi > rhi {
					r[n-i+1] = hi
				}
				return r
			}
		}
	}

	return append(r, lo, hi)
}

const (
	// minimum and maximum runes involved in folding.
	// checked during test.
	minFold = 0x0041
	maxFold = 0x1e943
)

// appendFoldedRange returns the result of appending the range lo-hi
// and its case folding-equivalent runes to the class r.
func appendFoldedRange(r []rune, lo, hi rune) []rune {
	// Optimizations.
	if lo <= minFold && hi >= maxFold {
		// Range is full: folding can't add more.
		return appendRange(r, lo, hi)
	}
	if hi < minFold || lo > maxFold {
		// Range is outside folding possibilities.
		return appendRange(r, lo, hi)
	}
	if lo < minFold {
		// [lo, minFold-1] needs no folding.
		r = appendRange(r, lo, minFold-1)
		lo = minFold
	}
	if hi > maxFold {
		// [maxFold+1, hi] needs no folding.
		r = appendRange(r, maxFold+1, hi)
		hi = maxFold
	}

	// Brute force. Depend on appendRange to coalesce ranges on the fly.
	for c := lo; c <= hi; c++ {
		r = appendRange(r, c, c)
		f := unicode.SimpleFold(c)
		for f != c {
			r = appendRange(r, f, f)
			f = unicode.SimpleFold(f)
		}
	}
	return r
}

// appendClass returns the result of appending the class x to the class r.
// It assume x is clean.
func appendClass(r []rune, x []rune) []rune {
	for i := 0; i < len(x); i += 2 {
		r = appendRange(r, x[i], x[i+1])
	}
	return r
}

// appendFolded returns the result of appending the case folding of the class x to the class r.
func appendFoldedClass(r []rune, x []rune) []rune {
	for i := 0; i < len(x); i += 2 {
		r = appendFoldedRange(r, x[i], x[i+1])
	}
	return r
}

// appendNegatedClass returns the result of appending the negation of the class x to the class r.
// It assumes x is clean.
func appendNegatedClass(r []rune, x []rune) []rune {
	nextLo := '\u0000'
	for i := 0; i < len(x); i += 2 {
		lo, hi := x[i], x[i+1]
		if nextLo <= lo-1 {
			r = appendRange(r, nextLo, lo-1)
		}
		nextLo = hi + 1
	}
	if nextLo <= unicode.MaxRune {
		r = appendRange(r, nextLo, unicode.MaxRune)
	}
	return r
}

// appendTable returns the result of appending x to the class r.
func appendTable(r []rune, x *unicode.RangeTable) []rune {
	for _, xr := range x.R16 {
		lo, hi, stride := rune(xr.Lo), rune(xr.Hi), rune(xr.Stride)
		if stride == 1 {
			r = appendRange(r, lo, hi)
			continue
		}
		for c := lo; c <= hi; c += stride {
			r = appendRange(r, c, c)
		}
	}
	for _, xr := range x.R32 {
		lo, hi, stride := rune(xr.Lo), rune(xr.Hi), rune(xr.Stride)
		if stride == 1 {
			r = appendRange(r, lo, hi)
			continue
		}
		for c := lo; c <= hi; c += stride {
			r = appendRange(r, c, c)
		}
	}
	return r
}

// appendNegatedTable returns the result of appending the negation of x to the class r.
func appendNegatedTable(r []rune, x *unicode.RangeTable) []rune {
	nextLo := '\u0000' // lo end of next class to add
	for _, xr := range x.R16 {
		lo, hi, stride := rune(xr.Lo), rune(xr.Hi), rune(xr.Stride)
		if stride == 1 {
			if nextLo <= lo-1 {
				r = appendRange(r, nextLo, lo-1)
			}
			nextLo = hi + 1
			continue
		}
		for c := lo; c <= hi; c += stride {
			if nextLo <= c-1 {
				r = appendRange(r, nextLo, c-1)
			}
			nextLo = c + 1
		}
	}
	for _, xr := range x.R32 {
		lo, hi, stride := rune(xr.Lo), rune(xr.Hi), rune(xr.Stride)
		if stride == 1 {
			if nextLo <= lo-1 {
				r = appendRange(r, nextLo, lo-1)
			}
			nextLo = hi + 1
			continue
		}
		for c := lo; c <= hi; c += stride {
			if nextLo <= c-1 {
				r = appendRange(r, nextLo, c-1)
			}
			nextLo = c + 1
		}
	}
	if nextLo <= unicode.MaxRune {
		r = appendRange(r, nextLo, unicode.MaxRune)
	}
	return r
}

// negateClass overwrites r and returns r's negation.
// It assumes the class r is already clean.
func negateClass(r []rune) []rune {
	nextLo := '\u0000' // lo end of next class to add
	w := 0             // write index
	for i := 0; i < len(r); i += 2 {
		lo, hi := r[i], r[i+1]
		if nextLo <= lo-1 {
			r[w] = nextLo
			r[w+1] = lo - 1
			w += 2
		}
		nextLo = hi + 1
	}
	r = r[:w]
	if nextLo <= unicode.MaxRune {
		// It's possible for the negation to have one more
		// range - this one - than the original class, so use append.
		r = append(r, nextLo, unicode.MaxRune)
	}
	return r
}

// ranges implements sort.Interface on a []rune.
// The choice of receiver type definition is strange
// but avoids an allocation since we already have
// a *[]rune.
type ranges struct {
	p *[]rune
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

func checkUTF8(s string) error {
	for s != "" {
		rune, size := utf8.DecodeRuneInString(s)
		if rune == utf8.RuneError && size == 1 {
			return &Error{Code: ErrInvalidUTF8, Expr: s}
		}
		s = s[size:]
	}
	return nil
}

func nextRune(s string) (c rune, t string, err error) {
	c, size := utf8.DecodeRuneInString(s)
	if c == utf8.RuneError && size == 1 {
		return 0, "", &Error{Code: ErrInvalidUTF8, Expr: s}
	}
	return c, s[size:], nil
}

func isalnum(c rune) bool {
	return '0' <= c && c <= '9' || 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z'
}

func unhex(c rune) rune {
	if '0' <= c && c <= '9' {
		return c - '0'
	}
	if 'a' <= c && c <= 'f' {
		return c - 'a' + 10
	}
	if 'A' <= c && c <= 'F' {
		return c - 'A' + 10
	}
	return -1
}
