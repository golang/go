// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package regexp implements regular expression search.
//
// The syntax of the regular expressions accepted is the same
// general syntax used by Perl, Python, and other languages.
// More precisely, it is the syntax accepted by RE2 and described at
// https://golang.org/s/re2syntax, except for \C.
// For an overview of the syntax, see the [regexp/syntax] package.
//
// The regexp implementation provided by this package is
// guaranteed to run in time linear in the size of the input.
// (This is a property not guaranteed by most open source
// implementations of regular expressions.) For more information
// about this property, see https://swtch.com/~rsc/regexp/regexp1.html
// or any book about automata theory.
//
// All characters are UTF-8-encoded code points.
// Following [utf8.DecodeRune], each byte of an invalid UTF-8 sequence
// is treated as if it encoded utf8.RuneError (U+FFFD).
//
// There are 24 methods of [Regexp] that match a regular expression and identify
// the matched text. Their names are matched by this regular expression:
//
//	(All|Find|FindAll)(String)?(Submatch)?(Index)?
//
// The ‘All’ variants return an iterator over successive non-overlapping
// matches of the entire expression. The ‘FindAll’ variants return a slice
// of those matches instead. Empty matches abutting a preceding
// match are ignored. The ‘FindAll’ variants take an extra integer argument, n.
// If n >= 0, the function returns at most n matches/submatches;
// otherwise, it returns all of them.
//
// The ‘Find’ variants return only the first match that All or FindAll would return.
//
// If ‘String’ is present, the argument is a string; otherwise it is a []byte.
//
// By default, each returned match is denoted by the substring matching the
// regular expression, of type string or []byte according to the type of the argument.
// If ‘Submatch’ is present, each match is represented instead by a slice of
// the substrings matching the regular expression's parenthesized subexpressions
// (also known as capturing groups), numbered from left to right in order of opening
// parenthesis. Submatch 0 is the match of the entire expression, submatch 1 is
// the match of the first parenthesized subexpression, and so on.
// If ‘Index’ is present, each substring is instead denoted by a pair of byte indexes
// within the input string. If an index is negative or substring is nil, it means that
// the subexpression did not match any string in the input. For ‘String’ versions,
// an empty string means either no match or an empty match.
//
// There is also a subset of the methods that can be applied to text read from
// an [io.RuneReader]: [Regexp.MatchReader], [Regexp.FindReaderIndex],
// [Regexp.FindReaderSubmatchIndex].
// Note that regular expression matches may need to
// examine text beyond the text returned by a match, so the methods that
// match text from an [io.RuneReader] may read arbitrarily far into the input
// before returning.
//
// (There are a few other methods that do not match this pattern.)
package regexp

import (
	"bytes"
	"io"
	"iter"
	"regexp/syntax"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// Regexp is the representation of a compiled regular expression.
// A Regexp is safe for concurrent use by multiple goroutines,
// except for configuration methods, such as [Regexp.Longest].
type Regexp struct {
	expr           string       // as passed to Compile
	prog           *syntax.Prog // compiled program
	onepass        *onePassProg // onepass program or nil
	numSubexp      int
	maxBitStateLen int
	subexpNames    []string
	prefix         string         // required prefix in unanchored matches
	prefixBytes    []byte         // prefix, as a []byte
	prefixRune     rune           // first rune in prefix
	prefixEnd      uint32         // pc for last rune in prefix
	mpool          int            // pool for machines
	matchcap       int            // size of recorded match lengths
	prefixComplete bool           // prefix is the entire regexp
	cond           syntax.EmptyOp // empty-width conditions required at start of match
	minInputLen    int            // minimum length of the input in bytes

	// This field can be modified by the Longest method,
	// but it is otherwise read-only.
	longest bool // whether regexp prefers leftmost-longest match
}

// String returns the source text used to compile the regular expression.
func (re *Regexp) String() string {
	return re.expr
}

// Copy returns a new [Regexp] object copied from re.
// Calling [Regexp.Longest] on one copy does not affect another.
//
// Deprecated: In earlier releases, when using a [Regexp] in multiple goroutines,
// giving each goroutine its own copy helped to avoid lock contention.
// As of Go 1.12, using Copy is no longer necessary to avoid lock contention.
// Copy may still be appropriate if the reason for its use is to make
// two copies with different [Regexp.Longest] settings.
func (re *Regexp) Copy() *Regexp {
	re2 := *re
	return &re2
}

// Compile parses a regular expression and returns, if successful,
// a [Regexp] object that can be used to match against text.
//
// When matching against text, the regexp returns a match that
// begins as early as possible in the input (leftmost), and among those
// it chooses the one that a backtracking search would have found first.
// This so-called leftmost-first matching is the same semantics
// that Perl, Python, and other implementations use, although this
// package implements it without the expense of backtracking.
// For POSIX leftmost-longest matching, see [CompilePOSIX].
func Compile(expr string) (*Regexp, error) {
	return compile(expr, syntax.Perl, false)
}

// CompilePOSIX is like [Compile] but restricts the regular expression
// to POSIX ERE (egrep) syntax and changes the match semantics to
// leftmost-longest.
//
// That is, when matching against text, the regexp returns a match that
// begins as early as possible in the input (leftmost), and among those
// it chooses a match that is as long as possible.
// This so-called leftmost-longest matching is the same semantics
// that early regular expression implementations used and that POSIX
// specifies.
//
// However, there can be multiple leftmost-longest matches, with different
// submatch choices, and here this package diverges from POSIX.
// Among the possible leftmost-longest matches, this package chooses
// the one that a backtracking search would have found first, while POSIX
// specifies that the match be chosen to maximize the length of the first
// subexpression, then the second, and so on from left to right.
// The POSIX rule is computationally prohibitive and not even well-defined.
// See https://swtch.com/~rsc/regexp/regexp2.html#posix for details.
func CompilePOSIX(expr string) (*Regexp, error) {
	return compile(expr, syntax.POSIX, true)
}

// Longest makes future searches prefer the leftmost-longest match.
// That is, when matching against text, the regexp returns a match that
// begins as early as possible in the input (leftmost), and among those
// it chooses a match that is as long as possible.
// This method modifies the [Regexp] and may not be called concurrently
// with any other methods.
func (re *Regexp) Longest() {
	re.longest = true
}

func compile(expr string, mode syntax.Flags, longest bool) (*Regexp, error) {
	re, err := syntax.Parse(expr, mode)
	if err != nil {
		return nil, err
	}
	maxCap := re.MaxCap()
	capNames := re.CapNames()

	re = re.Simplify()
	prog, err := syntax.Compile(re)
	if err != nil {
		return nil, err
	}
	matchcap := prog.NumCap
	if matchcap < 2 {
		matchcap = 2
	}
	regexp := &Regexp{
		expr:        expr,
		prog:        prog,
		onepass:     compileOnePass(prog),
		numSubexp:   maxCap,
		subexpNames: capNames,
		cond:        prog.StartCond(),
		longest:     longest,
		matchcap:    matchcap,
		minInputLen: minInputLen(re),
	}
	if regexp.onepass == nil {
		regexp.prefix, regexp.prefixComplete = prog.Prefix()
		regexp.maxBitStateLen = maxBitStateLen(prog)
	} else {
		regexp.prefix, regexp.prefixComplete, regexp.prefixEnd = onePassPrefix(prog)
	}
	if regexp.prefix != "" {
		// TODO(rsc): Remove this allocation by adding
		// IndexString to package bytes.
		regexp.prefixBytes = []byte(regexp.prefix)
		regexp.prefixRune, _ = utf8.DecodeRuneInString(regexp.prefix)
	}

	n := len(prog.Inst)
	i := 0
	for matchSize[i] != 0 && matchSize[i] < n {
		i++
	}
	regexp.mpool = i

	return regexp, nil
}

// Pools of *machine for use during (*Regexp).find,
// split up by the size of the execution queues.
// matchPool[i] machines have queue size matchSize[i].
// On a 64-bit system each queue entry is 16 bytes,
// so matchPool[0] has 16*2*128 = 4kB queues, etc.
// The final matchPool is a catch-all for very large queues.
var (
	matchSize = [...]int{128, 512, 2048, 16384, 0}
	matchPool [len(matchSize)]sync.Pool
)

// get returns a machine to use for matching re.
// It uses the re's machine cache if possible, to avoid
// unnecessary allocation.
func (re *Regexp) get() *machine {
	m, ok := matchPool[re.mpool].Get().(*machine)
	if !ok {
		m = new(machine)
	}
	m.re = re
	m.p = re.prog
	if cap(m.matchcap) < re.matchcap {
		m.matchcap = make([]int, re.matchcap)
		for _, t := range m.pool {
			t.cap = make([]int, re.matchcap)
		}
	}

	// Allocate queues if needed.
	// Or reallocate, for "large" match pool.
	n := matchSize[re.mpool]
	if n == 0 { // large pool
		n = len(re.prog.Inst)
	}
	if len(m.q0.sparse) < n {
		m.q0 = queue{make([]uint32, n), make([]entry, 0, n)}
		m.q1 = queue{make([]uint32, n), make([]entry, 0, n)}
	}
	return m
}

// put returns a machine to the correct machine pool.
func (re *Regexp) put(m *machine) {
	m.re = nil
	m.p = nil
	m.inputs.clear()
	matchPool[re.mpool].Put(m)
}

// minInputLen walks the regexp to find the minimum length of any matchable input.
func minInputLen(re *syntax.Regexp) int {
	switch re.Op {
	default:
		return 0
	case syntax.OpAnyChar, syntax.OpAnyCharNotNL, syntax.OpCharClass:
		return 1
	case syntax.OpLiteral:
		l := 0
		for _, r := range re.Rune {
			if r == utf8.RuneError {
				l++
			} else {
				l += utf8.RuneLen(r)
			}
		}
		return l
	case syntax.OpCapture, syntax.OpPlus:
		return minInputLen(re.Sub[0])
	case syntax.OpRepeat:
		return re.Min * minInputLen(re.Sub[0])
	case syntax.OpConcat:
		l := 0
		for _, sub := range re.Sub {
			l += minInputLen(sub)
		}
		return l
	case syntax.OpAlternate:
		l := minInputLen(re.Sub[0])
		var lnext int
		for _, sub := range re.Sub[1:] {
			lnext = minInputLen(sub)
			if lnext < l {
				l = lnext
			}
		}
		return l
	}
}

// MustCompile is like [Compile] but panics if the expression cannot be parsed.
// It simplifies safe initialization of global variables holding compiled regular
// expressions.
func MustCompile(str string) *Regexp {
	regexp, err := Compile(str)
	if err != nil {
		panic(`regexp: Compile(` + quote(str) + `): ` + err.Error())
	}
	return regexp
}

// MustCompilePOSIX is like [CompilePOSIX] but panics if the expression cannot be parsed.
// It simplifies safe initialization of global variables holding compiled regular
// expressions.
func MustCompilePOSIX(str string) *Regexp {
	regexp, err := CompilePOSIX(str)
	if err != nil {
		panic(`regexp: CompilePOSIX(` + quote(str) + `): ` + err.Error())
	}
	return regexp
}

func quote(s string) string {
	if strconv.CanBackquote(s) {
		return "`" + s + "`"
	}
	return strconv.Quote(s)
}

// NumSubexp returns the number of parenthesized subexpressions in this [Regexp].
func (re *Regexp) NumSubexp() int {
	return re.numSubexp
}

// SubexpNames returns the names of the parenthesized subexpressions
// in this [Regexp]. The name for the first sub-expression is names[1],
// so that if m is a match slice, the name for m[i] is SubexpNames()[i].
// Since the Regexp as a whole cannot be named, names[0] is always
// the empty string. The slice should not be modified.
func (re *Regexp) SubexpNames() []string {
	return re.subexpNames
}

// SubexpIndex returns the index of the first subexpression with the given name,
// or -1 if there is no subexpression with that name.
//
// Note that multiple subexpressions can be written using the same name, as in
// (?P<bob>a+)(?P<bob>b+), which declares two subexpressions named "bob".
// In this case, SubexpIndex returns the index of the leftmost such subexpression
// in the regular expression.
func (re *Regexp) SubexpIndex(name string) int {
	if name != "" {
		for i, s := range re.subexpNames {
			if name == s {
				return i
			}
		}
	}
	return -1
}

const endOfText rune = -1

// input abstracts different representations of the input text. It provides
// one-character lookahead.
type input interface {
	step(pos int) (r rune, width int) // advance one rune
	canCheckPrefix() bool             // can we look ahead without losing info?
	hasPrefix(re *Regexp) bool
	index(re *Regexp, pos int) int
	context(pos int) lazyFlag
}

// inputString scans a string.
type inputString struct {
	str string
}

func (i *inputString) step(pos int) (rune, int) {
	if pos < len(i.str) {
		return utf8.DecodeRuneInString(i.str[pos:])
	}
	return endOfText, 0
}

func (i *inputString) canCheckPrefix() bool {
	return true
}

func (i *inputString) hasPrefix(re *Regexp) bool {
	return strings.HasPrefix(i.str, re.prefix)
}

func (i *inputString) index(re *Regexp, pos int) int {
	return strings.Index(i.str[pos:], re.prefix)
}

func (i *inputString) context(pos int) lazyFlag {
	r1, r2 := endOfText, endOfText
	// 0 < pos && pos <= len(i.str)
	if uint(pos-1) < uint(len(i.str)) {
		r1, _ = utf8.DecodeLastRuneInString(i.str[:pos])
	}
	// 0 <= pos && pos < len(i.str)
	if uint(pos) < uint(len(i.str)) {
		r2, _ = utf8.DecodeRuneInString(i.str[pos:])
	}
	return newLazyFlag(r1, r2)
}

// inputBytes scans a byte slice.
type inputBytes struct {
	str []byte
}

func (i *inputBytes) step(pos int) (rune, int) {
	if pos < len(i.str) {
		return utf8.DecodeRune(i.str[pos:])
	}
	return endOfText, 0
}

func (i *inputBytes) canCheckPrefix() bool {
	return true
}

func (i *inputBytes) hasPrefix(re *Regexp) bool {
	return bytes.HasPrefix(i.str, re.prefixBytes)
}

func (i *inputBytes) index(re *Regexp, pos int) int {
	return bytes.Index(i.str[pos:], re.prefixBytes)
}

func (i *inputBytes) context(pos int) lazyFlag {
	r1, r2 := endOfText, endOfText
	// 0 < pos && pos <= len(i.str)
	if uint(pos-1) < uint(len(i.str)) {
		r1, _ = utf8.DecodeLastRune(i.str[:pos])
	}
	// 0 <= pos && pos < len(i.str)
	if uint(pos) < uint(len(i.str)) {
		r2, _ = utf8.DecodeRune(i.str[pos:])
	}
	return newLazyFlag(r1, r2)
}

// inputReader scans a RuneReader.
type inputReader struct {
	r     io.RuneReader
	atEOT bool
	pos   int
}

func (i *inputReader) step(pos int) (rune, int) {
	if !i.atEOT && pos != i.pos {
		return endOfText, 0

	}
	r, w, err := i.r.ReadRune()
	if err != nil {
		i.atEOT = true
		return endOfText, 0
	}
	i.pos += w
	return r, w
}

func (i *inputReader) canCheckPrefix() bool {
	return false
}

func (i *inputReader) hasPrefix(re *Regexp) bool {
	return false
}

func (i *inputReader) index(re *Regexp, pos int) int {
	return -1
}

func (i *inputReader) context(pos int) lazyFlag {
	return 0 // not used
}

// LiteralPrefix returns a literal string that must begin any match
// of the regular expression re. It returns the boolean true if the
// literal string comprises the entire regular expression.
func (re *Regexp) LiteralPrefix() (prefix string, complete bool) {
	return re.prefix, re.prefixComplete
}

// MatchReader reports whether the text returned by the [io.RuneReader]
// contains any match of the regular expression re.
func (re *Regexp) MatchReader(r io.RuneReader) bool {
	return re.doMatch(r, nil, "")
}

// MatchString reports whether the string s
// contains any match of the regular expression re.
func (re *Regexp) MatchString(s string) bool {
	return re.doMatch(nil, nil, s)
}

// Match reports whether the byte slice b
// contains any match of the regular expression re.
func (re *Regexp) Match(b []byte) bool {
	return re.doMatch(nil, b, "")
}

// MatchReader reports whether the text returned by the [io.RuneReader]
// contains any match of the regular expression pattern.
// More complicated queries need to use [Compile] and the full [Regexp] interface.
func MatchReader(pattern string, r io.RuneReader) (matched bool, err error) {
	re, err := Compile(pattern)
	if err != nil {
		return false, err
	}
	return re.MatchReader(r), nil
}

// MatchString reports whether the string s
// contains any match of the regular expression pattern.
// More complicated queries need to use [Compile] and the full [Regexp] interface.
func MatchString(pattern string, s string) (matched bool, err error) {
	re, err := Compile(pattern)
	if err != nil {
		return false, err
	}
	return re.MatchString(s), nil
}

// Match reports whether the byte slice b
// contains any match of the regular expression pattern.
// More complicated queries need to use [Compile] and the full [Regexp] interface.
func Match(pattern string, b []byte) (matched bool, err error) {
	re, err := Compile(pattern)
	if err != nil {
		return false, err
	}
	return re.Match(b), nil
}

// ReplaceAllString returns a copy of src, replacing matches of the [Regexp]
// with the replacement string repl.
// Inside repl, $ signs are interpreted as in [Regexp.Expand].
func (re *Regexp) ReplaceAllString(src, repl string) string {
	n := 2
	if strings.Contains(repl, "$") {
		n = 2 * (re.numSubexp + 1)
	}
	b := re.replaceAll(nil, src, n, func(dst []byte, match []int) []byte {
		return re.expand(dst, repl, nil, src, match)
	})
	return string(b)
}

// ReplaceAllLiteralString returns a copy of src, replacing matches of the [Regexp]
// with the replacement string repl. The replacement repl is substituted directly,
// without using [Regexp.Expand].
func (re *Regexp) ReplaceAllLiteralString(src, repl string) string {
	return string(re.replaceAll(nil, src, 2, func(dst []byte, match []int) []byte {
		return append(dst, repl...)
	}))
}

// ReplaceAllStringFunc returns a copy of src in which all matches of the
// [Regexp] have been replaced by the return value of function repl applied
// to the matched substring. The replacement returned by repl is substituted
// directly, without using [Regexp.Expand].
func (re *Regexp) ReplaceAllStringFunc(src string, repl func(string) string) string {
	b := re.replaceAll(nil, src, 2, func(dst []byte, match []int) []byte {
		return append(dst, repl(src[match[0]:match[1]])...)
	})
	return string(b)
}

func (re *Regexp) replaceAll(bsrc []byte, src string, nmatch int, repl func(dst []byte, m []int) []byte) []byte {
	lastMatchEnd := 0 // end position of the most recent match
	searchPos := 0    // position where we next look for a match
	var buf []byte
	var endPos int
	if bsrc != nil {
		endPos = len(bsrc)
	} else {
		endPos = len(src)
	}
	if nmatch > re.prog.NumCap {
		nmatch = re.prog.NumCap
	}

	var dstCap [2]int
	for searchPos <= endPos {
		a := re.find(nil, bsrc, src, searchPos, nmatch, dstCap[:0])
		if len(a) == 0 {
			break // no more matches
		}

		// Copy the unmatched characters before this match.
		if bsrc != nil {
			buf = append(buf, bsrc[lastMatchEnd:a[0]]...)
		} else {
			buf = append(buf, src[lastMatchEnd:a[0]]...)
		}

		// Now insert a copy of the replacement string, but not for a
		// match of the empty string immediately after another match.
		// (Otherwise, we get double replacement for patterns that
		// match both empty and nonempty strings.)
		if a[1] > lastMatchEnd || a[0] == 0 {
			buf = repl(buf, a)
		}
		lastMatchEnd = a[1]

		// Advance past this match; always advance at least one character.
		var width int
		if bsrc != nil {
			_, width = utf8.DecodeRune(bsrc[searchPos:])
		} else {
			_, width = utf8.DecodeRuneInString(src[searchPos:])
		}
		if searchPos+width > a[1] {
			searchPos += width
		} else if searchPos+1 > a[1] {
			// This clause is only needed at the end of the input
			// string. In that case, DecodeRuneInString returns width=0.
			searchPos++
		} else {
			searchPos = a[1]
		}
	}

	// Copy the unmatched characters after the last match.
	if bsrc != nil {
		buf = append(buf, bsrc[lastMatchEnd:]...)
	} else {
		buf = append(buf, src[lastMatchEnd:]...)
	}

	return buf
}

// ReplaceAll returns a copy of src, replacing matches of the [Regexp]
// with the replacement text repl.
// Inside repl, $ signs are interpreted as in [Regexp.Expand].
func (re *Regexp) ReplaceAll(src, repl []byte) []byte {
	n := 2
	if bytes.IndexByte(repl, '$') >= 0 {
		n = 2 * (re.numSubexp + 1)
	}
	srepl := ""
	b := re.replaceAll(src, "", n, func(dst []byte, match []int) []byte {
		if len(srepl) != len(repl) {
			srepl = string(repl)
		}
		return re.expand(dst, srepl, src, "", match)
	})
	return b
}

// ReplaceAllLiteral returns a copy of src, replacing matches of the [Regexp]
// with the replacement bytes repl. The replacement repl is substituted directly,
// without using [Regexp.Expand].
func (re *Regexp) ReplaceAllLiteral(src, repl []byte) []byte {
	return re.replaceAll(src, "", 2, func(dst []byte, match []int) []byte {
		return append(dst, repl...)
	})
}

// ReplaceAllFunc returns a copy of src in which all matches of the
// [Regexp] have been replaced by the return value of function repl applied
// to the matched byte slice. The replacement returned by repl is substituted
// directly, without using [Regexp.Expand].
func (re *Regexp) ReplaceAllFunc(src []byte, repl func([]byte) []byte) []byte {
	return re.replaceAll(src, "", 2, func(dst []byte, match []int) []byte {
		return append(dst, repl(src[match[0]:match[1]])...)
	})
}

// Bitmap used by func special to check whether a character needs to be escaped.
var specialBytes [16]byte

// special reports whether byte b needs to be escaped by QuoteMeta.
func special(b byte) bool {
	return b < utf8.RuneSelf && specialBytes[b%16]&(1<<(b/16)) != 0
}

func init() {
	for _, b := range []byte(`\.+*?()|[]{}^$`) {
		specialBytes[b%16] |= 1 << (b / 16)
	}
}

// QuoteMeta returns a string that escapes all regular expression metacharacters
// inside the argument text; the returned string is a regular expression matching
// the literal text.
func QuoteMeta(s string) string {
	// A byte loop is correct because all metacharacters are ASCII.
	var i int
	for i = 0; i < len(s); i++ {
		if special(s[i]) {
			break
		}
	}
	// No meta characters found, so return original string.
	if i >= len(s) {
		return s
	}

	b := make([]byte, 2*len(s)-i)
	copy(b, s[:i])
	j := i
	for ; i < len(s); i++ {
		if special(s[i]) {
			b[j] = '\\'
			j++
		}
		b[j] = s[i]
		j++
	}
	return string(b[:j])
}

// The number of capture values in the program may correspond
// to fewer capturing expressions than are in the regexp.
// For example, "(a){0}" turns into an empty program, so the
// maximum capture in the program is 0 but we need to return
// an expression for \1.  Pad appends -1s to the slice a as needed.
func (re *Regexp) pad(a []int) []int {
	if a == nil {
		// No match.
		return nil
	}
	n := (1 + re.numSubexp) * 2
	for len(a) < n {
		a = append(a, -1)
	}
	return a
}

// matches yields the location of successive matches in the input text.
// The input text is b if non-nil, otherwise s.
func (re *Regexp) matches(s string, b []byte, max, ncap int) iter.Seq[[]int] {
	return func(yield func([]int) bool) {
		if max == 0 {
			return
		}
		var end int
		if b == nil {
			end = len(s)
		} else {
			end = len(b)
		}
		var matches []int
		for pos, prevMatchEnd := 0, -1; pos <= end; {
			matches = re.find(nil, b, s, pos, ncap, matches[:0])
			if len(matches) == 0 {
				break
			}

			accept := true
			if matches[1] == pos {
				// We've found an empty match.
				if matches[0] == prevMatchEnd {
					// We don't allow an empty match right
					// after a previous match, so ignore it.
					accept = false
				}
				var width int
				if b == nil {
					is := inputString{str: s}
					_, width = is.step(pos)
				} else {
					ib := inputBytes{str: b}
					_, width = ib.step(pos)
				}
				if width > 0 {
					pos += width
				} else {
					pos = end + 1
				}
			} else {
				pos = matches[1]
			}
			prevMatchEnd = matches[1]

			if accept {
				if !yield(re.pad(matches)) {
					return
				}
				if max > 0 {
					if max--; max == 0 {
						return
					}
				}
			}
		}
	}
}

// Find returns the text of the leftmost match for re in b.
// The return value is nil for no match.
func (re *Regexp) Find(b []byte) []byte {
	var dstCap [2]int
	a := re.find(nil, b, "", 0, 2, dstCap[:0])
	if a == nil {
		return nil
	}
	return b[a[0]:a[1]:a[1]]
}

// FindString returns the text of the leftmost match for re in s.
// The return value is the empty string both for an empty match and for no match.
// To distinguish those two cases, use [Regexp.FindStringIndex] or [Regexp.FindStringSubmatch].
func (re *Regexp) FindString(s string) string {
	var dstCap [2]int
	a := re.find(nil, nil, s, 0, 2, dstCap[:0])
	if a == nil {
		return ""
	}
	return s[a[0]:a[1]]
}

// FindIndex returns the location of the leftmost match for re in b.
// The match itself is at b[m[0]:m[1]].
// The return value is nil for no match.
func (re *Regexp) FindIndex(b []byte) (m []int) {
	m = re.find(nil, b, "", 0, 2, nil)
	if m == nil {
		return nil
	}
	return m[0:2]
}

// FindStringIndex returns the location of the leftmost match for re in s.
// The match itself is at s[m[0]:m[1]].
// The return value is nil for no match.
func (re *Regexp) FindStringIndex(s string) (m []int) {
	m = re.find(nil, nil, s, 0, 2, nil)
	if m == nil {
		return nil
	}
	return m[0:2]
}

// FindReaderIndex returns the location of the leftmost match for re in r.
// The match starts at byte index m[0] and ends just before byte index m[1].
// The return value is nil for no match.
//
// FindReaderIndex may read arbitrarily far from r,
// including reading beyond the returned match.
func (re *Regexp) FindReaderIndex(r io.RuneReader) (m []int) {
	m = re.find(r, nil, "", 0, 2, nil)
	if m == nil {
		return nil
	}
	return m[0:2]
}

// FindSubmatch returns the first match for re in b, including submatches.
// The overall match is m[0], the first submatch is m[1], and so on.
// The return value is nil for no match.
func (re *Regexp) FindSubmatch(b []byte) [][]byte {
	var dstCap [4]int
	m := re.find(nil, b, "", 0, re.prog.NumCap, dstCap[:0])
	if m == nil {
		return nil
	}
	sub := make([][]byte, 1+re.numSubexp)
	for i := range sub {
		if 2*i < len(m) && m[2*i] >= 0 {
			sub[i] = b[m[2*i]:m[2*i+1]:m[2*i+1]]
		}
	}
	return sub
}

// FindStringSubmatch returns the first match for re in s, including submatches.
// The overall match is s[0], the first submatch is s[1], and so on.
// The return value is nil for no match.
func (re *Regexp) FindStringSubmatch(s string) []string {
	var dstCap [4]int
	a := re.find(nil, nil, s, 0, re.prog.NumCap, dstCap[:0])
	if a == nil {
		return nil
	}
	ret := make([]string, 1+re.numSubexp)
	for i := range ret {
		if 2*i < len(a) && a[2*i] >= 0 {
			ret[i] = s[a[2*i]:a[2*i+1]]
		}
	}
	return ret
}

// FindSubmatchIndex returns the first match for re in b, including submatches.
// The overall match is b[m[0]:m[1]], the first submatch is b[m[2]:m[3]], and so on.
// The return value is nil for no match.
func (re *Regexp) FindSubmatchIndex(b []byte) []int {
	return re.pad(re.find(nil, b, "", 0, re.prog.NumCap, nil))
}

// FindStringSubmatchIndex returns the first match for re in s, including submatches.
// The overall match is s[m[0]:m[1]], the first submatch is s[m[2]:m[3]], and so on.
// The return value is nil for no match.
func (re *Regexp) FindStringSubmatchIndex(s string) []int {
	return re.pad(re.find(nil, nil, s, 0, re.prog.NumCap, nil))
}

// FindReaderSubmatchIndex returns the first match for re in r, including submatches.
// The overall match is at byte index m[0] up to m[1],
// the first submatch is at byte index m[2] up to m[3], and so on.
// The return value is nil for no match.
//
// FindReaderSubmatchIndex may read arbitrarily far from r,
// including reading beyond the returned match.
func (re *Regexp) FindReaderSubmatchIndex(r io.RuneReader) []int {
	return re.pad(re.find(r, nil, "", 0, re.prog.NumCap, nil))
}

// all returns at most n matches for re in b.
func (re *Regexp) all(b []byte, n int) iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		for m := range re.matches("", b, n, 2) {
			if !yield(b[m[0]:m[1]:m[1]]) {
				break
			}
		}
	}
}

// allString returns at most n matches for re in s.
func (re *Regexp) allString(s string, n int) iter.Seq[string] {
	return func(yield func(string) bool) {
		for m := range re.matches(s, nil, n, 2) {
			if !yield(s[m[0]:m[1]]) {
				break
			}
		}
	}
}

// allIndex returns the locations of at most n matches for re in b.
func (re *Regexp) allIndex(b []byte, n int) iter.Seq[[]int] {
	return func(yield func([]int) bool) {
		for m := range re.matches("", b, n, 2) {
			if !yield([]int{m[0], m[1]}) {
				break
			}
		}
	}
}

// allStringIndex returns the locations of at most n matches for re in s.
func (re *Regexp) allStringIndex(s string, n int) iter.Seq[[]int] {
	return func(yield func([]int) bool) {
		for m := range re.matches(s, nil, n, 2) {
			if !yield([]int{m[0], m[1]}) {
				break
			}
		}
	}
}

// allSubmatch returns the locations of at most n matches for re in b,
// including submatch locations.
func (re *Regexp) allSubmatch(b []byte, n int) iter.Seq[[][]byte] {
	return func(yield func([][]byte) bool) {
		for m := range re.matches("", b, n, re.prog.NumCap) {
			sub := make([][]byte, len(m)/2)
			for i := range sub {
				if m[2*i] >= 0 {
					sub[i] = b[m[2*i]:m[2*i+1]:m[2*i+1]]
				}
			}
			if !yield(sub) {
				break
			}
		}
	}
}

// allStringSubmatch returns the locations of at most n matches for re in s,
// including submatch locations.
func (re *Regexp) allStringSubmatch(s string, n int) iter.Seq[[]string] {
	return func(yield func([]string) bool) {
		for m := range re.matches(s, nil, n, re.prog.NumCap) {
			sub := make([]string, len(m)/2)
			for i := range sub {
				if m[2*i] >= 0 {
					sub[i] = s[m[2*i]:m[2*i+1]]
				}
			}
			if !yield(sub) {
				break
			}
		}
	}
}

// allSubmatchIndex returns the locations of at most n matches for re in b,
// including submatch locations.
func (re *Regexp) allSubmatchIndex(b []byte, n int) iter.Seq[[]int] {
	return func(yield func([]int) bool) {
		for m := range re.matches("", b, n, re.prog.NumCap) {
			if !yield(slices.Clone(m)) {
				break
			}
		}
	}
}

// allStringSubmatchIndex returns the locations of at most n matches for re in s,
// including submatch locations.
func (re *Regexp) allStringSubmatchIndex(s string, n int) iter.Seq[[]int] {
	return func(yield func([]int) bool) {
		for m := range re.matches(s, nil, n, re.prog.NumCap) {
			if !yield(slices.Clone(m)) {
				break
			}
		}
	}
}

// All returns all the matches for re in b.
func (re *Regexp) _All(b []byte) iter.Seq[[]byte] {
	return re.all(b, -1)
}

// AllString returns all the matches for re in s.
func (re *Regexp) _AllString(s string) iter.Seq[string] {
	return re.allString(s, -1)
}

// AllIndex returns the locations of all matches for re in b.
func (re *Regexp) _AllIndex(b []byte) iter.Seq[[]int] {
	return re.allIndex(b, -1)
}

// AllStringIndex returns the locations of all matches for re in s.
func (re *Regexp) _AllStringIndex(s string) iter.Seq[[]int] {
	return re.allStringIndex(s, -1)
}

// AllSubmatch returns the locations of all matches for re in b,
// including submatch locations.
// In each returned match m, the overall match is m[0],
// the first submatch is m[1], and so on.
func (re *Regexp) _AllSubmatch(b []byte) iter.Seq[[][]byte] {
	return re.allSubmatch(b, -1)
}

// AllStringSubmatch returns the locations of all matches for re in s,
// including submatch locations.
// In each returned match m, m[0] is the overall match,
// m[1] is the first submatch, and so on.
func (re *Regexp) _AllStringSubmatch(s string) iter.Seq[[]string] {
	return re.allStringSubmatch(s, -1)
}

// AllSubmatchIndex returns the locations of all matches for re in b,
// including submatch locations.
// In each returned match m, the overall match is b[m[0]:m[1]],
// the first submatch is b[m[2]:m[3]], and so on.
func (re *Regexp) _AllSubmatchIndex(b []byte) iter.Seq[[]int] {
	return re.allSubmatchIndex(b, -1)
}

// AllStringSubmatchIndex returns the locations of all matches for re in s,
// including submatch locations.
// In each returned match m, the overall match is s[m[0]:m[1]],
// the first submatch is s[m[2]:m[3]], and so on.
func (re *Regexp) _AllStringSubmatchIndex(s string) iter.Seq[[]int] {
	return re.allStringSubmatchIndex(s, -1)
}

// FindAll returns all the matches for re in b.
// If n >= 0, FindAll returns no more than n matches.
// See [Regexp.All] for the equivalent iterator form.
func (re *Regexp) FindAll(b []byte, n int) [][]byte {
	return slices.Collect(re.all(b, n))
}

// FindAllString returns all the matches for re in s.
// If n >= 0, FindAllString returns no more than n matches.
// See [Regexp.AllString] for the equivalent iterator form.
func (re *Regexp) FindAllString(s string, n int) []string {
	return slices.Collect(re.allString(s, n))
}

// FindAllIndex returns the locations of all matches for re in b.
// If n >= 0, FindAllIndex returns no more than n matches.
// See [Regexp.AllIndex] for the equivalent iterator form.
func (re *Regexp) FindAllIndex(b []byte, n int) [][]int {
	return slices.Collect(re.allIndex(b, n))
}

// FindAllStringIndex returns the locations of all matches for re in s.
// If n >= 0, FindAllStringIndex returns no more than n matches.
// See [Regexp.AllStringIndex] for the equivalent iterator form.
func (re *Regexp) FindAllStringIndex(s string, n int) [][]int {
	return slices.Collect(re.allStringIndex(s, n))
}

// FindAllSubmatch returns the locations of all matches for re in b,
// including submatch locations.
// In each returned match m, the overall match is m[0],
// the first submatch is m[1], and so on.
// If n >= 0, FindAllSubmatch returns no more than n matches.
// See [Regexp.AllSubmatch] for the equivalent iterator form.
func (re *Regexp) FindAllSubmatch(b []byte, n int) [][][]byte {
	return slices.Collect(re.allSubmatch(b, n))
}

// FindAllStringSubmatch returns the locations of all matches for re in s,
// including submatch locations.
// In each returned match m, m[0] is the overall match,
// m[1] is the first submatch, and so on.
// If n >= 0, FindAllStringSubmatch returns no more than n matches.
// See [Regexp.AllStringSubmatch] for the equivalent iterator form.
func (re *Regexp) FindAllStringSubmatch(s string, n int) [][]string {
	return slices.Collect(re.allStringSubmatch(s, n))
}

// FindAllSubmatchIndex returns the locations of all matches for re in b,
// including submatch locations.
// In each returned match m, the overall match is b[m[0]:m[1]],
// the first submatch is b[m[2]:m[3]], and so on.
// If n >= 0, FindAllSubmatchIndex returns no more than n matches.
// See [Regexp.AllSubmatchIndex] for the equivalent iterator form.
func (re *Regexp) FindAllSubmatchIndex(b []byte, n int) [][]int {
	return slices.Collect(re.allSubmatchIndex(b, n))
}

// FindAllStringSubmatchIndex returns the locations of all matches for re in s,
// including submatch locations.
// In each returned match m, the overall match is s[m[0]:m[1]],
// the first submatch is s[m[2]:m[3]], and so on.
// If n >= 0, FindAllStringSubmatchIndex returns no more than n matches.
// See [Regexp.AllStringSubmatchIndex] for the equivalent iterator form.
func (re *Regexp) FindAllStringSubmatchIndex(s string, n int) [][]int {
	return slices.Collect(re.allStringSubmatchIndex(s, n))
}

// Expand appends template to dst and returns the result; during the
// append, Expand replaces variables in the template with corresponding
// matches drawn from src. The match slice should have been returned by
// [Regexp.FindSubmatchIndex].
//
// In the template, a variable is denoted by a substring of the form
// $name or ${name}, where name is a non-empty sequence of letters,
// digits, and underscores. A purely numeric name like $1 refers to
// the submatch with the corresponding index; other names refer to
// capturing parentheses named with the (?P<name>...) syntax. A
// reference to an out of range or unmatched index or a name that is not
// present in the regular expression is replaced with an empty slice.
//
// In the $name form, name is taken to be as long as possible: $1x is
// equivalent to ${1x}, not ${1}x, and, $10 is equivalent to ${10}, not ${1}0.
//
// To insert a literal $ in the output, use $$ in the template.
func (re *Regexp) Expand(dst []byte, template []byte, src []byte, match []int) []byte {
	return re.expand(dst, string(template), src, "", match)
}

// ExpandString is like [Regexp.Expand] but the template and source are strings.
// It appends to and returns a byte slice in order to give the calling
// code control over allocation.
func (re *Regexp) ExpandString(dst []byte, template string, src string, match []int) []byte {
	return re.expand(dst, template, nil, src, match)
}

func (re *Regexp) expand(dst []byte, template string, bsrc []byte, src string, match []int) []byte {
	for len(template) > 0 {
		before, after, ok := strings.Cut(template, "$")
		if !ok {
			break
		}
		dst = append(dst, before...)
		template = after
		if template != "" && template[0] == '$' {
			// Treat $$ as $.
			dst = append(dst, '$')
			template = template[1:]
			continue
		}
		name, num, rest, ok := extract(template)
		if !ok {
			// Malformed; treat $ as raw text.
			dst = append(dst, '$')
			continue
		}
		template = rest
		if num >= 0 {
			if 2*num+1 < len(match) && match[2*num] >= 0 {
				if bsrc != nil {
					dst = append(dst, bsrc[match[2*num]:match[2*num+1]]...)
				} else {
					dst = append(dst, src[match[2*num]:match[2*num+1]]...)
				}
			}
		} else {
			for i, namei := range re.subexpNames {
				if name == namei && 2*i+1 < len(match) && match[2*i] >= 0 {
					if bsrc != nil {
						dst = append(dst, bsrc[match[2*i]:match[2*i+1]]...)
					} else {
						dst = append(dst, src[match[2*i]:match[2*i+1]]...)
					}
					break
				}
			}
		}
	}
	dst = append(dst, template...)
	return dst
}

// extract returns the name from a leading "name" or "{name}" in str.
// (The $ has already been removed by the caller.)
// If it is a number, extract returns num set to that number; otherwise num = -1.
func extract(str string) (name string, num int, rest string, ok bool) {
	if str == "" {
		return
	}
	brace := false
	if str[0] == '{' {
		brace = true
		str = str[1:]
	}
	i := 0
	for i < len(str) {
		rune, size := utf8.DecodeRuneInString(str[i:])
		if !unicode.IsLetter(rune) && !unicode.IsDigit(rune) && rune != '_' {
			break
		}
		i += size
	}
	if i == 0 {
		// empty name is not okay
		return
	}
	name = str[:i]
	if brace {
		if i >= len(str) || str[i] != '}' {
			// missing closing brace
			return
		}
		i++
	}

	// Parse number.
	num = 0
	for i := 0; i < len(name); i++ {
		if name[i] < '0' || '9' < name[i] || num >= 1e8 {
			num = -1
			break
		}
		num = num*10 + int(name[i]) - '0'
	}
	// Disallow leading zeros.
	if name[0] == '0' && len(name) > 1 {
		num = -1
	}

	rest = str[i:]
	ok = true
	return
}

// Split slices s into substrings separated by the expression and returns a slice of
// the substrings between those expression matches.
//
// The slice returned by this method consists of all the substrings of s
// not contained in the slice returned by [Regexp.FindAllString]. When called on an expression
// that contains no metacharacters, it is equivalent to [strings.SplitN].
//
// Example:
//
//	s := regexp.MustCompile("a*").Split("abaabaccadaaae", 5)
//	// s: ["", "b", "b", "c", "cadaaae"]
//
// The count determines the number of substrings to return:
//   - n > 0: at most n substrings; the last substring will be the unsplit remainder;
//   - n == 0: the result is nil (zero substrings);
//   - n < 0: all substrings.
func (re *Regexp) Split(s string, n int) []string {
	if n == 0 {
		return nil
	}
	if len(re.expr) > 0 && len(s) == 0 {
		return []string{""}
	}

	matches := re.FindAllStringIndex(s, n)
	strings := make([]string, 0, len(matches))

	beg := 0
	end := 0
	for _, match := range matches {
		if n > 0 && len(strings) >= n-1 {
			break
		}

		end = match[0]
		if match[1] != 0 {
			strings = append(strings, s[beg:end])
		}
		beg = match[1]
	}

	if end != len(s) {
		strings = append(strings, s[beg:])
	}

	return strings
}

// AppendText implements [encoding.TextAppender]. The output
// matches that of calling the [Regexp.String] method.
//
// Note that the output is lossy in some cases: This method does not indicate
// POSIX regular expressions (i.e. those compiled by calling [CompilePOSIX]), or
// those for which the [Regexp.Longest] method has been called.
func (re *Regexp) AppendText(b []byte) ([]byte, error) {
	return append(b, re.String()...), nil
}

// MarshalText implements [encoding.TextMarshaler]. The output
// matches that of calling the [Regexp.AppendText] method.
//
// See [Regexp.AppendText] for more information.
func (re *Regexp) MarshalText() ([]byte, error) {
	return re.AppendText(nil)
}

// UnmarshalText implements [encoding.TextUnmarshaler] by calling
// [Compile] on the encoded value.
func (re *Regexp) UnmarshalText(text []byte) error {
	newRE, err := Compile(string(text))
	if err != nil {
		return err
	}
	*re = *newRE
	return nil
}
