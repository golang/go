// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package regexp implements regular expression search.
//
// The syntax of the regular expressions accepted is the same
// general syntax used by Perl, Python, and other languages.
// More precisely, it is the syntax accepted by RE2 and described at
// http://code.google.com/p/re2/wiki/Syntax, except for \C.
//
// All characters are UTF-8-encoded code points.
//
// There are 16 methods of Regexp that match a regular expression and identify
// the matched text.  Their names are matched by this regular expression:
//
//	Find(All)?(String)?(Submatch)?(Index)?
//
// If 'All' is present, the routine matches successive non-overlapping
// matches of the entire expression.  Empty matches abutting a preceding
// match are ignored.  The return value is a slice containing the successive
// return values of the corresponding non-'All' routine.  These routines take
// an extra integer argument, n; if n >= 0, the function returns at most n
// matches/submatches.
//
// If 'String' is present, the argument is a string; otherwise it is a slice
// of bytes; return values are adjusted as appropriate.
//
// If 'Submatch' is present, the return value is a slice identifying the
// successive submatches of the expression.  Submatches are matches of
// parenthesized subexpressions within the regular expression, numbered from
// left to right in order of opening parenthesis.  Submatch 0 is the match of
// the entire expression, submatch 1 the match of the first parenthesized
// subexpression, and so on.
//
// If 'Index' is present, matches and submatches are identified by byte index
// pairs within the input string: result[2*n:2*n+1] identifies the indexes of
// the nth submatch.  The pair for n==0 identifies the match of the entire
// expression.  If 'Index' is not present, the match is identified by the
// text of the match/submatch.  If an index is negative, it means that
// subexpression did not match any string in the input.
//
// There is also a subset of the methods that can be applied to text read
// from a RuneReader:
//
//	MatchReader, FindReaderIndex, FindReaderSubmatchIndex
//
// This set may grow.  Note that regular expression matches may need to
// examine text beyond the text returned by a match, so the methods that
// match text from a RuneReader may read arbitrarily far into the input
// before returning.
//
// (There are a few other methods that do not match this pattern.)
//
package regexp

import (
	"bytes"
	"io"
	"regexp/syntax"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"
)

var debug = false

// Error is the local type for a parsing error.
type Error string

func (e Error) Error() string {
	return string(e)
}

// Regexp is the representation of a compiled regular expression.
// The public interface is entirely through methods.
// A Regexp is safe for concurrent use by multiple goroutines.
type Regexp struct {
	// read-only after Compile
	expr           string         // as passed to Compile
	prog           *syntax.Prog   // compiled program
	prefix         string         // required prefix in unanchored matches
	prefixBytes    []byte         // prefix, as a []byte
	prefixComplete bool           // prefix is the entire regexp
	prefixRune     rune           // first rune in prefix
	cond           syntax.EmptyOp // empty-width conditions required at start of match
	numSubexp      int
	subexpNames    []string
	longest        bool

	// cache of machines for running regexp
	mu      sync.Mutex
	machine []*machine
}

// String returns the source text used to compile the regular expression.
func (re *Regexp) String() string {
	return re.expr
}

// Compile parses a regular expression and returns, if successful,
// a Regexp object that can be used to match against text.
//
// When matching against text, the regexp returns a match that
// begins as early as possible in the input (leftmost), and among those
// it chooses the one that a backtracking search would have found first.
// This so-called leftmost-first matching is the same semantics
// that Perl, Python, and other implementations use, although this
// package implements it without the expense of backtracking.
// For POSIX leftmost-longest matching, see CompilePOSIX.
func Compile(expr string) (*Regexp, error) {
	return compile(expr, syntax.Perl, false)
}

// CompilePOSIX is like Compile but restricts the regular expression
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
// See http://swtch.com/~rsc/regexp/regexp2.html#posix for details.
func CompilePOSIX(expr string) (*Regexp, error) {
	return compile(expr, syntax.POSIX, true)
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
	regexp := &Regexp{
		expr:        expr,
		prog:        prog,
		numSubexp:   maxCap,
		subexpNames: capNames,
		cond:        prog.StartCond(),
		longest:     longest,
	}
	regexp.prefix, regexp.prefixComplete = prog.Prefix()
	if regexp.prefix != "" {
		// TODO(rsc): Remove this allocation by adding
		// IndexString to package bytes.
		regexp.prefixBytes = []byte(regexp.prefix)
		regexp.prefixRune, _ = utf8.DecodeRuneInString(regexp.prefix)
	}
	return regexp, nil
}

// get returns a machine to use for matching re.
// It uses the re's machine cache if possible, to avoid
// unnecessary allocation.
func (re *Regexp) get() *machine {
	re.mu.Lock()
	if n := len(re.machine); n > 0 {
		z := re.machine[n-1]
		re.machine = re.machine[:n-1]
		re.mu.Unlock()
		return z
	}
	re.mu.Unlock()
	z := progMachine(re.prog)
	z.re = re
	return z
}

// put returns a machine to the re's machine cache.
// There is no attempt to limit the size of the cache, so it will
// grow to the maximum number of simultaneous matches
// run using re.  (The cache empties when re gets garbage collected.)
func (re *Regexp) put(z *machine) {
	re.mu.Lock()
	re.machine = append(re.machine, z)
	re.mu.Unlock()
}

// MustCompile is like Compile but panics if the expression cannot be parsed.
// It simplifies safe initialization of global variables holding compiled regular
// expressions.
func MustCompile(str string) *Regexp {
	regexp, error := Compile(str)
	if error != nil {
		panic(`regexp: Compile(` + quote(str) + `): ` + error.Error())
	}
	return regexp
}

// MustCompilePOSIX is like CompilePOSIX but panics if the expression cannot be parsed.
// It simplifies safe initialization of global variables holding compiled regular
// expressions.
func MustCompilePOSIX(str string) *Regexp {
	regexp, error := CompilePOSIX(str)
	if error != nil {
		panic(`regexp: CompilePOSIX(` + quote(str) + `): ` + error.Error())
	}
	return regexp
}

func quote(s string) string {
	if strconv.CanBackquote(s) {
		return "`" + s + "`"
	}
	return strconv.Quote(s)
}

// NumSubexp returns the number of parenthesized subexpressions in this Regexp.
func (re *Regexp) NumSubexp() int {
	return re.numSubexp
}

// SubexpNames returns the names of the parenthesized subexpressions
// in this Regexp.  The name for the first sub-expression is names[1],
// so that if m is a match slice, the name for m[i] is SubexpNames()[i].
// Since the Regexp as a whole cannot be named, names[0] is always
// the empty string.  The slice should not be modified.
func (re *Regexp) SubexpNames() []string {
	return re.subexpNames
}

const endOfText rune = -1

// input abstracts different representations of the input text. It provides
// one-character lookahead.
type input interface {
	step(pos int) (r rune, width int) // advance one rune
	canCheckPrefix() bool             // can we look ahead without losing info?
	hasPrefix(re *Regexp) bool
	index(re *Regexp, pos int) int
	context(pos int) syntax.EmptyOp
}

// inputString scans a string.
type inputString struct {
	str string
}

func (i *inputString) step(pos int) (rune, int) {
	if pos < len(i.str) {
		c := i.str[pos]
		if c < utf8.RuneSelf {
			return rune(c), 1
		}
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

func (i *inputString) context(pos int) syntax.EmptyOp {
	r1, r2 := endOfText, endOfText
	if pos > 0 && pos <= len(i.str) {
		r1, _ = utf8.DecodeLastRuneInString(i.str[:pos])
	}
	if pos < len(i.str) {
		r2, _ = utf8.DecodeRuneInString(i.str[pos:])
	}
	return syntax.EmptyOpContext(r1, r2)
}

// inputBytes scans a byte slice.
type inputBytes struct {
	str []byte
}

func (i *inputBytes) step(pos int) (rune, int) {
	if pos < len(i.str) {
		c := i.str[pos]
		if c < utf8.RuneSelf {
			return rune(c), 1
		}
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

func (i *inputBytes) context(pos int) syntax.EmptyOp {
	r1, r2 := endOfText, endOfText
	if pos > 0 && pos <= len(i.str) {
		r1, _ = utf8.DecodeLastRune(i.str[:pos])
	}
	if pos < len(i.str) {
		r2, _ = utf8.DecodeRune(i.str[pos:])
	}
	return syntax.EmptyOpContext(r1, r2)
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

func (i *inputReader) context(pos int) syntax.EmptyOp {
	return 0
}

// LiteralPrefix returns a literal string that must begin any match
// of the regular expression re.  It returns the boolean true if the
// literal string comprises the entire regular expression.
func (re *Regexp) LiteralPrefix() (prefix string, complete bool) {
	return re.prefix, re.prefixComplete
}

// MatchReader returns whether the Regexp matches the text read by the
// RuneReader.  The return value is a boolean: true for match, false for no
// match.
func (re *Regexp) MatchReader(r io.RuneReader) bool {
	return re.doExecute(r, nil, "", 0, 0) != nil
}

// MatchString returns whether the Regexp matches the string s.
// The return value is a boolean: true for match, false for no match.
func (re *Regexp) MatchString(s string) bool {
	return re.doExecute(nil, nil, s, 0, 0) != nil
}

// Match returns whether the Regexp matches the byte slice b.
// The return value is a boolean: true for match, false for no match.
func (re *Regexp) Match(b []byte) bool {
	return re.doExecute(nil, b, "", 0, 0) != nil
}

// MatchReader checks whether a textual regular expression matches the text
// read by the RuneReader.  More complicated queries need to use Compile and
// the full Regexp interface.
func MatchReader(pattern string, r io.RuneReader) (matched bool, error error) {
	re, err := Compile(pattern)
	if err != nil {
		return false, err
	}
	return re.MatchReader(r), nil
}

// MatchString checks whether a textual regular expression
// matches a string.  More complicated queries need
// to use Compile and the full Regexp interface.
func MatchString(pattern string, s string) (matched bool, error error) {
	re, err := Compile(pattern)
	if err != nil {
		return false, err
	}
	return re.MatchString(s), nil
}

// Match checks whether a textual regular expression
// matches a byte slice.  More complicated queries need
// to use Compile and the full Regexp interface.
func Match(pattern string, b []byte) (matched bool, error error) {
	re, err := Compile(pattern)
	if err != nil {
		return false, err
	}
	return re.Match(b), nil
}

// ReplaceAllString returns a copy of src in which all matches for the Regexp
// have been replaced by repl.  No support is provided for expressions
// (e.g. \1 or $1) in the replacement string.
func (re *Regexp) ReplaceAllString(src, repl string) string {
	return re.ReplaceAllStringFunc(src, func(string) string { return repl })
}

// ReplaceAllStringFunc returns a copy of src in which all matches for the
// Regexp have been replaced by the return value of of function repl (whose
// first argument is the matched string).  No support is provided for
// expressions (e.g. \1 or $1) in the replacement string.
func (re *Regexp) ReplaceAllStringFunc(src string, repl func(string) string) string {
	lastMatchEnd := 0 // end position of the most recent match
	searchPos := 0    // position where we next look for a match
	buf := new(bytes.Buffer)
	for searchPos <= len(src) {
		a := re.doExecute(nil, nil, src, searchPos, 2)
		if len(a) == 0 {
			break // no more matches
		}

		// Copy the unmatched characters before this match.
		io.WriteString(buf, src[lastMatchEnd:a[0]])

		// Now insert a copy of the replacement string, but not for a
		// match of the empty string immediately after another match.
		// (Otherwise, we get double replacement for patterns that
		// match both empty and nonempty strings.)
		if a[1] > lastMatchEnd || a[0] == 0 {
			io.WriteString(buf, repl(src[a[0]:a[1]]))
		}
		lastMatchEnd = a[1]

		// Advance past this match; always advance at least one character.
		_, width := utf8.DecodeRuneInString(src[searchPos:])
		if searchPos+width > a[1] {
			searchPos += width
		} else if searchPos+1 > a[1] {
			// This clause is only needed at the end of the input
			// string.  In that case, DecodeRuneInString returns width=0.
			searchPos++
		} else {
			searchPos = a[1]
		}
	}

	// Copy the unmatched characters after the last match.
	io.WriteString(buf, src[lastMatchEnd:])

	return buf.String()
}

// ReplaceAll returns a copy of src in which all matches for the Regexp
// have been replaced by repl.  No support is provided for expressions
// (e.g. \1 or $1) in the replacement text.
func (re *Regexp) ReplaceAll(src, repl []byte) []byte {
	return re.ReplaceAllFunc(src, func([]byte) []byte { return repl })
}

// ReplaceAllFunc returns a copy of src in which all matches for the
// Regexp have been replaced by the return value of of function repl (whose
// first argument is the matched []byte).  No support is provided for
// expressions (e.g. \1 or $1) in the replacement string.
func (re *Regexp) ReplaceAllFunc(src []byte, repl func([]byte) []byte) []byte {
	lastMatchEnd := 0 // end position of the most recent match
	searchPos := 0    // position where we next look for a match
	buf := new(bytes.Buffer)
	for searchPos <= len(src) {
		a := re.doExecute(nil, src, "", searchPos, 2)
		if len(a) == 0 {
			break // no more matches
		}

		// Copy the unmatched characters before this match.
		buf.Write(src[lastMatchEnd:a[0]])

		// Now insert a copy of the replacement string, but not for a
		// match of the empty string immediately after another match.
		// (Otherwise, we get double replacement for patterns that
		// match both empty and nonempty strings.)
		if a[1] > lastMatchEnd || a[0] == 0 {
			buf.Write(repl(src[a[0]:a[1]]))
		}
		lastMatchEnd = a[1]

		// Advance past this match; always advance at least one character.
		_, width := utf8.DecodeRune(src[searchPos:])
		if searchPos+width > a[1] {
			searchPos += width
		} else if searchPos+1 > a[1] {
			// This clause is only needed at the end of the input
			// string.  In that case, DecodeRuneInString returns width=0.
			searchPos++
		} else {
			searchPos = a[1]
		}
	}

	// Copy the unmatched characters after the last match.
	buf.Write(src[lastMatchEnd:])

	return buf.Bytes()
}

var specialBytes = []byte(`\.+*?()|[]{}^$`)

func special(b byte) bool {
	return bytes.IndexByte(specialBytes, b) >= 0
}

// QuoteMeta returns a string that quotes all regular expression metacharacters
// inside the argument text; the returned string is a regular expression matching
// the literal text.  For example, QuoteMeta(`[foo]`) returns `\[foo\]`.
func QuoteMeta(s string) string {
	b := make([]byte, 2*len(s))

	// A byte loop is correct because all metacharacters are ASCII.
	j := 0
	for i := 0; i < len(s); i++ {
		if special(s[i]) {
			b[j] = '\\'
			j++
		}
		b[j] = s[i]
		j++
	}
	return string(b[0:j])
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

// Find matches in slice b if b is non-nil, otherwise find matches in string s.
func (re *Regexp) allMatches(s string, b []byte, n int, deliver func([]int)) {
	var end int
	if b == nil {
		end = len(s)
	} else {
		end = len(b)
	}

	for pos, i, prevMatchEnd := 0, 0, -1; i < n && pos <= end; {
		matches := re.doExecute(nil, b, s, pos, re.prog.NumCap)
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
			// TODO: use step()
			if b == nil {
				_, width = utf8.DecodeRuneInString(s[pos:end])
			} else {
				_, width = utf8.DecodeRune(b[pos:end])
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
			deliver(re.pad(matches))
			i++
		}
	}
}

// Find returns a slice holding the text of the leftmost match in b of the regular expression.
// A return value of nil indicates no match.
func (re *Regexp) Find(b []byte) []byte {
	a := re.doExecute(nil, b, "", 0, 2)
	if a == nil {
		return nil
	}
	return b[a[0]:a[1]]
}

// FindIndex returns a two-element slice of integers defining the location of
// the leftmost match in b of the regular expression.  The match itself is at
// b[loc[0]:loc[1]].
// A return value of nil indicates no match.
func (re *Regexp) FindIndex(b []byte) (loc []int) {
	a := re.doExecute(nil, b, "", 0, 2)
	if a == nil {
		return nil
	}
	return a[0:2]
}

// FindString returns a string holding the text of the leftmost match in s of the regular
// expression.  If there is no match, the return value is an empty string,
// but it will also be empty if the regular expression successfully matches
// an empty string.  Use FindStringIndex or FindStringSubmatch if it is
// necessary to distinguish these cases.
func (re *Regexp) FindString(s string) string {
	a := re.doExecute(nil, nil, s, 0, 2)
	if a == nil {
		return ""
	}
	return s[a[0]:a[1]]
}

// FindStringIndex returns a two-element slice of integers defining the
// location of the leftmost match in s of the regular expression.  The match
// itself is at s[loc[0]:loc[1]].
// A return value of nil indicates no match.
func (re *Regexp) FindStringIndex(s string) []int {
	a := re.doExecute(nil, nil, s, 0, 2)
	if a == nil {
		return nil
	}
	return a[0:2]
}

// FindReaderIndex returns a two-element slice of integers defining the
// location of the leftmost match of the regular expression in text read from
// the RuneReader.  The match itself is at s[loc[0]:loc[1]].  A return
// value of nil indicates no match.
func (re *Regexp) FindReaderIndex(r io.RuneReader) []int {
	a := re.doExecute(r, nil, "", 0, 2)
	if a == nil {
		return nil
	}
	return a[0:2]
}

// FindSubmatch returns a slice of slices holding the text of the leftmost
// match of the regular expression in b and the matches, if any, of its
// subexpressions, as defined by the 'Submatch' descriptions in the package
// comment.
// A return value of nil indicates no match.
func (re *Regexp) FindSubmatch(b []byte) [][]byte {
	a := re.doExecute(nil, b, "", 0, re.prog.NumCap)
	if a == nil {
		return nil
	}
	ret := make([][]byte, 1+re.numSubexp)
	for i := range ret {
		if 2*i < len(a) && a[2*i] >= 0 {
			ret[i] = b[a[2*i]:a[2*i+1]]
		}
	}
	return ret
}

// FindSubmatchIndex returns a slice holding the index pairs identifying the
// leftmost match of the regular expression in b and the matches, if any, of
// its subexpressions, as defined by the 'Submatch' and 'Index' descriptions
// in the package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindSubmatchIndex(b []byte) []int {
	return re.pad(re.doExecute(nil, b, "", 0, re.prog.NumCap))
}

// FindStringSubmatch returns a slice of strings holding the text of the
// leftmost match of the regular expression in s and the matches, if any, of
// its subexpressions, as defined by the 'Submatch' description in the
// package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindStringSubmatch(s string) []string {
	a := re.doExecute(nil, nil, s, 0, re.prog.NumCap)
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

// FindStringSubmatchIndex returns a slice holding the index pairs
// identifying the leftmost match of the regular expression in s and the
// matches, if any, of its subexpressions, as defined by the 'Submatch' and
// 'Index' descriptions in the package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindStringSubmatchIndex(s string) []int {
	return re.pad(re.doExecute(nil, nil, s, 0, re.prog.NumCap))
}

// FindReaderSubmatchIndex returns a slice holding the index pairs
// identifying the leftmost match of the regular expression of text read by
// the RuneReader, and the matches, if any, of its subexpressions, as defined
// by the 'Submatch' and 'Index' descriptions in the package comment.  A
// return value of nil indicates no match.
func (re *Regexp) FindReaderSubmatchIndex(r io.RuneReader) []int {
	return re.pad(re.doExecute(r, nil, "", 0, re.prog.NumCap))
}

const startSize = 10 // The size at which to start a slice in the 'All' routines.

// FindAll is the 'All' version of Find; it returns a slice of all successive
// matches of the expression, as defined by the 'All' description in the
// package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindAll(b []byte, n int) [][]byte {
	if n < 0 {
		n = len(b) + 1
	}
	result := make([][]byte, 0, startSize)
	re.allMatches("", b, n, func(match []int) {
		result = append(result, b[match[0]:match[1]])
	})
	if len(result) == 0 {
		return nil
	}
	return result
}

// FindAllIndex is the 'All' version of FindIndex; it returns a slice of all
// successive matches of the expression, as defined by the 'All' description
// in the package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindAllIndex(b []byte, n int) [][]int {
	if n < 0 {
		n = len(b) + 1
	}
	result := make([][]int, 0, startSize)
	re.allMatches("", b, n, func(match []int) {
		result = append(result, match[0:2])
	})
	if len(result) == 0 {
		return nil
	}
	return result
}

// FindAllString is the 'All' version of FindString; it returns a slice of all
// successive matches of the expression, as defined by the 'All' description
// in the package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindAllString(s string, n int) []string {
	if n < 0 {
		n = len(s) + 1
	}
	result := make([]string, 0, startSize)
	re.allMatches(s, nil, n, func(match []int) {
		result = append(result, s[match[0]:match[1]])
	})
	if len(result) == 0 {
		return nil
	}
	return result
}

// FindAllStringIndex is the 'All' version of FindStringIndex; it returns a
// slice of all successive matches of the expression, as defined by the 'All'
// description in the package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindAllStringIndex(s string, n int) [][]int {
	if n < 0 {
		n = len(s) + 1
	}
	result := make([][]int, 0, startSize)
	re.allMatches(s, nil, n, func(match []int) {
		result = append(result, match[0:2])
	})
	if len(result) == 0 {
		return nil
	}
	return result
}

// FindAllSubmatch is the 'All' version of FindSubmatch; it returns a slice
// of all successive matches of the expression, as defined by the 'All'
// description in the package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindAllSubmatch(b []byte, n int) [][][]byte {
	if n < 0 {
		n = len(b) + 1
	}
	result := make([][][]byte, 0, startSize)
	re.allMatches("", b, n, func(match []int) {
		slice := make([][]byte, len(match)/2)
		for j := range slice {
			if match[2*j] >= 0 {
				slice[j] = b[match[2*j]:match[2*j+1]]
			}
		}
		result = append(result, slice)
	})
	if len(result) == 0 {
		return nil
	}
	return result
}

// FindAllSubmatchIndex is the 'All' version of FindSubmatchIndex; it returns
// a slice of all successive matches of the expression, as defined by the
// 'All' description in the package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindAllSubmatchIndex(b []byte, n int) [][]int {
	if n < 0 {
		n = len(b) + 1
	}
	result := make([][]int, 0, startSize)
	re.allMatches("", b, n, func(match []int) {
		result = append(result, match)
	})
	if len(result) == 0 {
		return nil
	}
	return result
}

// FindAllStringSubmatch is the 'All' version of FindStringSubmatch; it
// returns a slice of all successive matches of the expression, as defined by
// the 'All' description in the package comment.
// A return value of nil indicates no match.
func (re *Regexp) FindAllStringSubmatch(s string, n int) [][]string {
	if n < 0 {
		n = len(s) + 1
	}
	result := make([][]string, 0, startSize)
	re.allMatches(s, nil, n, func(match []int) {
		slice := make([]string, len(match)/2)
		for j := range slice {
			if match[2*j] >= 0 {
				slice[j] = s[match[2*j]:match[2*j+1]]
			}
		}
		result = append(result, slice)
	})
	if len(result) == 0 {
		return nil
	}
	return result
}

// FindAllStringSubmatchIndex is the 'All' version of
// FindStringSubmatchIndex; it returns a slice of all successive matches of
// the expression, as defined by the 'All' description in the package
// comment.
// A return value of nil indicates no match.
func (re *Regexp) FindAllStringSubmatchIndex(s string, n int) [][]int {
	if n < 0 {
		n = len(s) + 1
	}
	result := make([][]int, 0, startSize)
	re.allMatches(s, nil, n, func(match []int) {
		result = append(result, match)
	})
	if len(result) == 0 {
		return nil
	}
	return result
}
