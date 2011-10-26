// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code to parse a template.

package template

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"strconv"
	"strings"
	"unicode"
	"utf8"
)

// Errors returned during parsing and execution.  Users may extract the information and reformat
// if they desire.
type Error struct {
	Line int
	Msg  string
}

func (e *Error) String() string { return fmt.Sprintf("line %d: %s", e.Line, e.Msg) }

// checkError is a deferred function to turn a panic with type *Error into a plain error return.
// Other panics are unexpected and so are re-enabled.
func checkError(error *os.Error) {
	if v := recover(); v != nil {
		if e, ok := v.(*Error); ok {
			*error = e
		} else {
			// runtime errors should crash
			panic(v)
		}
	}
}

// Most of the literals are aces.
var lbrace = []byte{'{'}
var rbrace = []byte{'}'}
var space = []byte{' '}
var tab = []byte{'\t'}

// The various types of "tokens", which are plain text or (usually) brace-delimited descriptors
const (
	tokAlternates = iota
	tokComment
	tokEnd
	tokLiteral
	tokOr
	tokRepeated
	tokSection
	tokText
	tokVariable
)

// FormatterMap is the type describing the mapping from formatter
// names to the functions that implement them.
type FormatterMap map[string]func(io.Writer, string, ...interface{})

// Built-in formatters.
var builtins = FormatterMap{
	"html": HTMLFormatter,
	"str":  StringFormatter,
	"":     StringFormatter,
}

// The parsed state of a template is a vector of xxxElement structs.
// Sections have line numbers so errors can be reported better during execution.

// Plain text.
type textElement struct {
	text []byte
}

// A literal such as .meta-left or .meta-right
type literalElement struct {
	text []byte
}

// A variable invocation to be evaluated
type variableElement struct {
	linenum int
	args    []interface{} // The fields and literals in the invocation.
	fmts    []string      // Names of formatters to apply. len(fmts) > 0
}

// A variableElement arg to be evaluated as a field name
type fieldName string

// A .section block, possibly with a .or
type sectionElement struct {
	linenum int    // of .section itself
	field   string // cursor field for this block
	start   int    // first element
	or      int    // first element of .or block
	end     int    // one beyond last element
}

// A .repeated block, possibly with a .or and a .alternates
type repeatedElement struct {
	sectionElement     // It has the same structure...
	altstart       int // ... except for alternates
	altend         int
}

// Template is the type that represents a template definition.
// It is unchanged after parsing.
type Template struct {
	fmap FormatterMap // formatters for variables
	// Used during parsing:
	ldelim, rdelim []byte // delimiters; default {}
	buf            []byte // input text to process
	p              int    // position in buf
	linenum        int    // position in input
	// Parsed results:
	elems []interface{}
}

// New creates a new template with the specified formatter map (which
// may be nil) to define auxiliary functions for formatting variables.
func New(fmap FormatterMap) *Template {
	t := new(Template)
	t.fmap = fmap
	t.ldelim = lbrace
	t.rdelim = rbrace
	t.elems = make([]interface{}, 0, 16)
	return t
}

// Report error and stop executing.  The line number must be provided explicitly.
func (t *Template) execError(st *state, line int, err string, args ...interface{}) {
	panic(&Error{line, fmt.Sprintf(err, args...)})
}

// Report error, panic to terminate parsing.
// The line number comes from the template state.
func (t *Template) parseError(err string, args ...interface{}) {
	panic(&Error{t.linenum, fmt.Sprintf(err, args...)})
}

// Is this an exported - upper case - name?
func isExported(name string) bool {
	r, _ := utf8.DecodeRuneInString(name)
	return unicode.IsUpper(r)
}

// -- Lexical analysis

// Is c a space character?
func isSpace(c uint8) bool { return c == ' ' || c == '\t' || c == '\r' || c == '\n' }

// Safely, does s[n:n+len(t)] == t?
func equal(s []byte, n int, t []byte) bool {
	b := s[n:]
	if len(t) > len(b) { // not enough space left for a match.
		return false
	}
	for i, c := range t {
		if c != b[i] {
			return false
		}
	}
	return true
}

// isQuote returns true if c is a string- or character-delimiting quote character.
func isQuote(c byte) bool {
	return c == '"' || c == '`' || c == '\''
}

// endQuote returns the end quote index for the quoted string that
// starts at n, or -1 if no matching end quote is found before the end
// of the line.
func endQuote(s []byte, n int) int {
	quote := s[n]
	for n++; n < len(s); n++ {
		switch s[n] {
		case '\\':
			if quote == '"' || quote == '\'' {
				n++
			}
		case '\n':
			return -1
		case quote:
			return n
		}
	}
	return -1
}

// nextItem returns the next item from the input buffer.  If the returned
// item is empty, we are at EOF.  The item will be either a
// delimited string or a non-empty string between delimited
// strings. Tokens stop at (but include, if plain text) a newline.
// Action tokens on a line by themselves drop any space on
// either side, up to and including the newline.
func (t *Template) nextItem() []byte {
	startOfLine := t.p == 0 || t.buf[t.p-1] == '\n'
	start := t.p
	var i int
	newline := func() {
		t.linenum++
		i++
	}
	// Leading space up to but not including newline
	for i = start; i < len(t.buf); i++ {
		if t.buf[i] == '\n' || !isSpace(t.buf[i]) {
			break
		}
	}
	leadingSpace := i > start
	// What's left is nothing, newline, delimited string, or plain text
	switch {
	case i == len(t.buf):
		// EOF; nothing to do
	case t.buf[i] == '\n':
		newline()
	case equal(t.buf, i, t.ldelim):
		left := i         // Start of left delimiter.
		right := -1       // Will be (immediately after) right delimiter.
		haveText := false // Delimiters contain text.
		i += len(t.ldelim)
		// Find the end of the action.
		for ; i < len(t.buf); i++ {
			if t.buf[i] == '\n' {
				break
			}
			if isQuote(t.buf[i]) {
				i = endQuote(t.buf, i)
				if i == -1 {
					t.parseError("unmatched quote")
					return nil
				}
				continue
			}
			if equal(t.buf, i, t.rdelim) {
				i += len(t.rdelim)
				right = i
				break
			}
			haveText = true
		}
		if right < 0 {
			t.parseError("unmatched opening delimiter")
			return nil
		}
		// Is this a special action (starts with '.' or '#') and the only thing on the line?
		if startOfLine && haveText {
			firstChar := t.buf[left+len(t.ldelim)]
			if firstChar == '.' || firstChar == '#' {
				// It's special and the first thing on the line. Is it the last?
				for j := right; j < len(t.buf) && isSpace(t.buf[j]); j++ {
					if t.buf[j] == '\n' {
						// Yes it is. Drop the surrounding space and return the {.foo}
						t.linenum++
						t.p = j + 1
						return t.buf[left:right]
					}
				}
			}
		}
		// No it's not. If there's leading space, return that.
		if leadingSpace {
			// not trimming space: return leading space if there is some.
			t.p = left
			return t.buf[start:left]
		}
		// Return the word, leave the trailing space.
		start = left
		break
	default:
		for ; i < len(t.buf); i++ {
			if t.buf[i] == '\n' {
				newline()
				break
			}
			if equal(t.buf, i, t.ldelim) {
				break
			}
		}
	}
	item := t.buf[start:i]
	t.p = i
	return item
}

// Turn a byte array into a space-split array of strings,
// taking into account quoted strings.
func words(buf []byte) []string {
	s := make([]string, 0, 5)
	for i := 0; i < len(buf); {
		// One word per loop
		for i < len(buf) && isSpace(buf[i]) {
			i++
		}
		if i == len(buf) {
			break
		}
		// Got a word
		start := i
		if isQuote(buf[i]) {
			i = endQuote(buf, i)
			if i < 0 {
				i = len(buf)
			} else {
				i++
			}
		}
		// Even with quotes, break on space only.  This handles input
		// such as {""|} and catches quoting mistakes.
		for i < len(buf) && !isSpace(buf[i]) {
			i++
		}
		s = append(s, string(buf[start:i]))
	}
	return s
}

// Analyze an item and return its token type and, if it's an action item, an array of
// its constituent words.
func (t *Template) analyze(item []byte) (tok int, w []string) {
	// item is known to be non-empty
	if !equal(item, 0, t.ldelim) { // doesn't start with left delimiter
		tok = tokText
		return
	}
	if !equal(item, len(item)-len(t.rdelim), t.rdelim) { // doesn't end with right delimiter
		t.parseError("internal error: unmatched opening delimiter") // lexing should prevent this
		return
	}
	if len(item) <= len(t.ldelim)+len(t.rdelim) { // no contents
		t.parseError("empty directive")
		return
	}
	// Comment
	if item[len(t.ldelim)] == '#' {
		tok = tokComment
		return
	}
	// Split into words
	w = words(item[len(t.ldelim) : len(item)-len(t.rdelim)]) // drop final delimiter
	if len(w) == 0 {
		t.parseError("empty directive")
		return
	}
	first := w[0]
	if first[0] != '.' {
		tok = tokVariable
		return
	}
	if len(first) > 1 && first[1] >= '0' && first[1] <= '9' {
		// Must be a float.
		tok = tokVariable
		return
	}
	switch first {
	case ".meta-left", ".meta-right", ".space", ".tab":
		tok = tokLiteral
		return
	case ".or":
		tok = tokOr
		return
	case ".end":
		tok = tokEnd
		return
	case ".section":
		if len(w) != 2 {
			t.parseError("incorrect fields for .section: %s", item)
			return
		}
		tok = tokSection
		return
	case ".repeated":
		if len(w) != 3 || w[1] != "section" {
			t.parseError("incorrect fields for .repeated: %s", item)
			return
		}
		tok = tokRepeated
		return
	case ".alternates":
		if len(w) != 2 || w[1] != "with" {
			t.parseError("incorrect fields for .alternates: %s", item)
			return
		}
		tok = tokAlternates
		return
	}
	t.parseError("bad directive: %s", item)
	return
}

// formatter returns the Formatter with the given name in the Template, or nil if none exists.
func (t *Template) formatter(name string) func(io.Writer, string, ...interface{}) {
	if t.fmap != nil {
		if fn := t.fmap[name]; fn != nil {
			return fn
		}
	}
	return builtins[name]
}

// -- Parsing

// newVariable allocates a new variable-evaluation element.
func (t *Template) newVariable(words []string) *variableElement {
	formatters := extractFormatters(words)
	args := make([]interface{}, len(words))

	// Build argument list, processing any literals
	for i, word := range words {
		var lerr os.Error
		switch word[0] {
		case '"', '`', '\'':
			v, err := strconv.Unquote(word)
			if err == nil && word[0] == '\'' {
				args[i], _ = utf8.DecodeRuneInString(v)
			} else {
				args[i], lerr = v, err
			}

		case '.', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			v, err := strconv.Btoi64(word, 0)
			if err == nil {
				args[i] = v
			} else {
				v, err := strconv.Atof64(word)
				args[i], lerr = v, err
			}

		default:
			args[i] = fieldName(word)
		}
		if lerr != nil {
			t.parseError("invalid literal: %q: %s", word, lerr)
		}
	}

	// We could remember the function address here and avoid the lookup later,
	// but it's more dynamic to let the user change the map contents underfoot.
	// We do require the name to be present, though.

	// Is it in user-supplied map?
	for _, f := range formatters {
		if t.formatter(f) == nil {
			t.parseError("unknown formatter: %q", f)
		}
	}

	return &variableElement{t.linenum, args, formatters}
}

// extractFormatters extracts a list of formatters from words.
// After the final space-separated argument in a variable, formatters may be
// specified separated by pipe symbols. For example: {a b c|d|e}
// The words parameter still has the formatters joined by '|' in the last word.
// extractFormatters splits formatters, replaces the last word with the content
// found before the first '|' within it, and returns the formatters obtained.
// If no formatters are found in words, the default formatter is returned.
func extractFormatters(words []string) (formatters []string) {
	// "" is the default formatter.
	formatters = []string{""}
	if len(words) == 0 {
		return
	}
	var bar int
	lastWord := words[len(words)-1]
	if isQuote(lastWord[0]) {
		end := endQuote([]byte(lastWord), 0)
		if end < 0 || end+1 == len(lastWord) || lastWord[end+1] != '|' {
			return
		}
		bar = end + 1
	} else {
		bar = strings.IndexRune(lastWord, '|')
		if bar < 0 {
			return
		}
	}
	words[len(words)-1] = lastWord[0:bar]
	formatters = strings.Split(lastWord[bar+1:], "|")
	return
}

// Grab the next item.  If it's simple, just append it to the template.
// Otherwise return its details.
func (t *Template) parseSimple(item []byte) (done bool, tok int, w []string) {
	tok, w = t.analyze(item)
	done = true // assume for simplicity
	switch tok {
	case tokComment:
		return
	case tokText:
		t.elems = append(t.elems, &textElement{item})
		return
	case tokLiteral:
		switch w[0] {
		case ".meta-left":
			t.elems = append(t.elems, &literalElement{t.ldelim})
		case ".meta-right":
			t.elems = append(t.elems, &literalElement{t.rdelim})
		case ".space":
			t.elems = append(t.elems, &literalElement{space})
		case ".tab":
			t.elems = append(t.elems, &literalElement{tab})
		default:
			t.parseError("internal error: unknown literal: %s", w[0])
		}
		return
	case tokVariable:
		t.elems = append(t.elems, t.newVariable(w))
		return
	}
	return false, tok, w
}

// parseRepeated and parseSection are mutually recursive

func (t *Template) parseRepeated(words []string) *repeatedElement {
	r := new(repeatedElement)
	t.elems = append(t.elems, r)
	r.linenum = t.linenum
	r.field = words[2]
	// Scan section, collecting true and false (.or) blocks.
	r.start = len(t.elems)
	r.or = -1
	r.altstart = -1
	r.altend = -1
Loop:
	for {
		item := t.nextItem()
		if len(item) == 0 {
			t.parseError("missing .end for .repeated section")
			break
		}
		done, tok, w := t.parseSimple(item)
		if done {
			continue
		}
		switch tok {
		case tokEnd:
			break Loop
		case tokOr:
			if r.or >= 0 {
				t.parseError("extra .or in .repeated section")
				break Loop
			}
			r.altend = len(t.elems)
			r.or = len(t.elems)
		case tokSection:
			t.parseSection(w)
		case tokRepeated:
			t.parseRepeated(w)
		case tokAlternates:
			if r.altstart >= 0 {
				t.parseError("extra .alternates in .repeated section")
				break Loop
			}
			if r.or >= 0 {
				t.parseError(".alternates inside .or block in .repeated section")
				break Loop
			}
			r.altstart = len(t.elems)
		default:
			t.parseError("internal error: unknown repeated section item: %s", item)
			break Loop
		}
	}
	if r.altend < 0 {
		r.altend = len(t.elems)
	}
	r.end = len(t.elems)
	return r
}

func (t *Template) parseSection(words []string) *sectionElement {
	s := new(sectionElement)
	t.elems = append(t.elems, s)
	s.linenum = t.linenum
	s.field = words[1]
	// Scan section, collecting true and false (.or) blocks.
	s.start = len(t.elems)
	s.or = -1
Loop:
	for {
		item := t.nextItem()
		if len(item) == 0 {
			t.parseError("missing .end for .section")
			break
		}
		done, tok, w := t.parseSimple(item)
		if done {
			continue
		}
		switch tok {
		case tokEnd:
			break Loop
		case tokOr:
			if s.or >= 0 {
				t.parseError("extra .or in .section")
				break Loop
			}
			s.or = len(t.elems)
		case tokSection:
			t.parseSection(w)
		case tokRepeated:
			t.parseRepeated(w)
		case tokAlternates:
			t.parseError(".alternates not in .repeated")
		default:
			t.parseError("internal error: unknown section item: %s", item)
		}
	}
	s.end = len(t.elems)
	return s
}

func (t *Template) parse() {
	for {
		item := t.nextItem()
		if len(item) == 0 {
			break
		}
		done, tok, w := t.parseSimple(item)
		if done {
			continue
		}
		switch tok {
		case tokOr, tokEnd, tokAlternates:
			t.parseError("unexpected %s", w[0])
		case tokSection:
			t.parseSection(w)
		case tokRepeated:
			t.parseRepeated(w)
		default:
			t.parseError("internal error: bad directive in parse: %s", item)
		}
	}
}

// -- Execution

// -- Public interface

// Parse initializes a Template by parsing its definition.  The string
// s contains the template text.  If any errors occur, Parse returns
// the error.
func (t *Template) Parse(s string) (err os.Error) {
	if t.elems == nil {
		return &Error{1, "template not allocated with New"}
	}
	if !validDelim(t.ldelim) || !validDelim(t.rdelim) {
		return &Error{1, fmt.Sprintf("bad delimiter strings %q %q", t.ldelim, t.rdelim)}
	}
	defer checkError(&err)
	t.buf = []byte(s)
	t.p = 0
	t.linenum = 1
	t.parse()
	return nil
}

// ParseFile is like Parse but reads the template definition from the
// named file.
func (t *Template) ParseFile(filename string) (err os.Error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	return t.Parse(string(b))
}

// Execute applies a parsed template to the specified data object,
// generating output to wr.
func (t *Template) Execute(wr io.Writer, data interface{}) (err os.Error) {
	// Extract the driver data.
	val := reflect.ValueOf(data)
	defer checkError(&err)
	t.p = 0
	t.execute(0, len(t.elems), &state{parent: nil, data: val, wr: wr})
	return nil
}

// SetDelims sets the left and right delimiters for operations in the
// template.  They are validated during parsing.  They could be
// validated here but it's better to keep the routine simple.  The
// delimiters are very rarely invalid and Parse has the necessary
// error-handling interface already.
func (t *Template) SetDelims(left, right string) {
	t.ldelim = []byte(left)
	t.rdelim = []byte(right)
}

// Parse creates a Template with default parameters (such as {} for
// metacharacters).  The string s contains the template text while
// the formatter map fmap, which may be nil, defines auxiliary functions
// for formatting variables.  The template is returned. If any errors
// occur, err will be non-nil.
func Parse(s string, fmap FormatterMap) (t *Template, err os.Error) {
	t = New(fmap)
	err = t.Parse(s)
	if err != nil {
		t = nil
	}
	return
}

// ParseFile is a wrapper function that creates a Template with default
// parameters (such as {} for metacharacters).  The filename identifies
// a file containing the template text, while the formatter map fmap, which
// may be nil, defines auxiliary functions for formatting variables.
// The template is returned. If any errors occur, err will be non-nil.
func ParseFile(filename string, fmap FormatterMap) (t *Template, err os.Error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return Parse(string(b), fmap)
}

// MustParse is like Parse but panics if the template cannot be parsed.
func MustParse(s string, fmap FormatterMap) *Template {
	t, err := Parse(s, fmap)
	if err != nil {
		panic("template.MustParse error: " + err.String())
	}
	return t
}

// MustParseFile is like ParseFile but panics if the file cannot be read
// or the template cannot be parsed.
func MustParseFile(filename string, fmap FormatterMap) *Template {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		panic("template.MustParseFile error: " + err.String())
	}
	return MustParse(string(b), fmap)
}
