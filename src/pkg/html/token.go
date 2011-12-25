// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"io"
	"strconv"
	"strings"
)

// A TokenType is the type of a Token.
type TokenType int

const (
	// ErrorToken means that an error occurred during tokenization.
	ErrorToken TokenType = iota
	// TextToken means a text node.
	TextToken
	// A StartTagToken looks like <a>.
	StartTagToken
	// An EndTagToken looks like </a>.
	EndTagToken
	// A SelfClosingTagToken tag looks like <br/>.
	SelfClosingTagToken
	// A CommentToken looks like <!--x-->.
	CommentToken
	// A DoctypeToken looks like <!DOCTYPE x>
	DoctypeToken
)

// String returns a string representation of the TokenType.
func (t TokenType) String() string {
	switch t {
	case ErrorToken:
		return "Error"
	case TextToken:
		return "Text"
	case StartTagToken:
		return "StartTag"
	case EndTagToken:
		return "EndTag"
	case SelfClosingTagToken:
		return "SelfClosingTag"
	case CommentToken:
		return "Comment"
	case DoctypeToken:
		return "Doctype"
	}
	return "Invalid(" + strconv.Itoa(int(t)) + ")"
}

// An Attribute is an attribute namespace-key-value triple. Namespace is
// non-empty for foreign attributes like xlink, Key is alphabetic (and hence
// does not contain escapable characters like '&', '<' or '>'), and Val is
// unescaped (it looks like "a<b" rather than "a&lt;b").
//
// Namespace is only used by the parser, not the tokenizer.
type Attribute struct {
	Namespace, Key, Val string
}

// A Token consists of a TokenType and some Data (tag name for start and end
// tags, content for text, comments and doctypes). A tag Token may also contain
// a slice of Attributes. Data is unescaped for all Tokens (it looks like "a<b"
// rather than "a&lt;b").
type Token struct {
	Type TokenType
	Data string
	Attr []Attribute
}

// tagString returns a string representation of a tag Token's Data and Attr.
func (t Token) tagString() string {
	if len(t.Attr) == 0 {
		return t.Data
	}
	buf := bytes.NewBuffer(nil)
	buf.WriteString(t.Data)
	for _, a := range t.Attr {
		buf.WriteByte(' ')
		buf.WriteString(a.Key)
		buf.WriteString(`="`)
		escape(buf, a.Val)
		buf.WriteByte('"')
	}
	return buf.String()
}

// String returns a string representation of the Token.
func (t Token) String() string {
	switch t.Type {
	case ErrorToken:
		return ""
	case TextToken:
		return EscapeString(t.Data)
	case StartTagToken:
		return "<" + t.tagString() + ">"
	case EndTagToken:
		return "</" + t.tagString() + ">"
	case SelfClosingTagToken:
		return "<" + t.tagString() + "/>"
	case CommentToken:
		return "<!--" + t.Data + "-->"
	case DoctypeToken:
		return "<!DOCTYPE " + t.Data + ">"
	}
	return "Invalid(" + strconv.Itoa(int(t.Type)) + ")"
}

// span is a range of bytes in a Tokenizer's buffer. The start is inclusive,
// the end is exclusive.
type span struct {
	start, end int
}

// A Tokenizer returns a stream of HTML Tokens.
type Tokenizer struct {
	// r is the source of the HTML text.
	r io.Reader
	// tt is the TokenType of the current token.
	tt TokenType
	// err is the first error encountered during tokenization. It is possible
	// for tt != Error && err != nil to hold: this means that Next returned a
	// valid token but the subsequent Next call will return an error token.
	// For example, if the HTML text input was just "plain", then the first
	// Next call would set z.err to io.EOF but return a TextToken, and all
	// subsequent Next calls would return an ErrorToken.
	// err is never reset. Once it becomes non-nil, it stays non-nil.
	err error
	// buf[raw.start:raw.end] holds the raw bytes of the current token.
	// buf[raw.end:] is buffered input that will yield future tokens.
	raw span
	buf []byte
	// buf[data.start:data.end] holds the raw bytes of the current token's data:
	// a text token's text, a tag token's tag name, etc.
	data span
	// pendingAttr is the attribute key and value currently being tokenized.
	// When complete, pendingAttr is pushed onto attr. nAttrReturned is
	// incremented on each call to TagAttr.
	pendingAttr   [2]span
	attr          [][2]span
	nAttrReturned int
	// rawTag is the "script" in "</script>" that closes the next token. If
	// non-empty, the subsequent call to Next will return a raw or RCDATA text
	// token: one that treats "<p>" as text instead of an element.
	// rawTag's contents are lower-cased.
	rawTag string
	// textIsRaw is whether the current text token's data is not escaped.
	textIsRaw bool
}

// Err returns the error associated with the most recent ErrorToken token.
// This is typically io.EOF, meaning the end of tokenization.
func (z *Tokenizer) Err() error {
	if z.tt != ErrorToken {
		return nil
	}
	return z.err
}

// readByte returns the next byte from the input stream, doing a buffered read
// from z.r into z.buf if necessary. z.buf[z.raw.start:z.raw.end] remains a contiguous byte
// slice that holds all the bytes read so far for the current token.
// It sets z.err if the underlying reader returns an error.
// Pre-condition: z.err == nil.
func (z *Tokenizer) readByte() byte {
	if z.raw.end >= len(z.buf) {
		// Our buffer is exhausted and we have to read from z.r.
		// We copy z.buf[z.raw.start:z.raw.end] to the beginning of z.buf. If the length
		// z.raw.end - z.raw.start is more than half the capacity of z.buf, then we
		// allocate a new buffer before the copy.
		c := cap(z.buf)
		d := z.raw.end - z.raw.start
		var buf1 []byte
		if 2*d > c {
			buf1 = make([]byte, d, 2*c)
		} else {
			buf1 = z.buf[:d]
		}
		copy(buf1, z.buf[z.raw.start:z.raw.end])
		if x := z.raw.start; x != 0 {
			// Adjust the data/attr spans to refer to the same contents after the copy.
			z.data.start -= x
			z.data.end -= x
			z.pendingAttr[0].start -= x
			z.pendingAttr[0].end -= x
			z.pendingAttr[1].start -= x
			z.pendingAttr[1].end -= x
			for i := range z.attr {
				z.attr[i][0].start -= x
				z.attr[i][0].end -= x
				z.attr[i][1].start -= x
				z.attr[i][1].end -= x
			}
		}
		z.raw.start, z.raw.end, z.buf = 0, d, buf1[:d]
		// Now that we have copied the live bytes to the start of the buffer,
		// we read from z.r into the remainder.
		n, err := z.r.Read(buf1[d:cap(buf1)])
		if err != nil {
			z.err = err
			return 0
		}
		z.buf = buf1[:d+n]
	}
	x := z.buf[z.raw.end]
	z.raw.end++
	return x
}

// skipWhiteSpace skips past any white space.
func (z *Tokenizer) skipWhiteSpace() {
	if z.err != nil {
		return
	}
	for {
		c := z.readByte()
		if z.err != nil {
			return
		}
		switch c {
		case ' ', '\n', '\r', '\t', '\f':
			// No-op.
		default:
			z.raw.end--
			return
		}
	}
}

// readRawOrRCDATA reads until the next "</foo>", where "foo" is z.rawTag and
// is typically something like "script" or "textarea".
func (z *Tokenizer) readRawOrRCDATA() {
loop:
	for {
		c := z.readByte()
		if z.err != nil {
			break loop
		}
		if c != '<' {
			continue loop
		}
		c = z.readByte()
		if z.err != nil {
			break loop
		}
		if c != '/' {
			continue loop
		}
		for i := 0; i < len(z.rawTag); i++ {
			c = z.readByte()
			if z.err != nil {
				break loop
			}
			if c != z.rawTag[i] && c != z.rawTag[i]-('a'-'A') {
				continue loop
			}
		}
		c = z.readByte()
		if z.err != nil {
			break loop
		}
		switch c {
		case ' ', '\n', '\r', '\t', '\f', '/', '>':
			// The 3 is 2 for the leading "</" plus 1 for the trailing character c.
			z.raw.end -= 3 + len(z.rawTag)
			break loop
		case '<':
			// Step back one, to catch "</foo</foo>".
			z.raw.end--
		}
	}
	z.data.end = z.raw.end
	// A textarea's or title's RCDATA can contain escaped entities.
	z.textIsRaw = z.rawTag != "textarea" && z.rawTag != "title"
	z.rawTag = ""
}

// readComment reads the next comment token starting with "<!--". The opening
// "<!--" has already been consumed.
func (z *Tokenizer) readComment() {
	z.data.start = z.raw.end
	defer func() {
		if z.data.end < z.data.start {
			// It's a comment with no data, like <!-->.
			z.data.end = z.data.start
		}
	}()
	for dashCount := 2; ; {
		c := z.readByte()
		if z.err != nil {
			// Ignore up to two dashes at EOF.
			if dashCount > 2 {
				dashCount = 2
			}
			z.data.end = z.raw.end - dashCount
			return
		}
		switch c {
		case '-':
			dashCount++
			continue
		case '>':
			if dashCount >= 2 {
				z.data.end = z.raw.end - len("-->")
				return
			}
		case '!':
			if dashCount >= 2 {
				c = z.readByte()
				if z.err != nil {
					z.data.end = z.raw.end
					return
				}
				if c == '>' {
					z.data.end = z.raw.end - len("--!>")
					return
				}
			}
		}
		dashCount = 0
	}
}

// readUntilCloseAngle reads until the next ">".
func (z *Tokenizer) readUntilCloseAngle() {
	z.data.start = z.raw.end
	for {
		c := z.readByte()
		if z.err != nil {
			z.data.end = z.raw.end
			return
		}
		if c == '>' {
			z.data.end = z.raw.end - len(">")
			return
		}
	}
}

// readMarkupDeclaration reads the next token starting with "<!". It might be
// a "<!--comment-->", a "<!DOCTYPE foo>", or "<!a bogus comment". The opening
// "<!" has already been consumed.
func (z *Tokenizer) readMarkupDeclaration() TokenType {
	z.data.start = z.raw.end
	var c [2]byte
	for i := 0; i < 2; i++ {
		c[i] = z.readByte()
		if z.err != nil {
			z.data.end = z.raw.end
			return CommentToken
		}
	}
	if c[0] == '-' && c[1] == '-' {
		z.readComment()
		return CommentToken
	}
	z.raw.end -= 2
	const s = "DOCTYPE"
	for i := 0; i < len(s); i++ {
		c := z.readByte()
		if z.err != nil {
			z.data.end = z.raw.end
			return CommentToken
		}
		if c != s[i] && c != s[i]+('a'-'A') {
			// Back up to read the fragment of "DOCTYPE" again.
			z.raw.end = z.data.start
			z.readUntilCloseAngle()
			return CommentToken
		}
	}
	if z.skipWhiteSpace(); z.err != nil {
		z.data.start = z.raw.end
		z.data.end = z.raw.end
		return DoctypeToken
	}
	z.readUntilCloseAngle()
	return DoctypeToken
}

// startTagIn returns whether the start tag in z.buf[z.data.start:z.data.end]
// case-insensitively matches any element of ss.
func (z *Tokenizer) startTagIn(ss ...string) bool {
loop:
	for _, s := range ss {
		if z.data.end-z.data.start != len(s) {
			continue loop
		}
		for i := 0; i < len(s); i++ {
			c := z.buf[z.data.start+i]
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != s[i] {
				continue loop
			}
		}
		return true
	}
	return false
}

// readStartTag reads the next start tag token. The opening "<a" has already
// been consumed, where 'a' means anything in [A-Za-z].
func (z *Tokenizer) readStartTag() TokenType {
	z.attr = z.attr[:0]
	z.nAttrReturned = 0
	// Read the tag name and attribute key/value pairs.
	z.readTagName()
	if z.skipWhiteSpace(); z.err != nil {
		return ErrorToken
	}
	for {
		c := z.readByte()
		if z.err != nil || c == '>' {
			break
		}
		z.raw.end--
		z.readTagAttrKey()
		z.readTagAttrVal()
		// Save pendingAttr if it has a non-empty key.
		if z.pendingAttr[0].start != z.pendingAttr[0].end {
			z.attr = append(z.attr, z.pendingAttr)
		}
		if z.skipWhiteSpace(); z.err != nil {
			break
		}
	}
	// Several tags flag the tokenizer's next token as raw.
	c, raw := z.buf[z.data.start], false
	if 'A' <= c && c <= 'Z' {
		c += 'a' - 'A'
	}
	switch c {
	case 'i':
		raw = z.startTagIn("iframe")
	case 'n':
		raw = z.startTagIn("noembed", "noframes", "noscript")
	case 'p':
		raw = z.startTagIn("plaintext")
	case 's':
		raw = z.startTagIn("script", "style")
	case 't':
		raw = z.startTagIn("textarea", "title")
	case 'x':
		raw = z.startTagIn("xmp")
	}
	if raw {
		z.rawTag = strings.ToLower(string(z.buf[z.data.start:z.data.end]))
	}
	// Look for a self-closing token like "<br/>".
	if z.err == nil && z.buf[z.raw.end-2] == '/' {
		return SelfClosingTagToken
	}
	return StartTagToken
}

// readEndTag reads the next end tag token. The opening "</a" has already
// been consumed, where 'a' means anything in [A-Za-z].
func (z *Tokenizer) readEndTag() {
	z.attr = z.attr[:0]
	z.nAttrReturned = 0
	z.readTagName()
	for {
		c := z.readByte()
		if z.err != nil || c == '>' {
			return
		}
	}
}

// readTagName sets z.data to the "div" in "<div k=v>". The reader (z.raw.end)
// is positioned such that the first byte of the tag name (the "d" in "<div")
// has already been consumed.
func (z *Tokenizer) readTagName() {
	z.data.start = z.raw.end - 1
	for {
		c := z.readByte()
		if z.err != nil {
			z.data.end = z.raw.end
			return
		}
		switch c {
		case ' ', '\n', '\r', '\t', '\f':
			z.data.end = z.raw.end - 1
			return
		case '/', '>':
			z.raw.end--
			z.data.end = z.raw.end
			return
		}
	}
}

// readTagAttrKey sets z.pendingAttr[0] to the "k" in "<div k=v>".
// Precondition: z.err == nil.
func (z *Tokenizer) readTagAttrKey() {
	z.pendingAttr[0].start = z.raw.end
	for {
		c := z.readByte()
		if z.err != nil {
			z.pendingAttr[0].end = z.raw.end
			return
		}
		switch c {
		case ' ', '\n', '\r', '\t', '\f', '/':
			z.pendingAttr[0].end = z.raw.end - 1
			return
		case '=', '>':
			z.raw.end--
			z.pendingAttr[0].end = z.raw.end
			return
		}
	}
}

// readTagAttrVal sets z.pendingAttr[1] to the "v" in "<div k=v>".
func (z *Tokenizer) readTagAttrVal() {
	z.pendingAttr[1].start = z.raw.end
	z.pendingAttr[1].end = z.raw.end
	if z.skipWhiteSpace(); z.err != nil {
		return
	}
	c := z.readByte()
	if z.err != nil {
		return
	}
	if c != '=' {
		z.raw.end--
		return
	}
	if z.skipWhiteSpace(); z.err != nil {
		return
	}
	quote := z.readByte()
	if z.err != nil {
		return
	}
	switch quote {
	case '>':
		z.raw.end--
		return

	case '\'', '"':
		z.pendingAttr[1].start = z.raw.end
		for {
			c := z.readByte()
			if z.err != nil {
				z.pendingAttr[1].end = z.raw.end
				return
			}
			if c == quote {
				z.pendingAttr[1].end = z.raw.end - 1
				return
			}
		}

	default:
		z.pendingAttr[1].start = z.raw.end - 1
		for {
			c := z.readByte()
			if z.err != nil {
				z.pendingAttr[1].end = z.raw.end
				return
			}
			switch c {
			case ' ', '\n', '\r', '\t', '\f':
				z.pendingAttr[1].end = z.raw.end - 1
				return
			case '>':
				z.raw.end--
				z.pendingAttr[1].end = z.raw.end
				return
			}
		}
	}
}

// Next scans the next token and returns its type.
func (z *Tokenizer) Next() TokenType {
	if z.err != nil {
		z.tt = ErrorToken
		return z.tt
	}
	z.raw.start = z.raw.end
	z.data.start = z.raw.end
	z.data.end = z.raw.end
	if z.rawTag != "" {
		if z.rawTag == "plaintext" {
			// Read everything up to EOF.
			for z.err == nil {
				z.readByte()
			}
			z.textIsRaw = true
		} else {
			z.readRawOrRCDATA()
		}
		if z.data.end > z.data.start {
			z.tt = TextToken
			return z.tt
		}
	}
	z.textIsRaw = false

loop:
	for {
		c := z.readByte()
		if z.err != nil {
			break loop
		}
		if c != '<' {
			continue loop
		}

		// Check if the '<' we have just read is part of a tag, comment
		// or doctype. If not, it's part of the accumulated text token.
		c = z.readByte()
		if z.err != nil {
			break loop
		}
		var tokenType TokenType
		switch {
		case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
			tokenType = StartTagToken
		case c == '/':
			tokenType = EndTagToken
		case c == '!' || c == '?':
			// We use CommentToken to mean any of "<!--actual comments-->",
			// "<!DOCTYPE declarations>" and "<?xml processing instructions?>".
			tokenType = CommentToken
		default:
			continue
		}

		// We have a non-text token, but we might have accumulated some text
		// before that. If so, we return the text first, and return the non-
		// text token on the subsequent call to Next.
		if x := z.raw.end - len("<a"); z.raw.start < x {
			z.raw.end = x
			z.data.end = x
			z.tt = TextToken
			return z.tt
		}
		switch tokenType {
		case StartTagToken:
			z.tt = z.readStartTag()
			return z.tt
		case EndTagToken:
			c = z.readByte()
			if z.err != nil {
				break loop
			}
			if c == '>' {
				// "</>" does not generate a token at all.
				// Reset the tokenizer state and start again.
				z.raw.start = z.raw.end
				z.data.start = z.raw.end
				z.data.end = z.raw.end
				continue loop
			}
			if 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' {
				z.readEndTag()
				z.tt = EndTagToken
				return z.tt
			}
			z.raw.end--
			z.readUntilCloseAngle()
			z.tt = CommentToken
			return z.tt
		case CommentToken:
			if c == '!' {
				z.tt = z.readMarkupDeclaration()
				return z.tt
			}
			z.raw.end--
			z.readUntilCloseAngle()
			z.tt = CommentToken
			return z.tt
		}
	}
	if z.raw.start < z.raw.end {
		z.data.end = z.raw.end
		z.tt = TextToken
		return z.tt
	}
	z.tt = ErrorToken
	return z.tt
}

// Raw returns the unmodified text of the current token. Calling Next, Token,
// Text, TagName or TagAttr may change the contents of the returned slice.
func (z *Tokenizer) Raw() []byte {
	return z.buf[z.raw.start:z.raw.end]
}

// Text returns the unescaped text of a text, comment or doctype token. The
// contents of the returned slice may change on the next call to Next.
func (z *Tokenizer) Text() []byte {
	switch z.tt {
	case TextToken, CommentToken, DoctypeToken:
		s := z.buf[z.data.start:z.data.end]
		z.data.start = z.raw.end
		z.data.end = z.raw.end
		if !z.textIsRaw {
			s = unescape(s)
		}
		return s
	}
	return nil
}

// TagName returns the lower-cased name of a tag token (the `img` out of
// `<IMG SRC="foo">`) and whether the tag has attributes.
// The contents of the returned slice may change on the next call to Next.
func (z *Tokenizer) TagName() (name []byte, hasAttr bool) {
	if z.data.start < z.data.end {
		switch z.tt {
		case StartTagToken, EndTagToken, SelfClosingTagToken:
			s := z.buf[z.data.start:z.data.end]
			z.data.start = z.raw.end
			z.data.end = z.raw.end
			return lower(s), z.nAttrReturned < len(z.attr)
		}
	}
	return nil, false
}

// TagAttr returns the lower-cased key and unescaped value of the next unparsed
// attribute for the current tag token and whether there are more attributes.
// The contents of the returned slices may change on the next call to Next.
func (z *Tokenizer) TagAttr() (key, val []byte, moreAttr bool) {
	if z.nAttrReturned < len(z.attr) {
		switch z.tt {
		case StartTagToken, SelfClosingTagToken:
			x := z.attr[z.nAttrReturned]
			z.nAttrReturned++
			key = z.buf[x[0].start:x[0].end]
			val = z.buf[x[1].start:x[1].end]
			return lower(key), unescape(val), z.nAttrReturned < len(z.attr)
		}
	}
	return nil, nil, false
}

// Token returns the next Token. The result's Data and Attr values remain valid
// after subsequent Next calls.
func (z *Tokenizer) Token() Token {
	t := Token{Type: z.tt}
	switch z.tt {
	case TextToken, CommentToken, DoctypeToken:
		t.Data = string(z.Text())
	case StartTagToken, SelfClosingTagToken:
		var attr []Attribute
		name, moreAttr := z.TagName()
		for moreAttr {
			var key, val []byte
			key, val, moreAttr = z.TagAttr()
			attr = append(attr, Attribute{"", string(key), string(val)})
		}
		t.Data = string(name)
		t.Attr = attr
	case EndTagToken:
		name, _ := z.TagName()
		t.Data = string(name)
	}
	return t
}

// NewTokenizer returns a new HTML Tokenizer for the given Reader.
// The input is assumed to be UTF-8 encoded.
func NewTokenizer(r io.Reader) *Tokenizer {
	return &Tokenizer{
		r:   r,
		buf: make([]byte, 0, 4096),
	}
}
