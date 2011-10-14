// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"io"
	"os"
	"strconv"
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

// An Attribute is an attribute key-value pair. Key is alphabetic (and hence
// does not contain escapable characters like '&', '<' or '>'), and Val is
// unescaped (it looks like "a<b" rather than "a&lt;b").
type Attribute struct {
	Key, Val string
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
		return "<!--" + EscapeString(t.Data) + "-->"
	case DoctypeToken:
		return "<!DOCTYPE " + EscapeString(t.Data) + ">"
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
	// If ReturnComments is set, Next returns comment tokens;
	// otherwise it skips over comments (default).
	ReturnComments bool

	// r is the source of the HTML text.
	r io.Reader
	// tt is the TokenType of the current token.
	tt TokenType
	// err is the first error encountered during tokenization. It is possible
	// for tt != Error && err != nil to hold: this means that Next returned a
	// valid token but the subsequent Next call will return an error token.
	// For example, if the HTML text input was just "plain", then the first
	// Next call would set z.err to os.EOF but return a TextToken, and all
	// subsequent Next calls would return an ErrorToken.
	// err is never reset. Once it becomes non-nil, it stays non-nil.
	err os.Error
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
}

// Error returns the error associated with the most recent ErrorToken token.
// This is typically os.EOF, meaning the end of tokenization.
func (z *Tokenizer) Error() os.Error {
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

// nextComment reads the next token starting with "<!--".
// The opening "<!--" has already been consumed.
// Pre-condition: z.tt == TextToken && z.err == nil &&
//   z.raw.start + 4 <= z.raw.end.
func (z *Tokenizer) nextComment() {
	// <!--> is a valid comment.
	for dashCount := 2; ; {
		c := z.readByte()
		if z.err != nil {
			z.data = z.raw
			return
		}
		switch c {
		case '-':
			dashCount++
		case '>':
			if dashCount >= 2 {
				z.tt = CommentToken
				// TODO: adjust z.data to be only the "x" in "<!--x-->".
				// Note that "<!>" is also a valid HTML5 comment.
				z.data = z.raw
				return
			}
			dashCount = 0
		default:
			dashCount = 0
		}
	}
}

// nextMarkupDeclaration reads the next token starting with "<!".
// It might be a "<!--comment-->", a "<!DOCTYPE foo>", or "<!malformed text".
// The opening "<!" has already been consumed.
// Pre-condition: z.tt == TextToken && z.err == nil &&
//   z.raw.start + 2 <= z.raw.end.
func (z *Tokenizer) nextMarkupDeclaration() {
	var c [2]byte
	for i := 0; i < 2; i++ {
		c[i] = z.readByte()
		if z.err != nil {
			return
		}
	}
	if c[0] == '-' && c[1] == '-' {
		z.nextComment()
		return
	}
	z.raw.end -= 2
	const s = "DOCTYPE "
	for i := 0; ; i++ {
		c := z.readByte()
		if z.err != nil {
			z.data = z.raw
			return
		}
		// Capitalize c.
		if 'a' <= c && c <= 'z' {
			c = 'A' + (c - 'a')
		}
		if i < len(s) && c != s[i] {
			z.nextText()
			return
		}
		if c == '>' {
			if i >= len(s) {
				z.tt = DoctypeToken
				z.data.start = z.raw.start + len("<!DOCTYPE ")
				z.data.end = z.raw.end - len(">")
			}
			return
		}
	}
}

// nextTag reads the next token starting with "<". It might be a "<startTag>",
// an "</endTag>", a "<!markup declaration>", or "<malformed text".
// The opening "<" has already been consumed.
// Pre-condition: z.tt == TextToken && z.err == nil &&
//   z.raw.start + 1 <= z.raw.end.
func (z *Tokenizer) nextTag() {
	c := z.readByte()
	if z.err != nil {
		z.data = z.raw
		return
	}
	switch {
	// TODO: check that the "</" is followed by something in A-Za-z.
	case c == '/':
		z.tt = EndTagToken
		z.data.start += len("</")
	// Lower-cased characters are more common in tag names, so we check for them first.
	case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
		z.tt = StartTagToken
		z.data.start += len("<")
	case c == '!':
		z.nextMarkupDeclaration()
		return
	case c == '?':
		z.tt, z.err = ErrorToken, os.NewError("html: TODO: implement XML processing instructions")
		return
	default:
		z.tt, z.err = ErrorToken, os.NewError("html: TODO: handle malformed tags")
		return
	}
	// Read the tag name and attribute key/value pairs.
	z.readTagName()
	for {
		if z.skipWhiteSpace(); z.err != nil {
			break
		}
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
	}
	// Check for a self-closing token.
	if z.err == nil && z.tt == StartTagToken && z.buf[z.raw.end-2] == '/' {
		z.tt = SelfClosingTagToken
	}
}

// readTagName sets z.data to the "p" in "<p k=v>".
func (z *Tokenizer) readTagName() {
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

// readTagAttrKey sets z.pendingAttr[0] to the "k" in "<p k=v>".
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

// readTagAttrVal sets z.pendingAttr[1] to the "v" in "<p k=v>".
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

// nextText reads all text up until an '<'.
// Pre-condition: z.tt == TextToken && z.err == nil && z.raw.start + 1 <= z.raw.end.
func (z *Tokenizer) nextText() {
	for {
		c := z.readByte()
		if z.err != nil {
			z.data = z.raw
			return
		}
		if c == '<' {
			z.raw.end--
			z.data = z.raw
			return
		}
	}
}

// Next scans the next token and returns its type.
func (z *Tokenizer) Next() TokenType {
	for {
		if z.err != nil {
			z.tt = ErrorToken
			return z.tt
		}
		z.raw.start = z.raw.end
		z.data.start = z.raw.end
		z.data.end = z.raw.end
		z.attr = z.attr[:0]
		z.nAttrReturned = 0

		c := z.readByte()
		if z.err != nil {
			z.tt = ErrorToken
			return z.tt
		}
		// We assume that the next token is text unless proven otherwise.
		z.tt = TextToken
		if c != '<' {
			z.nextText()
		} else {
			z.nextTag()
			if z.tt == CommentToken && !z.ReturnComments {
				continue
			}
		}
		return z.tt
	}
	panic("unreachable")
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
		return unescape(s)
	}
	return nil
}

// TagName returns the lower-cased name of a tag token (the `img` out of
// `<IMG SRC="foo">`) and whether the tag has attributes.
// The contents of the returned slice may change on the next call to Next.
func (z *Tokenizer) TagName() (name []byte, hasAttr bool) {
	switch z.tt {
	case StartTagToken, EndTagToken, SelfClosingTagToken:
		s := z.buf[z.data.start:z.data.end]
		z.data.start = z.raw.end
		z.data.end = z.raw.end
		return lower(s), z.nAttrReturned < len(z.attr)
	}
	return nil, false
}

// TagAttr returns the lower-cased key and unescaped value of the next unparsed
// attribute for the current tag token and whether there are more attributes.
// The contents of the returned slices may change on the next call to Next.
func (z *Tokenizer) TagAttr() (key, val []byte, moreAttr bool) {
	if z.nAttrReturned < len(z.attr) {
		switch z.tt {
		case StartTagToken, EndTagToken, SelfClosingTagToken:
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
	case StartTagToken, EndTagToken, SelfClosingTagToken:
		var attr []Attribute
		name, moreAttr := z.TagName()
		for moreAttr {
			var key, val []byte
			key, val, moreAttr = z.TagAttr()
			attr = append(attr, Attribute{string(key), string(val)})
		}
		t.Data = string(name)
		t.Attr = attr
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
