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

// A Tokenizer returns a stream of HTML Tokens.
type Tokenizer struct {
	// If ReturnComments is set, Next returns comment tokens;
	// otherwise it skips over comments (default).
	ReturnComments bool

	// r is the source of the HTML text.
	r io.Reader
	// tt is the TokenType of the most recently read token.
	tt TokenType
	// err is the first error encountered during tokenization. It is possible
	// for tt != Error && err != nil to hold: this means that Next returned a
	// valid token but the subsequent Next call will return an error token.
	// For example, if the HTML text input was just "plain", then the first
	// Next call would set z.err to os.EOF but return a TextToken, and all
	// subsequent Next calls would return an ErrorToken.
	// err is never reset. Once it becomes non-nil, it stays non-nil.
	err os.Error
	// buf[p0:p1] holds the raw data of the most recent token.
	// buf[p1:] is buffered input that will yield future tokens.
	p0, p1 int
	buf    []byte
}

// Error returns the error associated with the most recent ErrorToken token.
// This is typically os.EOF, meaning the end of tokenization.
func (z *Tokenizer) Error() os.Error {
	if z.tt != ErrorToken {
		return nil
	}
	return z.err
}

// Raw returns the unmodified text of the current token. Calling Next, Token,
// Text, TagName or TagAttr may change the contents of the returned slice.
func (z *Tokenizer) Raw() []byte {
	return z.buf[z.p0:z.p1]
}

// readByte returns the next byte from the input stream, doing a buffered read
// from z.r into z.buf if necessary. z.buf[z.p0:z.p1] remains a contiguous byte
// slice that holds all the bytes read so far for the current token.
// It sets z.err if the underlying reader returns an error.
// Pre-condition: z.err == nil.
func (z *Tokenizer) readByte() byte {
	if z.p1 >= len(z.buf) {
		// Our buffer is exhausted and we have to read from z.r.
		// We copy z.buf[z.p0:z.p1] to the beginning of z.buf. If the length
		// z.p1 - z.p0 is more than half the capacity of z.buf, then we
		// allocate a new buffer before the copy.
		c := cap(z.buf)
		d := z.p1 - z.p0
		var buf1 []byte
		if 2*d > c {
			buf1 = make([]byte, d, 2*c)
		} else {
			buf1 = z.buf[:d]
		}
		copy(buf1, z.buf[z.p0:z.p1])
		z.p0, z.p1, z.buf = 0, d, buf1[:d]
		// Now that we have copied the live bytes to the start of the buffer,
		// we read from z.r into the remainder.
		n, err := z.r.Read(buf1[d:cap(buf1)])
		if err != nil {
			z.err = err
			return 0
		}
		z.buf = buf1[:d+n]
	}
	x := z.buf[z.p1]
	z.p1++
	return x
}

// readTo keeps reading bytes until x is found or a read error occurs. If an
// error does occur, z.err is set to that error.
// Pre-condition: z.err == nil.
func (z *Tokenizer) readTo(x uint8) {
	for {
		c := z.readByte()
		if z.err != nil {
			return
		}
		switch c {
		case x:
			return
		case '\\':
			z.readByte()
			if z.err != nil {
				return
			}
		}
	}
}

// nextComment reads the next token starting with "<!--".
// The opening "<!--" has already been consumed.
// Pre-condition: z.tt == TextToken && z.err == nil && z.p0 + 4 <= z.p1.
func (z *Tokenizer) nextComment() {
	// <!--> is a valid comment.
	for dashCount := 2; ; {
		c := z.readByte()
		if z.err != nil {
			return
		}
		switch c {
		case '-':
			dashCount++
		case '>':
			if dashCount >= 2 {
				z.tt = CommentToken
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
// Pre-condition: z.tt == TextToken && z.err == nil && z.p0 + 2 <= z.p1.
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
	z.p1 -= 2
	const s = "DOCTYPE "
	for i := 0; ; i++ {
		c := z.readByte()
		if z.err != nil {
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
			}
			return
		}
	}
}

// nextTag reads the next token starting with "<". It might be a "<startTag>",
// an "</endTag>", a "<!markup declaration>", or "<malformed text".
// The opening "<" has already been consumed.
// Pre-condition: z.tt == TextToken && z.err == nil && z.p0 + 1 <= z.p1.
func (z *Tokenizer) nextTag() {
	c := z.readByte()
	if z.err != nil {
		return
	}
	switch {
	case c == '/':
		z.tt = EndTagToken
	// Lower-cased characters are more common in tag names, so we check for them first.
	case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
		z.tt = StartTagToken
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
	for {
		c := z.readByte()
		if z.err != nil {
			return
		}
		switch c {
		case '"', '\'':
			z.readTo(c)
			if z.err != nil {
				return
			}
		case '>':
			if z.buf[z.p1-2] == '/' && z.tt == StartTagToken {
				z.tt = SelfClosingTagToken
			}
			return
		}
	}
}

// nextText reads all text up until an '<'.
// Pre-condition: z.tt == TextToken && z.err == nil && z.p0 + 1 <= z.p1.
func (z *Tokenizer) nextText() {
	for {
		c := z.readByte()
		if z.err != nil {
			return
		}
		if c == '<' {
			z.p1--
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
		z.p0 = z.p1
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

// trim returns the largest j such that z.buf[i:j] contains only white space,
// or only white space plus the final ">" or "/>" of the raw data.
func (z *Tokenizer) trim(i int) int {
	k := z.p1
	for ; i < k; i++ {
		switch z.buf[i] {
		case ' ', '\n', '\t', '\f':
			continue
		case '>':
			if i == k-1 {
				return k
			}
		case '/':
			if i == k-2 {
				return k
			}
		}
		return i
	}
	return k
}

// tagName finds the tag name at the start of z.buf[i:] and returns that name
// lower-cased, as well as the trimmed cursor location afterwards.
func (z *Tokenizer) tagName(i int) ([]byte, int) {
	i0 := i
loop:
	for ; i < z.p1; i++ {
		c := z.buf[i]
		switch c {
		case ' ', '\n', '\t', '\f', '/', '>':
			break loop
		}
		if 'A' <= c && c <= 'Z' {
			z.buf[i] = c + 'a' - 'A'
		}
	}
	return z.buf[i0:i], z.trim(i)
}

// unquotedAttrVal finds the unquoted attribute value at the start of z.buf[i:]
// and returns that value, as well as the trimmed cursor location afterwards.
func (z *Tokenizer) unquotedAttrVal(i int) ([]byte, int) {
	i0 := i
loop:
	for ; i < z.p1; i++ {
		switch z.buf[i] {
		case ' ', '\n', '\t', '\f', '>':
			break loop
		case '&':
			// TODO: unescape the entity.
		}
	}
	return z.buf[i0:i], z.trim(i)
}

// attrName finds the largest attribute name at the start
// of z.buf[i:] and returns it lower-cased, as well
// as the trimmed cursor location after that name.
//
// http://dev.w3.org/html5/spec/Overview.html#syntax-attribute-name
// TODO: unicode characters
func (z *Tokenizer) attrName(i int) ([]byte, int) {
	for z.buf[i] == '/' {
		i++
		if z.buf[i] == '>' {
			return nil, z.trim(i)
		}
	}
	i0 := i
loop:
	for ; i < z.p1; i++ {
		c := z.buf[i]
		switch c {
		case '>', '/', '=':
			break loop
		}
		switch {
		case 'A' <= c && c <= 'Z':
			z.buf[i] = c + 'a' - 'A'
		case c > ' ' && c < 0x7f:
			// No-op.
		default:
			break loop
		}
	}
	return z.buf[i0:i], z.trim(i)
}

// Text returns the unescaped text of a text, comment or doctype token. The
// contents of the returned slice may change on the next call to Next.
func (z *Tokenizer) Text() []byte {
	var i0, i1 int
	switch z.tt {
	case TextToken:
		i0 = z.p0
		i1 = z.p1
	case CommentToken:
		// Trim the "<!--" from the left and the "-->" from the right.
		// "<!-->" is a valid comment, so the adjusted endpoints might overlap.
		i0 = z.p0 + 4
		i1 = z.p1 - 3
	case DoctypeToken:
		// Trim the "<!DOCTYPE " from the left and the ">" from the right.
		i0 = z.p0 + 10
		i1 = z.p1 - 1
	default:
		return nil
	}
	z.p0 = z.p1
	if i0 < i1 {
		return unescape(z.buf[i0:i1])
	}
	return nil
}

// TagName returns the lower-cased name of a tag token (the `img` out of
// `<IMG SRC="foo">`) and whether the tag has attributes.
// The contents of the returned slice may change on the next call to Next.
func (z *Tokenizer) TagName() (name []byte, hasAttr bool) {
	i := z.p0 + 1
	if i >= z.p1 {
		z.p0 = z.p1
		return nil, false
	}
	if z.buf[i] == '/' {
		i++
	}
	name, z.p0 = z.tagName(i)
	hasAttr = z.p0 != z.p1
	return
}

// TagAttr returns the lower-cased key and unescaped value of the next unparsed
// attribute for the current tag token and whether there are more attributes.
// The contents of the returned slices may change on the next call to Next.
func (z *Tokenizer) TagAttr() (key, val []byte, moreAttr bool) {
	key, i := z.attrName(z.p0)
	// Check for an empty attribute value.
	if i == z.p1 {
		z.p0 = i
		return
	}
	// Get past the equals and quote characters.
	if z.buf[i] != '=' {
		z.p0, moreAttr = i, true
		return
	}
	i = z.trim(i + 1)
	if i == z.p1 {
		z.p0 = i
		return
	}
	closeQuote := z.buf[i]
	if closeQuote != '\'' && closeQuote != '"' {
		val, z.p0 = z.unquotedAttrVal(i)
		moreAttr = z.p0 != z.p1
		return
	}
	i = z.trim(i + 1)
	// Copy and unescape everything up to the closing quote.
	dst, src := i, i
loop:
	for src < z.p1 {
		c := z.buf[src]
		switch c {
		case closeQuote:
			src++
			break loop
		case '&':
			dst, src = unescapeEntity(z.buf, dst, src, true)
		case '\\':
			if src == z.p1 {
				z.buf[dst] = '\\'
				dst++
			} else {
				z.buf[dst] = z.buf[src+1]
				dst, src = dst+1, src+2
			}
		default:
			z.buf[dst] = c
			dst, src = dst+1, src+1
		}
	}
	val, z.p0 = z.buf[i:dst], z.trim(src)
	moreAttr = z.p0 != z.p1
	return
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
