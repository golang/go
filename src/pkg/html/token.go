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
// tags, content for text). A tag Token may also contain a slice of Attributes.
// Data is unescaped for both tag and text Tokens (it looks like "a<b" rather
// than "a&lt;b").
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
	}
	return "Invalid(" + strconv.Itoa(int(t.Type)) + ")"
}

// A Tokenizer returns a stream of HTML Tokens.
type Tokenizer struct {
	// r is the source of the HTML text.
	r io.Reader
	// tt is the TokenType of the most recently read token. If tt == Error
	// then err is the error associated with trying to read that token.
	tt  TokenType
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
func (z *Tokenizer) readByte() (byte, os.Error) {
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
			buf1 = z.buf[0:d]
		}
		copy(buf1, z.buf[z.p0:z.p1])
		z.p0, z.p1, z.buf = 0, d, buf1[0:d]
		// Now that we have copied the live bytes to the start of the buffer,
		// we read from z.r into the remainder.
		n, err := z.r.Read(buf1[d:cap(buf1)])
		if err != nil {
			return 0, err
		}
		z.buf = buf1[0 : d+n]
	}
	x := z.buf[z.p1]
	z.p1++
	return x, nil
}

// readTo keeps reading bytes until x is found.
func (z *Tokenizer) readTo(x uint8) os.Error {
	for {
		c, err := z.readByte()
		if err != nil {
			return err
		}
		switch c {
		case x:
			return nil
		case '\\':
			_, err = z.readByte()
			if err != nil {
				return err
			}
		}
	}
	panic("unreachable")
}

// nextTag returns the next TokenType starting from the tag open state.
func (z *Tokenizer) nextTag() (tt TokenType, err os.Error) {
	c, err := z.readByte()
	if err != nil {
		return ErrorToken, err
	}
	switch {
	case c == '/':
		tt = EndTagToken
	// Lower-cased characters are more common in tag names, so we check for them first.
	case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
		tt = StartTagToken
	case c == '!':
		return ErrorToken, os.NewError("html: TODO(nigeltao): implement comments")
	case c == '?':
		return ErrorToken, os.NewError("html: TODO(nigeltao): implement XML processing instructions")
	default:
		return ErrorToken, os.NewError("html: TODO(nigeltao): handle malformed tags")
	}
	for {
		c, err := z.readByte()
		if err != nil {
			return TextToken, err
		}
		switch c {
		case '"':
			err = z.readTo('"')
			if err != nil {
				return TextToken, err
			}
		case '\'':
			err = z.readTo('\'')
			if err != nil {
				return TextToken, err
			}
		case '>':
			if z.buf[z.p1-2] == '/' && tt == StartTagToken {
				return SelfClosingTagToken, nil
			}
			return tt, nil
		}
	}
	panic("unreachable")
}

// Next scans the next token and returns its type.
func (z *Tokenizer) Next() TokenType {
	if z.err != nil {
		z.tt = ErrorToken
		return z.tt
	}
	z.p0 = z.p1
	c, err := z.readByte()
	if err != nil {
		z.tt, z.err = ErrorToken, err
		return z.tt
	}
	if c == '<' {
		z.tt, z.err = z.nextTag()
		return z.tt
	}
	for {
		c, err := z.readByte()
		if err != nil {
			z.tt, z.err = ErrorToken, err
			if err == os.EOF {
				z.tt = TextToken
			}
			return z.tt
		}
		if c == '<' {
			z.p1--
			z.tt = TextToken
			return z.tt
		}
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

// lower finds the largest alphabetic [0-9A-Za-z]* word at the start of z.buf[i:]
// and returns that word lower-cased, as well as the trimmed cursor location
// after that word.
func (z *Tokenizer) lower(i int) ([]byte, int) {
	i0 := i
loop:
	for ; i < z.p1; i++ {
		c := z.buf[i]
		switch {
		case '0' <= c && c <= '9':
			// No-op.
		case 'A' <= c && c <= 'Z':
			z.buf[i] = c + 'a' - 'A'
		case 'a' <= c && c <= 'z':
			// No-op.
		default:
			break loop
		}
	}
	return z.buf[i0:i], z.trim(i)
}

// Text returns the raw data after unescaping.
// The contents of the returned slice may change on the next call to Next.
func (z *Tokenizer) Text() []byte {
	s := unescape(z.Raw())
	z.p0 = z.p1
	return s
}

// TagName returns the lower-cased name of a tag token (the `img` out of
// `<IMG SRC="foo">`), and whether the tag has attributes.
// The contents of the returned slice may change on the next call to Next.
func (z *Tokenizer) TagName() (name []byte, remaining bool) {
	i := z.p0 + 1
	if i >= z.p1 {
		z.p0 = z.p1
		return nil, false
	}
	if z.buf[i] == '/' {
		i++
	}
	name, z.p0 = z.lower(i)
	remaining = z.p0 != z.p1
	return
}

// TagAttr returns the lower-cased key and unescaped value of the next unparsed
// attribute for the current tag token, and whether there are more attributes.
// The contents of the returned slices may change on the next call to Next.
func (z *Tokenizer) TagAttr() (key, val []byte, remaining bool) {
	key, i := z.lower(z.p0)
	// Get past the "=\"".
	if i == z.p1 || z.buf[i] != '=' {
		return
	}
	i = z.trim(i + 1)
	if i == z.p1 || z.buf[i] != '"' {
		return
	}
	i = z.trim(i + 1)
	// Copy and unescape everything up to the closing '"'.
	dst, src := i, i
loop:
	for src < z.p1 {
		c := z.buf[src]
		switch c {
		case '"':
			src++
			break loop
		case '&':
			dst, src = unescapeEntity(z.buf, dst, src)
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
	remaining = z.p0 != z.p1
	return
}

// Token returns the next Token. The result's Data and Attr values remain valid
// after subsequent Next calls.
func (z *Tokenizer) Token() Token {
	t := Token{Type: z.tt}
	switch z.tt {
	case TextToken:
		t.Data = string(z.Text())
	case StartTagToken, EndTagToken, SelfClosingTagToken:
		var attr []Attribute
		name, remaining := z.TagName()
		for remaining {
			var key, val []byte
			key, val, remaining = z.TagAttr()
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
