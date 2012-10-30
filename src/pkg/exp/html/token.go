// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"exp/html/atom"
	"io"
	"strconv"
	"strings"
)

// A TokenType is the type of a Token.
type TokenType uint32

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
// rather than "a&lt;b"). For tag Tokens, DataAtom is the atom for Data, or
// zero if Data is not a known tag name.
type Token struct {
	Type     TokenType
	DataAtom atom.Atom
	Data     string
	Attr     []Attribute
}

// tagString returns a string representation of a tag Token's Data and Attr.
func (t Token) tagString() string {
	if len(t.Attr) == 0 {
		return t.Data
	}
	buf := bytes.NewBufferString(t.Data)
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
	// convertNUL is whether NUL bytes in the current token's data should
	// be converted into \ufffd replacement characters.
	convertNUL bool
	// allowCDATA is whether CDATA sections are allowed in the current context.
	allowCDATA bool
}

// AllowCDATA sets whether or not the tokenizer recognizes <![CDATA[foo]]> as
// the text "foo". The default value is false, which means to recognize it as
// a bogus comment "<!-- [CDATA[foo]] -->" instead.
//
// Strictly speaking, an HTML5 compliant tokenizer should allow CDATA if and
// only if tokenizing foreign content, such as MathML and SVG. However,
// tracking foreign-contentness is difficult to do purely in the tokenizer,
// as opposed to the parser, due to HTML integration points: an <svg> element
// can contain a <foreignObject> that is foreign-to-SVG but not foreign-to-
// HTML. For strict compliance with the HTML5 tokenization algorithm, it is the
// responsibility of the user of a tokenizer to call AllowCDATA as appropriate.
// In practice, if using the tokenizer without caring whether MathML or SVG
// CDATA is text or comments, such as tokenizing HTML to find all the anchor
// text, it is acceptable to ignore this responsibility.
func (z *Tokenizer) AllowCDATA(allowCDATA bool) {
	z.allowCDATA = allowCDATA
}

// NextIsNotRawText instructs the tokenizer that the next token should not be
// considered as 'raw text'. Some elements, such as script and title elements,
// normally require the next token after the opening tag to be 'raw text' that
// has no child elements. For example, tokenizing "<title>a<b>c</b>d</title>"
// yields a start tag token for "<title>", a text token for "a<b>c</b>d", and
// an end tag token for "</title>". There are no distinct start tag or end tag
// tokens for the "<b>" and "</b>".
//
// This tokenizer implementation will generally look for raw text at the right
// times. Strictly speaking, an HTML5 compliant tokenizer should not look for
// raw text if in foreign content: <title> generally needs raw text, but a
// <title> inside an <svg> does not. Another example is that a <textarea>
// generally needs raw text, but a <textarea> is not allowed as an immediate
// child of a <select>; in normal parsing, a <textarea> implies </select>, but
// one cannot close the implicit element when parsing a <select>'s InnerHTML.
// Similarly to AllowCDATA, tracking the correct moment to override raw-text-
// ness is difficult to do purely in the tokenizer, as opposed to the parser.
// For strict compliance with the HTML5 tokenization algorithm, it is the
// responsibility of the user of a tokenizer to call NextIsNotRawText as
// appropriate. In practice, like AllowCDATA, it is acceptable to ignore this
// responsibility for basic usage.
//
// Note that this 'raw text' concept is different from the one offered by the
// Tokenizer.Raw method.
func (z *Tokenizer) NextIsNotRawText() {
	z.rawTag = ""
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
	if z.rawTag == "script" {
		z.readScript()
		z.textIsRaw = true
		z.rawTag = ""
		return
	}
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
		if z.readRawEndTag() || z.err != nil {
			break loop
		}
	}
	z.data.end = z.raw.end
	// A textarea's or title's RCDATA can contain escaped entities.
	z.textIsRaw = z.rawTag != "textarea" && z.rawTag != "title"
	z.rawTag = ""
}

// readRawEndTag attempts to read a tag like "</foo>", where "foo" is z.rawTag.
// If it succeeds, it backs up the input position to reconsume the tag and
// returns true. Otherwise it returns false. The opening "</" has already been
// consumed.
func (z *Tokenizer) readRawEndTag() bool {
	for i := 0; i < len(z.rawTag); i++ {
		c := z.readByte()
		if z.err != nil {
			return false
		}
		if c != z.rawTag[i] && c != z.rawTag[i]-('a'-'A') {
			z.raw.end--
			return false
		}
	}
	c := z.readByte()
	if z.err != nil {
		return false
	}
	switch c {
	case ' ', '\n', '\r', '\t', '\f', '/', '>':
		// The 3 is 2 for the leading "</" plus 1 for the trailing character c.
		z.raw.end -= 3 + len(z.rawTag)
		return true
	}
	z.raw.end--
	return false
}

// readScript reads until the next </script> tag, following the byzantine
// rules for escaping/hiding the closing tag.
func (z *Tokenizer) readScript() {
	defer func() {
		z.data.end = z.raw.end
	}()
	var c byte

scriptData:
	c = z.readByte()
	if z.err != nil {
		return
	}
	if c == '<' {
		goto scriptDataLessThanSign
	}
	goto scriptData

scriptDataLessThanSign:
	c = z.readByte()
	if z.err != nil {
		return
	}
	switch c {
	case '/':
		goto scriptDataEndTagOpen
	case '!':
		goto scriptDataEscapeStart
	}
	z.raw.end--
	goto scriptData

scriptDataEndTagOpen:
	if z.readRawEndTag() || z.err != nil {
		return
	}
	goto scriptData

scriptDataEscapeStart:
	c = z.readByte()
	if z.err != nil {
		return
	}
	if c == '-' {
		goto scriptDataEscapeStartDash
	}
	z.raw.end--
	goto scriptData

scriptDataEscapeStartDash:
	c = z.readByte()
	if z.err != nil {
		return
	}
	if c == '-' {
		goto scriptDataEscapedDashDash
	}
	z.raw.end--
	goto scriptData

scriptDataEscaped:
	c = z.readByte()
	if z.err != nil {
		return
	}
	switch c {
	case '-':
		goto scriptDataEscapedDash
	case '<':
		goto scriptDataEscapedLessThanSign
	}
	goto scriptDataEscaped

scriptDataEscapedDash:
	c = z.readByte()
	if z.err != nil {
		return
	}
	switch c {
	case '-':
		goto scriptDataEscapedDashDash
	case '<':
		goto scriptDataEscapedLessThanSign
	}
	goto scriptDataEscaped

scriptDataEscapedDashDash:
	c = z.readByte()
	if z.err != nil {
		return
	}
	switch c {
	case '-':
		goto scriptDataEscapedDashDash
	case '<':
		goto scriptDataEscapedLessThanSign
	case '>':
		goto scriptData
	}
	goto scriptDataEscaped

scriptDataEscapedLessThanSign:
	c = z.readByte()
	if z.err != nil {
		return
	}
	if c == '/' {
		goto scriptDataEscapedEndTagOpen
	}
	if 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' {
		goto scriptDataDoubleEscapeStart
	}
	z.raw.end--
	goto scriptData

scriptDataEscapedEndTagOpen:
	if z.readRawEndTag() || z.err != nil {
		return
	}
	goto scriptDataEscaped

scriptDataDoubleEscapeStart:
	z.raw.end--
	for i := 0; i < len("script"); i++ {
		c = z.readByte()
		if z.err != nil {
			return
		}
		if c != "script"[i] && c != "SCRIPT"[i] {
			z.raw.end--
			goto scriptDataEscaped
		}
	}
	c = z.readByte()
	if z.err != nil {
		return
	}
	switch c {
	case ' ', '\n', '\r', '\t', '\f', '/', '>':
		goto scriptDataDoubleEscaped
	}
	z.raw.end--
	goto scriptDataEscaped

scriptDataDoubleEscaped:
	c = z.readByte()
	if z.err != nil {
		return
	}
	switch c {
	case '-':
		goto scriptDataDoubleEscapedDash
	case '<':
		goto scriptDataDoubleEscapedLessThanSign
	}
	goto scriptDataDoubleEscaped

scriptDataDoubleEscapedDash:
	c = z.readByte()
	if z.err != nil {
		return
	}
	switch c {
	case '-':
		goto scriptDataDoubleEscapedDashDash
	case '<':
		goto scriptDataDoubleEscapedLessThanSign
	}
	goto scriptDataDoubleEscaped

scriptDataDoubleEscapedDashDash:
	c = z.readByte()
	if z.err != nil {
		return
	}
	switch c {
	case '-':
		goto scriptDataDoubleEscapedDashDash
	case '<':
		goto scriptDataDoubleEscapedLessThanSign
	case '>':
		goto scriptData
	}
	goto scriptDataDoubleEscaped

scriptDataDoubleEscapedLessThanSign:
	c = z.readByte()
	if z.err != nil {
		return
	}
	if c == '/' {
		goto scriptDataDoubleEscapeEnd
	}
	z.raw.end--
	goto scriptDataDoubleEscaped

scriptDataDoubleEscapeEnd:
	if z.readRawEndTag() {
		z.raw.end += len("</script>")
		goto scriptDataEscaped
	}
	if z.err != nil {
		return
	}
	goto scriptDataDoubleEscaped
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
// a "<!--comment-->", a "<!DOCTYPE foo>", a "<![CDATA[section]]>" or
// "<!a bogus comment". The opening "<!" has already been consumed.
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
	if z.readDoctype() {
		return DoctypeToken
	}
	if z.allowCDATA && z.readCDATA() {
		z.convertNUL = true
		return TextToken
	}
	// It's a bogus comment.
	z.readUntilCloseAngle()
	return CommentToken
}

// readDoctype attempts to read a doctype declaration and returns true if
// successful. The opening "<!" has already been consumed.
func (z *Tokenizer) readDoctype() bool {
	const s = "DOCTYPE"
	for i := 0; i < len(s); i++ {
		c := z.readByte()
		if z.err != nil {
			z.data.end = z.raw.end
			return false
		}
		if c != s[i] && c != s[i]+('a'-'A') {
			// Back up to read the fragment of "DOCTYPE" again.
			z.raw.end = z.data.start
			return false
		}
	}
	if z.skipWhiteSpace(); z.err != nil {
		z.data.start = z.raw.end
		z.data.end = z.raw.end
		return true
	}
	z.readUntilCloseAngle()
	return true
}

// readCDATA attempts to read a CDATA section and returns true if
// successful. The opening "<!" has already been consumed.
func (z *Tokenizer) readCDATA() bool {
	const s = "[CDATA["
	for i := 0; i < len(s); i++ {
		c := z.readByte()
		if z.err != nil {
			z.data.end = z.raw.end
			return false
		}
		if c != s[i] {
			// Back up to read the fragment of "[CDATA[" again.
			z.raw.end = z.data.start
			return false
		}
	}
	z.data.start = z.raw.end
	brackets := 0
	for {
		c := z.readByte()
		if z.err != nil {
			z.data.end = z.raw.end
			return true
		}
		switch c {
		case ']':
			brackets++
		case '>':
			if brackets >= 2 {
				z.data.end = z.raw.end - len("]]>")
				return true
			}
			brackets = 0
		default:
			brackets = 0
		}
	}
	panic("unreachable")
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
	z.readTag(true)
	if z.err != nil {
		return ErrorToken
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

// readTag reads the next tag token and its attributes. If saveAttr, those
// attributes are saved in z.attr, otherwise z.attr is set to an empty slice.
// The opening "<a" or "</a" has already been consumed, where 'a' means anything
// in [A-Za-z].
func (z *Tokenizer) readTag(saveAttr bool) {
	z.attr = z.attr[:0]
	z.nAttrReturned = 0
	// Read the tag name and attribute key/value pairs.
	z.readTagName()
	if z.skipWhiteSpace(); z.err != nil {
		return
	}
	for {
		c := z.readByte()
		if z.err != nil || c == '>' {
			break
		}
		z.raw.end--
		z.readTagAttrKey()
		z.readTagAttrVal()
		// Save pendingAttr if saveAttr and that attribute has a non-empty key.
		if saveAttr && z.pendingAttr[0].start != z.pendingAttr[0].end {
			z.attr = append(z.attr, z.pendingAttr)
		}
		if z.skipWhiteSpace(); z.err != nil {
			break
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
			z.data.end = z.raw.end
			z.textIsRaw = true
		} else {
			z.readRawOrRCDATA()
		}
		if z.data.end > z.data.start {
			z.tt = TextToken
			z.convertNUL = true
			return z.tt
		}
	}
	z.textIsRaw = false
	z.convertNUL = false

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
				z.readTag(false)
				if z.err != nil {
					z.tt = ErrorToken
				} else {
					z.tt = EndTagToken
				}
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

// convertNewlines converts "\r" and "\r\n" in s to "\n".
// The conversion happens in place, but the resulting slice may be shorter.
func convertNewlines(s []byte) []byte {
	for i, c := range s {
		if c != '\r' {
			continue
		}

		src := i + 1
		if src >= len(s) || s[src] != '\n' {
			s[i] = '\n'
			continue
		}

		dst := i
		for src < len(s) {
			if s[src] == '\r' {
				if src+1 < len(s) && s[src+1] == '\n' {
					src++
				}
				s[dst] = '\n'
			} else {
				s[dst] = s[src]
			}
			src++
			dst++
		}
		return s[:dst]
	}
	return s
}

var (
	nul         = []byte("\x00")
	replacement = []byte("\ufffd")
)

// Text returns the unescaped text of a text, comment or doctype token. The
// contents of the returned slice may change on the next call to Next.
func (z *Tokenizer) Text() []byte {
	switch z.tt {
	case TextToken, CommentToken, DoctypeToken:
		s := z.buf[z.data.start:z.data.end]
		z.data.start = z.raw.end
		z.data.end = z.raw.end
		s = convertNewlines(s)
		if (z.convertNUL || z.tt == CommentToken) && bytes.Contains(s, nul) {
			s = bytes.Replace(s, nul, replacement, -1)
		}
		if !z.textIsRaw {
			s = unescape(s, false)
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
			return lower(key), unescape(convertNewlines(val), true), z.nAttrReturned < len(z.attr)
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
	case StartTagToken, SelfClosingTagToken, EndTagToken:
		name, moreAttr := z.TagName()
		for moreAttr {
			var key, val []byte
			key, val, moreAttr = z.TagAttr()
			t.Attr = append(t.Attr, Attribute{"", atom.String(key), string(val)})
		}
		if a := atom.Lookup(name); a != 0 {
			t.DataAtom, t.Data = a, a.String()
		} else {
			t.DataAtom, t.Data = 0, string(name)
		}
	}
	return t
}

// NewTokenizer returns a new HTML Tokenizer for the given Reader.
// The input is assumed to be UTF-8 encoded.
func NewTokenizer(r io.Reader) *Tokenizer {
	return NewTokenizerFragment(r, "")
}

// NewTokenizerFragment returns a new HTML Tokenizer for the given Reader, for
// tokenizing an exisitng element's InnerHTML fragment. contextTag is that
// element's tag, such as "div" or "iframe".
//
// For example, how the InnerHTML "a<b" is tokenized depends on whether it is
// for a <p> tag or a <script> tag.
//
// The input is assumed to be UTF-8 encoded.
func NewTokenizerFragment(r io.Reader, contextTag string) *Tokenizer {
	z := &Tokenizer{
		r:   r,
		buf: make([]byte, 0, 4096),
	}
	if contextTag != "" {
		switch s := strings.ToLower(contextTag); s {
		case "iframe", "noembed", "noframes", "noscript", "plaintext", "script", "style", "title", "textarea", "xmp":
			z.rawTag = s
		}
	}
	return z
}
