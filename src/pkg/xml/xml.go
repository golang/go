// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package xml implements a simple XML 1.0 parser that
// understands XML name spaces.
package xml

// References:
//    Annotated XML spec: http://www.xml.com/axml/testaxml.htm
//    XML name spaces: http://www.w3.org/TR/REC-xml-names/

// TODO(rsc):
//	Test error handling.
//	Expose parser line number in errors.

import (
	"bufio";
	"bytes";
	"io";
	"os";
	"strconv";
	"strings";
	"unicode";
	"utf8";
)

// A SyntaxError represents a syntax error in the XML input stream.
type SyntaxError string
func (e SyntaxError) String() string {
	return "XML syntax error: " + string(e);
}

// A Name represents an XML name (Local) annotated
// with a name space identifier (Space).
// In tokens returned by Parser.Token, the Space identifier
// is given as a canonical URL, not the short prefix used
// in the document being parsed.
type Name struct {
	Space, Local string;
}

// An Attr represents an attribute in an XML element (Name=Value).
type Attr struct {
	Name Name;
	Value string;
}

// A Token is an interface holding one of the token types:
// StartElement, EndElement, CharData, Comment, ProcInst, or Directive.
type Token interface{}

// A StartElement represents an XML start element.
type StartElement struct {
	Name Name;
	Attr []Attr;
}

// An EndElement represents an XML end element.
type EndElement  struct {
	Name Name;
}

// A CharData represents XML character data (raw text),
// in which XML escape sequences have been replaced by
// the characters they represent.
type CharData []byte

func copy(b []byte) []byte {
	b1 := make([]byte, len(b));
	bytes.Copy(b1, b);
	return b1;
}

func (c CharData) Copy() CharData {
	return CharData(copy(c));
}

// A Comment represents an XML comment of the form <!--comment-->.
// The bytes do not include the <!-- and --> comment markers.
type Comment []byte

func (c Comment) Copy() Comment {
	return Comment(copy(c));
}

// A ProcInst represents an XML processing instruction of the form <?target inst?>
type ProcInst struct {
	Target string;
	Inst []byte;
}

func (p ProcInst) Copy() ProcInst {
	p.Inst = copy(p.Inst);
	return p;
}

// A Directive represents an XML directive of the form <!text>.
// The bytes do not include the <! and > markers.
type Directive []byte

func (d Directive) Copy() Directive {
	return Directive(copy(d));
}

type readByter interface {
	ReadByte() (b byte, err os.Error)
}

// A Parser represents an XML parser reading a particular input stream.
// The parser assumes that its input is encoded in UTF-8.
type Parser struct {
	r readByter;
	buf bytes.Buffer;
	stk *stack;
	free *stack;
	needClose bool;
	toClose Name;
	nextByte int;
	ns map[string]string;
	err os.Error;
	line int;
	tmp [32]byte;
}

// NewParser creates a new XML parser reading from r.
func NewParser(r io.Reader) *Parser {
	p := &Parser{
		ns: make(map[string]string),
		nextByte: -1,
		line: 1,
	};

	// Get efficient byte at a time reader.
	// Assume that if reader has its own
	// ReadByte, it's efficient enough.
	// Otherwise, use bufio.
	if rb, ok := r.(readByter); ok {
		p.r = rb;
	} else {
		p.r = bufio.NewReader(r);
	}

	return p;
}

// Token returns the next XML token in the input stream.
// At the end of the input stream, Token returns nil, os.EOF.
//
// Slices of bytes in the returned token data refer to the
// parser's internal buffer and remain valid only until the next
// call to Token.  To acquire a copy of the bytes, call the token's
// Copy method.
//
// Token expands self-closing elements such as <br/>
// into separate start and end elements returned by successive calls.
//
// Token guarantees that the StartElement and EndElement
// tokens it returns are properly nested and matched:
// if Token encounters an unexpected end element,
// it will return an error.
//
// Token implements XML name spaces as described by
// http://www.w3.org/TR/REC-xml-names/.  Each of the
// Name structures contained in the Token has the Space
// set to the URL identifying its name space when known.
// If Token encounters an unrecognized name space prefix,
// it uses the prefix as the Space rather than report an error.
//
func (p *Parser) Token() (t Token, err os.Error) {
	if t, err = p.RawToken(); err != nil {
		return;
	}
	switch t1 := t.(type) {
	case StartElement:
		// In XML name spaces, the translations listed in the
		// attributes apply to the element name and
		// to the other attribute names, so process
		// the translations first.
		for _, a := range t1.Attr {
			if a.Name.Space == "xmlns" {
				v, ok := p.ns[a.Name.Local];
				p.pushNs(a.Name.Local, v, ok);
				p.ns[a.Name.Local] = a.Value;
			}
			if a.Name.Space == "" && a.Name.Local == "xmlns" {
				// Default space for untagged names
				v, ok := p.ns[""];
				p.pushNs("", v, ok);
				p.ns[""] = a.Value;
			}
		}

		p.translate(&t1.Name, true);
		for i := range t1.Attr {
			p.translate(&t1.Attr[i].Name, false);
		}
		p.pushElement(t1.Name);
		t = t1;

	case EndElement:
		p.translate(&t1.Name, true);
		if !p.popElement(t1.Name) {
			return nil, p.err;
		}
		t = t1;
	}
	return;
}

// Apply name space translation to name n.
// The default name space (for Space=="")
// applies only to element names, not to attribute names.
func (p *Parser) translate(n *Name, isElementName bool) {
	switch {
	case n.Space == "xmlns":
		return;
	case n.Space == "" && !isElementName:
		return;
	case n.Space == "" && n.Local == "xmlns":
		return;
	}
	if v, ok := p.ns[n.Space]; ok {
		n.Space = v;
	}
}

// Parsing state - stack holds old name space translations
// and the current set of open elements.  The translations to pop when
// ending a given tag are *below* it on the stack, which is
// more work but forced on us by XML.
type stack struct {
	next *stack;
	kind int;
	name Name;
	ok bool;
}

const (
	stkStart = iota;
	stkNs;
)

func (p *Parser) push(kind int) *stack {
	s := p.free;
	if s != nil {
		p.free = s.next;
	} else {
		s = new(stack);
	}
	s.next = p.stk;
	s.kind = kind;
	p.stk = s;
	return s;
}

func (p *Parser) pop() *stack {
	s := p.stk;
	if s != nil {
		p.stk = s.next;
		s.next = p.free;
		p.free = s;
	}
	return s;
}

// Record that we are starting an element with the given name.
func (p *Parser) pushElement(name Name) {
	s := p.push(stkStart);
	s.name = name;
}

// Record that we are changing the value of ns[local].
// The old value is url, ok.
func (p *Parser) pushNs(local string, url string, ok bool) {
	s := p.push(stkNs);
	s.name.Local = local;
	s.name.Space = url;
	s.ok = ok;
}

// Record that we are ending an element with the given name.
// The name must match the record at the top of the stack,
// which must be a pushElement record.
// After popping the element, apply any undo records from
// the stack to restore the name translations that existed
// before we saw this element.
func (p *Parser) popElement(name Name) bool {
	s := p.pop();
	switch {
	case s == nil || s.kind != stkStart:
		p.err = SyntaxError("unexpected end element </" + name.Local + ">");
		return false;
	case s.name.Local != name.Local:
		p.err = SyntaxError("element <" + s.name.Local + "> closed by </" + name.Local + ">");
		return false;
	case s.name.Space != name.Space:
		p.err = SyntaxError("element <" + s.name.Local + "> in space " + s.name.Space +
			"closed by </" + name.Local + "> in space " + name.Space);
		return false;
	}

	// Pop stack until a Start is on the top, undoing the
	// translations that were associated with the element we just closed.
	for p.stk != nil && p.stk.kind != stkStart {
		s := p.pop();
		p.ns[s.name.Local] = s.name.Space, s.ok;
	}

	return true;
}

// RawToken is like Token but does not verify that
// start and end elements match and does not translate
// name space prefixes to their corresponding URLs.
func (p *Parser) RawToken() (Token, os.Error) {
	if p.err != nil {
		return nil, p.err;
	}
	if p.needClose {
		// The last element we read was self-closing and
		// we returned just the StartElement half.
		// Return the EndElement half now.
		p.needClose = false;
		return EndElement{p.toClose}, nil;
	}

	b, ok := p.getc();
	if !ok {
		return nil, p.err;
	}

	if b != '<' {
		// Text section.
		p.ungetc(b);
		data := p.text(-1, false);
		if data == nil {
			return nil, p.err;
		}
		return CharData(data), nil;
	}

	if b, ok = p.getc(); !ok {
		return nil, p.err;
	}
	switch b {
	case '/':
		// </: End element
		var name Name;
		if name, ok = p.nsname(); !ok {
			if p.err == nil {
				p.err = SyntaxError("expected element name after </");
			}
			return nil, p.err;
		}
		p.space();
		if b, ok = p.getc(); !ok {
			return nil, p.err;
		}
		if b != '>' {
			p.err = SyntaxError("invalid characters between </" + name.Local + " and >");
			return nil, p.err;
		}
		return EndElement{name}, nil;

	case '?':
		// <?: Processing instruction.
		// TODO(rsc): Should parse the <?xml declaration to make sure
		// the version is 1.0 and the encoding is UTF-8.
		var target string;
		if target, ok = p.name(); !ok {
			return nil, p.err;
		}
		p.space();
		p.buf.Reset();
		var b0 byte;
		for {
			if b, ok = p.getc(); !ok {
				if p.err == os.EOF {
					p.err = SyntaxError("unterminated <? directive");
				}
				return nil, p.err;
			}
			p.buf.WriteByte(b);
			if b0 == '?' && b == '>' {
				break;
			}
			b0 = b;
		}
		data := p.buf.Bytes();
		data = data[0:len(data)-2];	// chop ?>
		return ProcInst{target, data}, nil;

	case '!':
		// <!: Maybe comment, maybe CDATA.
		if b, ok = p.getc(); !ok {
			return nil, p.err;
		}
		switch b {
		case '-':  // <!-
			// Probably <!-- for a comment.
			if b, ok = p.getc(); !ok {
				return nil, p.err;
			}
			if b != '-' {
				p.err = SyntaxError("invalid sequence <!- not part of <!--");
				return nil, p.err;
			}
			// Look for terminator.
			p.buf.Reset();
			var b0, b1 byte;
			for {
				if b, ok = p.getc(); !ok {
					if p.err == os.EOF {
						p.err = SyntaxError("unterminated <!-- comment");
					}
					return nil, p.err;
				}
				p.buf.WriteByte(b);
				if b0 == '-' && b1 == '-' && b == '>' {
					break;
				}
				b0, b1 = b1, b;
			}
			data := p.buf.Bytes();
			data = data[0:len(data)-3];	// chop -->
			return Comment(data), nil;

		case '[':  // <![
			// Probably <![CDATA[.
			for i := 0; i < 7; i++ {
				if b, ok = p.getc(); !ok {
					return nil, p.err;
				}
				if b != "[CDATA["[i] {
					p.err = SyntaxError("invalid <![ sequence");
					return nil, p.err;
				}
			}
			// Have <![CDATA[.  Read text until ]]>.
			data := p.text(-1, true);
			if data == nil {
				return nil, p.err;
			}
			return CharData(data), nil;
		}

		// Probably a directive: <!DOCTYPE ...>, <!ENTITY ...>, etc.
		// We don't care, but accumulate for caller.
		p.buf.Reset();
		p.buf.WriteByte(b);
		for {
			if b, ok = p.getc(); !ok {
				return nil, p.err;
			}
			if b == '>' {
				break;
			}
			p.buf.WriteByte(b);
		}
		return Directive(p.buf.Bytes()), nil;
	}

	// Must be an open element like <a href="foo">
	p.ungetc(b);

	var (
		name Name;
		empty bool;
		attr []Attr;
	)
	if name, ok = p.nsname(); !ok {
		if p.err == nil {
			p.err = SyntaxError("expected element name after <");
		}
		return nil, p.err;
	}

	attr = make([]Attr, 0, 4);
	for {
		p.space();
		if b, ok = p.getc(); !ok {
			return nil, p.err;
		}
		if b == '/' {
			empty = true;
			if b, ok = p.getc(); !ok {
				return nil, p.err;
			}
			if b != '>' {
				p.err = SyntaxError("expected /> in element");
				return nil, p.err;
			}
			break;
		}
		if b == '>' {
			break;
		}
		p.ungetc(b);

		n := len(attr);
		if n >= cap(attr) {
			nattr := make([]Attr, n, 2*cap(attr));
			for i, a := range attr {
				nattr[i] = a;
			}
			attr = nattr;
		}
		attr = attr[0:n+1];
		a := &attr[n];
		if a.Name, ok = p.nsname(); !ok {
			if p.err == nil {
				p.err = SyntaxError("expected attribute name in element");
			}
			return nil, p.err;
		}
		p.space();
		if b, ok = p.getc(); !ok {
			return nil, p.err;
		}
		if b != '=' {
			p.err = SyntaxError("attribute name without = in element");
			return nil, p.err;
		}
		p.space();
		if b, ok = p.getc(); !ok {
			return nil, p.err;
		}
		if b != '"' && b != '\'' {
			p.err = SyntaxError("unquoted or missing attribute value in element");
			return nil, p.err;
		}
		data := p.text(int(b), false);
		if data == nil {
			return nil, p.err;
		}
		a.Value = string(data);
	}

	if empty {
		p.needClose = true;
		p.toClose = name;
	}
	return StartElement{name, attr}, nil;
}

// Skip spaces if any
func (p *Parser) space() {
	for {
		b, ok := p.getc();
		if !ok {
			return;
		}
		switch b {
		case ' ', '\r', '\n', '\t':
		default:
			p.ungetc(b);
			return;
		}
	}
}

// Read a single byte.
// If there is no byte to read, return ok==false
// and leave the error in p.err.
// Maintain line number.
func (p *Parser) getc() (b byte, ok bool) {
	if p.err != nil {
		return 0, false;
	}
	if p.nextByte >= 0 {
		b = byte(p.nextByte);
		p.nextByte = -1;
	} else {
		b, p.err = p.r.ReadByte();
		if p.err != nil {
			return 0, false;
		}
	}
	if b == '\n' {
		p.line++;
	}
	return b, true;
}

// Unread a single byte.
func (p *Parser) ungetc(b byte) {
	if b == '\n' {
		p.line--;
	}
	p.nextByte = int(b);
}

var entity = map[string]int {
	"lt": '<',
	"gt": '>',
	"amp": '&',
	"apos": '\'',
	"quot": '"',
}

// Read plain text section (XML calls it character data).
// If quote >= 0, we are in a quoted string and need to find the matching quote.
// If cdata == true, we are in a <![CDATA[ section and need to find ]]>.
// On failure return nil and leave the error in p.err.
func (p *Parser) text(quote int, cdata bool) []byte {
	var b0, b1 byte;
	var trunc int;
	p.buf.Reset();
Input:
	for {
		b, ok := p.getc();
		if !ok {
			return nil;
		}

		// <![CDATA[ section ends with ]]>.
		// It is an error for ]]> to appear in ordinary text.
		if b0 == ']' && b1 == ']' && b == '>' {
			if cdata {
				trunc = 2;
				break Input;
			}
			p.err = SyntaxError("unescaped ]]> not in CDATA section");
			return nil;
		}

		// Stop reading text if we see a <.
		if b == '<' && !cdata {
			if quote >= 0 {
				p.err = SyntaxError("unescaped < inside quoted string");
				return nil;
			}
			p.ungetc('<');
			break Input;
		}
		if quote >= 0 && b == byte(quote) {
			break Input;
		}
		if b == '&' {
			// Read escaped character expression up to semicolon.
			// XML in all its glory allows a document to define and use
			// its own character names with <!ENTITY ...> directives.
			// Parsers are required to recognize lt, gt, amp, apos, and quot
			// even if they have not been declared.  That's all we allow.
			var i int;
			for i = 0; i < len(p.tmp); i++ {
				p.tmp[i], p.err = p.r.ReadByte();
				if p.err != nil {
					return nil;
				}
				if p.tmp[i] == ';' {
					break;
				}
			}
			s := string(p.tmp[0:i]);
			if i >= len(p.tmp) {
				p.err = SyntaxError("character entity expression &" + s + "... too long");
				return nil;
			}
			rune := -1;
			if i >= 2 && s[0] == '#' {
				var n uint64;
				var err os.Error;
				if i >= 3 && s[1] == 'x' {
					n, err = strconv.Btoui64(s[2:len(s)], 16);
				} else {
					n, err = strconv.Btoui64(s[1:len(s)], 10);
				}
				if err == nil && n <= unicode.MaxRune {
					rune = int(n);
				}
			} else {
				if r, ok := entity[s]; ok {
					rune = r;
				}
			}
			if rune < 0 {
				p.err = SyntaxError("invalid character entity &" + s + ";");
				return nil;
			}
			i = utf8.EncodeRune(rune, &p.tmp);
			p.buf.Write(p.tmp[0:i]);
			b0, b1 = 0, 0;
			continue Input;
		}
		p.buf.WriteByte(b);
		b0, b1 = b1, b;
	}
	data := p.buf.Bytes();
	data = data[0:len(data)-trunc];

	// Must rewrite \r and \r\n into \n.
	w := 0;
	for r := 0; r < len(data); r++ {
		b := data[r];
		if b == '\r' {
			if r+1 < len(data) && data[r+1] == '\n' {
				continue;
			}
			b = '\n';
		}
		data[w] = b;
		w++;
	}
	return data[0:w];
}

// Get name space name: name with a : stuck in the middle.
// The part before the : is the name space identifier.
func (p *Parser) nsname() (name Name, ok bool) {
	s, ok := p.name();
	if !ok {
		return;
	}
	i := strings.Index(s, ":");
	if i < 0 {
		name.Local = s;
	} else {
		name.Space = s[0:i];
		name.Local = s[i+1:len(s)];
	}
	return name, true;
}

// Get name: /first(first|second)*/
// Do not set p.err if the name is missing: let the caller provide better context.
func (p *Parser) name() (s string, ok bool) {
	var b byte;
	if b, ok = p.getc(); !ok {
		return;
	}

	// As a first approximation, we gather the bytes [A-Za-z_:.-\x80-\xFF]*
	if b < utf8.RuneSelf && !isNameByte(b) {
		p.ungetc(b);
		return;
	}
	p.buf.Reset();
	p.buf.WriteByte(b);
	for {
		if b, ok = p.getc(); !ok {
			return;
		}
		if b < utf8.RuneSelf && !isNameByte(b) {
			p.ungetc(b);
			break;
		}
		p.buf.WriteByte(b);
	}

	// Then we check the characters.
	s = p.buf.String();
	for i, c := range s {
		if !unicode.Is(first, c) && (i == 0 || !unicode.Is(second, c)) {
			p.err = SyntaxError("invalid XML name: " + s);
			return "", false;
		}
	}
	return s, true;
}

func isNameByte(c byte) bool {
	return 'A' <= c && c <= 'Z' ||
		'a' <= c && c <= 'z' ||
		c == '_' || c == ':' || c == '.' || c == '-';
}

// These tables were generated by cut and paste from Appendix B of
// the XML spec at http://www.xml.com/axml/testaxml.htm
// and then reformatting.  First corresponds to (Letter | '_' | ':')
// and second corresponds to NameChar.

var first = []unicode.Range{
	unicode.Range{0x003A, 0x003A, 1},
	unicode.Range{0x0041, 0x005A, 1},
	unicode.Range{0x005F, 0x005F, 1},
	unicode.Range{0x0061, 0x007A, 1},
	unicode.Range{0x00C0, 0x00D6, 1},
	unicode.Range{0x00D8, 0x00F6, 1},
	unicode.Range{0x00F8, 0x00FF, 1},
	unicode.Range{0x0100, 0x0131, 1},
	unicode.Range{0x0134, 0x013E, 1},
	unicode.Range{0x0141, 0x0148, 1},
	unicode.Range{0x014A, 0x017E, 1},
	unicode.Range{0x0180, 0x01C3, 1},
	unicode.Range{0x01CD, 0x01F0, 1},
	unicode.Range{0x01F4, 0x01F5, 1},
	unicode.Range{0x01FA, 0x0217, 1},
	unicode.Range{0x0250, 0x02A8, 1},
	unicode.Range{0x02BB, 0x02C1, 1},
	unicode.Range{0x0386, 0x0386, 1},
	unicode.Range{0x0388, 0x038A, 1},
	unicode.Range{0x038C, 0x038C, 1},
	unicode.Range{0x038E, 0x03A1, 1},
	unicode.Range{0x03A3, 0x03CE, 1},
	unicode.Range{0x03D0, 0x03D6, 1},
	unicode.Range{0x03DA, 0x03E0, 2},
	unicode.Range{0x03E2, 0x03F3, 1},
	unicode.Range{0x0401, 0x040C, 1},
	unicode.Range{0x040E, 0x044F, 1},
	unicode.Range{0x0451, 0x045C, 1},
	unicode.Range{0x045E, 0x0481, 1},
	unicode.Range{0x0490, 0x04C4, 1},
	unicode.Range{0x04C7, 0x04C8, 1},
	unicode.Range{0x04CB, 0x04CC, 1},
	unicode.Range{0x04D0, 0x04EB, 1},
	unicode.Range{0x04EE, 0x04F5, 1},
	unicode.Range{0x04F8, 0x04F9, 1},
	unicode.Range{0x0531, 0x0556, 1},
	unicode.Range{0x0559, 0x0559, 1},
	unicode.Range{0x0561, 0x0586, 1},
	unicode.Range{0x05D0, 0x05EA, 1},
	unicode.Range{0x05F0, 0x05F2, 1},
	unicode.Range{0x0621, 0x063A, 1},
	unicode.Range{0x0641, 0x064A, 1},
	unicode.Range{0x0671, 0x06B7, 1},
	unicode.Range{0x06BA, 0x06BE, 1},
	unicode.Range{0x06C0, 0x06CE, 1},
	unicode.Range{0x06D0, 0x06D3, 1},
	unicode.Range{0x06D5, 0x06D5, 1},
	unicode.Range{0x06E5, 0x06E6, 1},
	unicode.Range{0x0905, 0x0939, 1},
	unicode.Range{0x093D, 0x093D, 1},
	unicode.Range{0x0958, 0x0961, 1},
	unicode.Range{0x0985, 0x098C, 1},
	unicode.Range{0x098F, 0x0990, 1},
	unicode.Range{0x0993, 0x09A8, 1},
	unicode.Range{0x09AA, 0x09B0, 1},
	unicode.Range{0x09B2, 0x09B2, 1},
	unicode.Range{0x09B6, 0x09B9, 1},
	unicode.Range{0x09DC, 0x09DD, 1},
	unicode.Range{0x09DF, 0x09E1, 1},
	unicode.Range{0x09F0, 0x09F1, 1},
	unicode.Range{0x0A05, 0x0A0A, 1},
	unicode.Range{0x0A0F, 0x0A10, 1},
	unicode.Range{0x0A13, 0x0A28, 1},
	unicode.Range{0x0A2A, 0x0A30, 1},
	unicode.Range{0x0A32, 0x0A33, 1},
	unicode.Range{0x0A35, 0x0A36, 1},
	unicode.Range{0x0A38, 0x0A39, 1},
	unicode.Range{0x0A59, 0x0A5C, 1},
	unicode.Range{0x0A5E, 0x0A5E, 1},
	unicode.Range{0x0A72, 0x0A74, 1},
	unicode.Range{0x0A85, 0x0A8B, 1},
	unicode.Range{0x0A8D, 0x0A8D, 1},
	unicode.Range{0x0A8F, 0x0A91, 1},
	unicode.Range{0x0A93, 0x0AA8, 1},
	unicode.Range{0x0AAA, 0x0AB0, 1},
	unicode.Range{0x0AB2, 0x0AB3, 1},
	unicode.Range{0x0AB5, 0x0AB9, 1},
	unicode.Range{0x0ABD, 0x0AE0, 0x23},
	unicode.Range{0x0B05, 0x0B0C, 1},
	unicode.Range{0x0B0F, 0x0B10, 1},
	unicode.Range{0x0B13, 0x0B28, 1},
	unicode.Range{0x0B2A, 0x0B30, 1},
	unicode.Range{0x0B32, 0x0B33, 1},
	unicode.Range{0x0B36, 0x0B39, 1},
	unicode.Range{0x0B3D, 0x0B3D, 1},
	unicode.Range{0x0B5C, 0x0B5D, 1},
	unicode.Range{0x0B5F, 0x0B61, 1},
	unicode.Range{0x0B85, 0x0B8A, 1},
	unicode.Range{0x0B8E, 0x0B90, 1},
	unicode.Range{0x0B92, 0x0B95, 1},
	unicode.Range{0x0B99, 0x0B9A, 1},
	unicode.Range{0x0B9C, 0x0B9C, 1},
	unicode.Range{0x0B9E, 0x0B9F, 1},
	unicode.Range{0x0BA3, 0x0BA4, 1},
	unicode.Range{0x0BA8, 0x0BAA, 1},
	unicode.Range{0x0BAE, 0x0BB5, 1},
	unicode.Range{0x0BB7, 0x0BB9, 1},
	unicode.Range{0x0C05, 0x0C0C, 1},
	unicode.Range{0x0C0E, 0x0C10, 1},
	unicode.Range{0x0C12, 0x0C28, 1},
	unicode.Range{0x0C2A, 0x0C33, 1},
	unicode.Range{0x0C35, 0x0C39, 1},
	unicode.Range{0x0C60, 0x0C61, 1},
	unicode.Range{0x0C85, 0x0C8C, 1},
	unicode.Range{0x0C8E, 0x0C90, 1},
	unicode.Range{0x0C92, 0x0CA8, 1},
	unicode.Range{0x0CAA, 0x0CB3, 1},
	unicode.Range{0x0CB5, 0x0CB9, 1},
	unicode.Range{0x0CDE, 0x0CDE, 1},
	unicode.Range{0x0CE0, 0x0CE1, 1},
	unicode.Range{0x0D05, 0x0D0C, 1},
	unicode.Range{0x0D0E, 0x0D10, 1},
	unicode.Range{0x0D12, 0x0D28, 1},
	unicode.Range{0x0D2A, 0x0D39, 1},
	unicode.Range{0x0D60, 0x0D61, 1},
	unicode.Range{0x0E01, 0x0E2E, 1},
	unicode.Range{0x0E30, 0x0E30, 1},
	unicode.Range{0x0E32, 0x0E33, 1},
	unicode.Range{0x0E40, 0x0E45, 1},
	unicode.Range{0x0E81, 0x0E82, 1},
	unicode.Range{0x0E84, 0x0E84, 1},
	unicode.Range{0x0E87, 0x0E88, 1},
	unicode.Range{0x0E8A, 0x0E8D, 3},
	unicode.Range{0x0E94, 0x0E97, 1},
	unicode.Range{0x0E99, 0x0E9F, 1},
	unicode.Range{0x0EA1, 0x0EA3, 1},
	unicode.Range{0x0EA5, 0x0EA7, 2},
	unicode.Range{0x0EAA, 0x0EAB, 1},
	unicode.Range{0x0EAD, 0x0EAE, 1},
	unicode.Range{0x0EB0, 0x0EB0, 1},
	unicode.Range{0x0EB2, 0x0EB3, 1},
	unicode.Range{0x0EBD, 0x0EBD, 1},
	unicode.Range{0x0EC0, 0x0EC4, 1},
	unicode.Range{0x0F40, 0x0F47, 1},
	unicode.Range{0x0F49, 0x0F69, 1},
	unicode.Range{0x10A0, 0x10C5, 1},
	unicode.Range{0x10D0, 0x10F6, 1},
	unicode.Range{0x1100, 0x1100, 1},
	unicode.Range{0x1102, 0x1103, 1},
	unicode.Range{0x1105, 0x1107, 1},
	unicode.Range{0x1109, 0x1109, 1},
	unicode.Range{0x110B, 0x110C, 1},
	unicode.Range{0x110E, 0x1112, 1},
	unicode.Range{0x113C, 0x1140, 2},
	unicode.Range{0x114C, 0x1150, 2},
	unicode.Range{0x1154, 0x1155, 1},
	unicode.Range{0x1159, 0x1159, 1},
	unicode.Range{0x115F, 0x1161, 1},
	unicode.Range{0x1163, 0x1169, 2},
	unicode.Range{0x116D, 0x116E, 1},
	unicode.Range{0x1172, 0x1173, 1},
	unicode.Range{0x1175, 0x119E, 0x119E-0x1175},
	unicode.Range{0x11A8, 0x11AB, 0x11AB-0x11A8},
	unicode.Range{0x11AE, 0x11AF, 1},
	unicode.Range{0x11B7, 0x11B8, 1},
	unicode.Range{0x11BA, 0x11BA, 1},
	unicode.Range{0x11BC, 0x11C2, 1},
	unicode.Range{0x11EB, 0x11F0, 0x11F0-0x11EB},
	unicode.Range{0x11F9, 0x11F9, 1},
	unicode.Range{0x1E00, 0x1E9B, 1},
	unicode.Range{0x1EA0, 0x1EF9, 1},
	unicode.Range{0x1F00, 0x1F15, 1},
	unicode.Range{0x1F18, 0x1F1D, 1},
	unicode.Range{0x1F20, 0x1F45, 1},
	unicode.Range{0x1F48, 0x1F4D, 1},
	unicode.Range{0x1F50, 0x1F57, 1},
	unicode.Range{0x1F59, 0x1F5B, 0x1F5B-0x1F59},
	unicode.Range{0x1F5D, 0x1F5D, 1},
	unicode.Range{0x1F5F, 0x1F7D, 1},
	unicode.Range{0x1F80, 0x1FB4, 1},
	unicode.Range{0x1FB6, 0x1FBC, 1},
	unicode.Range{0x1FBE, 0x1FBE, 1},
	unicode.Range{0x1FC2, 0x1FC4, 1},
	unicode.Range{0x1FC6, 0x1FCC, 1},
	unicode.Range{0x1FD0, 0x1FD3, 1},
	unicode.Range{0x1FD6, 0x1FDB, 1},
	unicode.Range{0x1FE0, 0x1FEC, 1},
	unicode.Range{0x1FF2, 0x1FF4, 1},
	unicode.Range{0x1FF6, 0x1FFC, 1},
	unicode.Range{0x2126, 0x2126, 1},
	unicode.Range{0x212A, 0x212B, 1},
	unicode.Range{0x212E, 0x212E, 1},
	unicode.Range{0x2180, 0x2182, 1},
	unicode.Range{0x3007, 0x3007, 1},
	unicode.Range{0x3021, 0x3029, 1},
	unicode.Range{0x3041, 0x3094, 1},
	unicode.Range{0x30A1, 0x30FA, 1},
	unicode.Range{0x3105, 0x312C, 1},
	unicode.Range{0x4E00, 0x9FA5, 1},
	unicode.Range{0xAC00, 0xD7A3, 1},
}

var second = []unicode.Range{
	unicode.Range{0x002D, 0x002E, 1},
	unicode.Range{0x0030, 0x0039, 1},
	unicode.Range{0x00B7, 0x00B7, 1},
	unicode.Range{0x02D0, 0x02D1, 1},
	unicode.Range{0x0300, 0x0345, 1},
	unicode.Range{0x0360, 0x0361, 1},
	unicode.Range{0x0387, 0x0387, 1},
	unicode.Range{0x0483, 0x0486, 1},
	unicode.Range{0x0591, 0x05A1, 1},
	unicode.Range{0x05A3, 0x05B9, 1},
	unicode.Range{0x05BB, 0x05BD, 1},
	unicode.Range{0x05BF, 0x05BF, 1},
	unicode.Range{0x05C1, 0x05C2, 1},
	unicode.Range{0x05C4, 0x0640, 0x0640-0x05C4},
	unicode.Range{0x064B, 0x0652, 1},
	unicode.Range{0x0660, 0x0669, 1},
	unicode.Range{0x0670, 0x0670, 1},
	unicode.Range{0x06D6, 0x06DC, 1},
	unicode.Range{0x06DD, 0x06DF, 1},
	unicode.Range{0x06E0, 0x06E4, 1},
	unicode.Range{0x06E7, 0x06E8, 1},
	unicode.Range{0x06EA, 0x06ED, 1},
	unicode.Range{0x06F0, 0x06F9, 1},
	unicode.Range{0x0901, 0x0903, 1},
	unicode.Range{0x093C, 0x093C, 1},
	unicode.Range{0x093E, 0x094C, 1},
	unicode.Range{0x094D, 0x094D, 1},
	unicode.Range{0x0951, 0x0954, 1},
	unicode.Range{0x0962, 0x0963, 1},
	unicode.Range{0x0966, 0x096F, 1},
	unicode.Range{0x0981, 0x0983, 1},
	unicode.Range{0x09BC, 0x09BC, 1},
	unicode.Range{0x09BE, 0x09BF, 1},
	unicode.Range{0x09C0, 0x09C4, 1},
	unicode.Range{0x09C7, 0x09C8, 1},
	unicode.Range{0x09CB, 0x09CD, 1},
	unicode.Range{0x09D7, 0x09D7, 1},
	unicode.Range{0x09E2, 0x09E3, 1},
	unicode.Range{0x09E6, 0x09EF, 1},
	unicode.Range{0x0A02, 0x0A3C, 0x3A},
	unicode.Range{0x0A3E, 0x0A3F, 1},
	unicode.Range{0x0A40, 0x0A42, 1},
	unicode.Range{0x0A47, 0x0A48, 1},
	unicode.Range{0x0A4B, 0x0A4D, 1},
	unicode.Range{0x0A66, 0x0A6F, 1},
	unicode.Range{0x0A70, 0x0A71, 1},
	unicode.Range{0x0A81, 0x0A83, 1},
	unicode.Range{0x0ABC, 0x0ABC, 1},
	unicode.Range{0x0ABE, 0x0AC5, 1},
	unicode.Range{0x0AC7, 0x0AC9, 1},
	unicode.Range{0x0ACB, 0x0ACD, 1},
	unicode.Range{0x0AE6, 0x0AEF, 1},
	unicode.Range{0x0B01, 0x0B03, 1},
	unicode.Range{0x0B3C, 0x0B3C, 1},
	unicode.Range{0x0B3E, 0x0B43, 1},
	unicode.Range{0x0B47, 0x0B48, 1},
	unicode.Range{0x0B4B, 0x0B4D, 1},
	unicode.Range{0x0B56, 0x0B57, 1},
	unicode.Range{0x0B66, 0x0B6F, 1},
	unicode.Range{0x0B82, 0x0B83, 1},
	unicode.Range{0x0BBE, 0x0BC2, 1},
	unicode.Range{0x0BC6, 0x0BC8, 1},
	unicode.Range{0x0BCA, 0x0BCD, 1},
	unicode.Range{0x0BD7, 0x0BD7, 1},
	unicode.Range{0x0BE7, 0x0BEF, 1},
	unicode.Range{0x0C01, 0x0C03, 1},
	unicode.Range{0x0C3E, 0x0C44, 1},
	unicode.Range{0x0C46, 0x0C48, 1},
	unicode.Range{0x0C4A, 0x0C4D, 1},
	unicode.Range{0x0C55, 0x0C56, 1},
	unicode.Range{0x0C66, 0x0C6F, 1},
	unicode.Range{0x0C82, 0x0C83, 1},
	unicode.Range{0x0CBE, 0x0CC4, 1},
	unicode.Range{0x0CC6, 0x0CC8, 1},
	unicode.Range{0x0CCA, 0x0CCD, 1},
	unicode.Range{0x0CD5, 0x0CD6, 1},
	unicode.Range{0x0CE6, 0x0CEF, 1},
	unicode.Range{0x0D02, 0x0D03, 1},
	unicode.Range{0x0D3E, 0x0D43, 1},
	unicode.Range{0x0D46, 0x0D48, 1},
	unicode.Range{0x0D4A, 0x0D4D, 1},
	unicode.Range{0x0D57, 0x0D57, 1},
	unicode.Range{0x0D66, 0x0D6F, 1},
	unicode.Range{0x0E31, 0x0E31, 1},
	unicode.Range{0x0E34, 0x0E3A, 1},
	unicode.Range{0x0E46, 0x0E46, 1},
	unicode.Range{0x0E47, 0x0E4E, 1},
	unicode.Range{0x0E50, 0x0E59, 1},
	unicode.Range{0x0EB1, 0x0EB1, 1},
	unicode.Range{0x0EB4, 0x0EB9, 1},
	unicode.Range{0x0EBB, 0x0EBC, 1},
	unicode.Range{0x0EC6, 0x0EC6, 1},
	unicode.Range{0x0EC8, 0x0ECD, 1},
	unicode.Range{0x0ED0, 0x0ED9, 1},
	unicode.Range{0x0F18, 0x0F19, 1},
	unicode.Range{0x0F20, 0x0F29, 1},
	unicode.Range{0x0F35, 0x0F39, 2},
	unicode.Range{0x0F3E, 0x0F3F, 1},
	unicode.Range{0x0F71, 0x0F84, 1},
	unicode.Range{0x0F86, 0x0F8B, 1},
	unicode.Range{0x0F90, 0x0F95, 1},
	unicode.Range{0x0F97, 0x0F97, 1},
	unicode.Range{0x0F99, 0x0FAD, 1},
	unicode.Range{0x0FB1, 0x0FB7, 1},
	unicode.Range{0x0FB9, 0x0FB9, 1},
	unicode.Range{0x20D0, 0x20DC, 1},
	unicode.Range{0x20E1, 0x3005, 0x3005-0x20E1},
	unicode.Range{0x302A, 0x302F, 1},
	unicode.Range{0x3031, 0x3035, 1},
	unicode.Range{0x3099, 0x309A, 1},
	unicode.Range{0x309D, 0x309E, 1},
	unicode.Range{0x30FC, 0x30FE, 1},
}
