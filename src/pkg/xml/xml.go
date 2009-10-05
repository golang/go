// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package xml implements a simple XML 1.0 parser that
// understands XML name spaces.
package xml

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
// Unlike most routines, do not set p.err if the name is
// merely malformed.  Let the caller provide better context.
func (p *Parser) name() (s string, ok bool) {
	var b byte;
	if b, ok = p.getc(); !ok {
		return;
	}
	if b < utf8.RuneSelf && !isFirst(b) {
		p.ungetc(b);
		return;
	}
	p.buf.Reset();
	p.buf.WriteByte(b);
	for {
		if b, ok = p.getc(); !ok {
			return;
		}
		if b < utf8.RuneSelf && !isFirst(b) && !isSecond(b) {
			p.ungetc(b);
			break;
		}
		p.buf.WriteByte(b);
	}
	return p.buf.String(), true;
}

// We allow any Unicode char >= 0x80, but the XML spec is pickier:
// the exact character sets are listed in the comment at the end of the file.
func isFirst(c byte) bool {
	return 'A' <= c && c <= 'Z' ||
		'a' <= c && c <= 'z' ||
		c == '_' ||
		c == ':';
}

func isSecond(c byte) bool {
	return c == '.' || c == '-';
}

// The precise form of an XML name is /first(first|second)*/, where
// first is one of these characters:
//
// 003A        04D0-04EB   0A59-0A5C   0C35-0C39   0F49-0F69   1E00-1E9B
// 0041-005A   04EE-04F5   0A5E        0C60-0C61   10A0-10C5   1EA0-1EF9
// 005F        04F8-04F9   0A72-0A74   0C85-0C8C   10D0-10F6   1F00-1F15
// 0061-007A   0531-0556   0A85-0A8B   0C8E-0C90   1100        1F18-1F1D
// 00C0-00D6   0559        0A8D        0C92-0CA8   1102-1103   1F20-1F45
// 00D8-00F6   0561-0586   0A8F-0A91   0CAA-0CB3   1105-1107   1F48-1F4D
// 00F8-00FF   05D0-05EA   0A93-0AA8   0CB5-0CB9   1109        1F50-1F57
// 0100-0131   05F0-05F2   0AAA-0AB0   0CDE        110B-110C   1F59
// 0134-013E   0621-063A   0AB2-0AB3   0CE0-0CE1   110E-1112   1F5B
// 0141-0148   0641-064A   0AB5-0AB9   0D05-0D0C   113C        1F5D
// 014A-017E   0671-06B7   0ABD        0D0E-0D10   113E        1F5F-1F7D
// 0180-01C3   06BA-06BE   0AE0        0D12-0D28   1140        1F80-1FB4
// 01CD-01F0   06C0-06CE   0B05-0B0C   0D2A-0D39   114C        1FB6-1FBC
// 01F4-01F5   06D0-06D3   0B0F-0B10   0D60-0D61   114E        1FBE
// 01FA-0217   06D5        0B13-0B28   0E01-0E2E   1150        1FC2-1FC4
// 0250-02A8   06E5-06E6   0B2A-0B30   0E30        1154-1155   1FC6-1FCC
// 02BB-02C1   0905-0939   0B32-0B33   0E32-0E33   1159        1FD0-1FD3
// 0386        093D        0B36-0B39   0E40-0E45   115F-1161   1FD6-1FDB
// 0388-038A   0958-0961   0B3D        0E81-0E82   1163        1FE0-1FEC
// 038C        0985-098C   0B5C-0B5D   0E84        1165        1FF2-1FF4
// 038E-03A1   098F-0990   0B5F-0B61   0E87-0E88   1167        1FF6-1FFC
// 03A3-03CE   0993-09A8   0B85-0B8A   0E8A        1169        2126
// 03D0-03D6   09AA-09B0   0B8E-0B90   0E8D        116D-116E   212A-212B
// 03DA        09B2        0B92-0B95   0E94-0E97   1172-1173   212E
// 03DC        09B6-09B9   0B99-0B9A   0E99-0E9F   1175        2180-2182
// 03DE        09DC-09DD   0B9C        0EA1-0EA3   119E        3007
// 03E0        09DF-09E1   0B9E-0B9F   0EA5        11A8        3021-3029
// 03E2-03F3   09F0-09F1   0BA3-0BA4   0EA7        11AB        3041-3094
// 0401-040C   0A05-0A0A   0BA8-0BAA   0EAA-0EAB   11AE-11AF   30A1-30FA
// 040E-044F   0A0F-0A10   0BAE-0BB5   0EAD-0EAE   11B7-11B8   3105-312C
// 0451-045C   0A13-0A28   0BB7-0BB9   0EB0        11BA        4E00-9FA5
// 045E-0481   0A2A-0A30   0C05-0C0C   0EB2-0EB3   11BC-11C2   AC00-D7A3
// 0490-04C4   0A32-0A33   0C0E-0C10   0EBD        11EB
// 04C7-04C8   0A35-0A36   0C12-0C28   0EC0-0EC4   11F0
// 04CB-04CC   0A38-0A39   0C2A-0C33   0F40-0F47   11F9
//
// and a second is one of these:
//
// 002D        06DD-06DF   09E6-09EF   0B56-0B57   0D3E-0D43   0F3E
// 002E        06E0-06E4   0A02        0B66-0B6F   0D46-0D48   0F3F
// 0030-0039   06E7-06E8   0A3C        0B82-0B83   0D4A-0D4D   0F71-0F84
// 00B7        06EA-06ED   0A3E        0BBE-0BC2   0D57        0F86-0F8B
// 02D0        06F0-06F9   0A3F        0BC6-0BC8   0D66-0D6F   0F90-0F95
// 02D1        0901-0903   0A40-0A42   0BCA-0BCD   0E31        0F97
// 0300-0345   093C        0A47-0A48   0BD7        0E34-0E3A   0F99-0FAD
// 0360-0361   093E-094C   0A4B-0A4D   0BE7-0BEF   0E46        0FB1-0FB7
// 0387        094D        0A66-0A6F   0C01-0C03   0E47-0E4E   0FB9
// 0483-0486   0951-0954   0A70-0A71   0C3E-0C44   0E50-0E59   20D0-20DC
// 0591-05A1   0962-0963   0A81-0A83   0C46-0C48   0EB1        20E1
// 05A3-05B9   0966-096F   0ABC        0C4A-0C4D   0EB4-0EB9   3005
// 05BB-05BD   0981-0983   0ABE-0AC5   0C55-0C56   0EBB-0EBC   302A-302F
// 05BF        09BC        0AC7-0AC9   0C66-0C6F   0EC6        3031-3035
// 05C1-05C2   09BE        0ACB-0ACD   0C82-0C83   0EC8-0ECD   3099
// 05C4        09BF        0AE6-0AEF   0CBE-0CC4   0ED0-0ED9   309A
// 0640        09C0-09C4   0B01-0B03   0CC6-0CC8   0F18-0F19   309D-309E
// 064B-0652   09C7-09C8   0B3C        0CCA-0CCD   0F20-0F29   30FC-30FE
// 0660-0669   09CB-09CD   0B3E-0B43   0CD5-0CD6   0F35
// 0670        09D7        0B47-0B48   0CE6-0CEF   0F37
// 06D6-06DC   09E2-09E3   0B4B-0B4D   0D02-0D03   0F39

