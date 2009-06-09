// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NOTE(rsc): Actually, this package is just a description
// of an implementation that hasn't been written yet.

// This package implements an XML parser but relies on
// clients to implement the parsing actions.

// An XML document is a single XML element.
//
// An XML element is either a start tag and an end tag,
// like <tag>...</tag>, or a combined start/end tag <tag/>.
// The latter is identical in semantics to <tag></tag>,
// and this parser does not distinguish them.
//
// The start (or combined start/end) tag can have
// name="value" attributes inside the angle brackets after
// the tag name, as in <img src="http://google.com/icon.png" alt="Google">.
// Names are drawn from a fixed set of alphabetic letters;
// Values are strings quoted with single or double quotes.
//
// An element made up of distinct start and end tags can
// contain free-form text and other elements inside it,
// as in <a href="http://www.google.com">Google</a>
// or <b><a href="http://www.google.com">Google</a></b>.
// The former is an <a> element with the text "Google" inside it.
// The latter is a <b> element with that <a> element inside it.
// In general, an element can contain a sequence of elements
// and text inside it.  In XML, white space inside an element is
// always counted as text--it is never discarded by the parser.
// XML parsers do translate \r and \r\n into \n in text.
//
// This parser reads an XML document and calls methods on a
// Builder interface object in response to the text.
// It calls the builder's StartElement, Text, and EndElement
// methods, mimicking the structure of the text.
// For example, the simple XML document:
//
//	<a href="http://www.google.com">
//		<img src="http://www.google.com/icon.png" alt="Google" />
//	<br/></a>
//
// results in the following sequence of builder calls:
//
//	StartElement("a", []Attr(Attr("href", "http://www.google.com")));
//	Text("\n\t");
//	StartElement("img", []Attr(Attr("src", "http://www.google.com/icon.png"),
//	                           Attr("alt", "Google")));
//	EndElement("img");
//	Text("\n");
//	StartElement("br", []Attr());
//	EndElement("br");
//	EndElement("a");
//
// There are, of course, a few more details, but the story so far
// should be enough for the majority of uses.  The details are:
//
// * XML documents typically begin with an XML declaration line like
// <?xml version="1.0" encoding="UTF-8"?>.
// This line is strongly recommended, but not strictly required.
// It introduces the XML version and text encoding for the rest
// of the file.  XML parsers are required to recognize UTF-8 and
// UTF-16.  This parser only recognizes UTF-8 (for now?).
//
// * After the XML declaration comes an optional doctype declaration like
// <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
//   "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
// The parser should pass this information on to the client in some
// form, but does not.  It discards such lines.
//
// * The XML declaration line is an instance of a more general tag
// called a processing instruction, XML's #pragma.  The general form is
// <?target text?>, where target is a name (like "xml") specifying
// the intended recipient of the instruction, and text is the
// instruction itself.  This XML parser keeps the <?xml ...?> declaration
// to itself but passes along other processing instructions using
// the ProcInst method.  Processing instructions can appear anywhere
// in an XML document.  Most clients will simply ignore them.
//
// * An XML comment can appear anywhere in an XML document.
// Comments have the form <!--text-->.  The XML parser passes
// them along by calling the Comment method.  Again, most clients
// will simply ignore them.
//
// * Text inside an XML element must be escaped to avoid looking like
// a start/end tag.  Specifically, the characters < and & must be
// written as &lt; and &amp;.  An alternate quoting mechanism is to
// use the construct <![CDATA[...]]>.  The quoted text ... can contain
// < characters, but not the sequence ]]>.  Ampersands must still be
// escaped.  For some reason, the existence of the CDATA quoting mechanism
// infects the processing of ordinary unquoted text, which is not allowed
// to contain the literal sequence ]]>.  Instead, it would be written
// escaped, as in ]]&gt;.  The parser hides all these considerations
// from the library client -- it reports all text, regardless of original
// form and already unescaped, using the Text method.
//
// * A revision to XML 1.0 introduced the concept of name spaces
// for attribute and tag names.  A start tag with an attribute
// xmlns:prefix="URL" introduces `prefix' as a shorthand
// for the name space whose identifier is URL.  Inside the element
// with that start tag, an element name or attribute prefix:foo
// (as in <prefix:foo prefix:bar="baz">) is understood to refer
// to name `foo' in the name space denoted by `URL'.  Although
// this is a shorthand, there is no canonical expansion.  Thus:
//
//	<tag xmlns:foo="http://google.com/foo" xmlns:bar="http://google.com/bar">
//		<foo:red bar:attr="value">text1</foo:red>
//		<bar:red>text2</bar:red>
//	</tag>
//
// and
//
//	<tag xmlns:bar="http://google.com/foo" xmlns:foo="http://google.com/bar">
//		<bar:red foo:attr="value">text1</bar:red>
//		<foo:red>text2</foo:red>
//	</tag>
//
// are equivalent XML documents, and there is no canonical form.
//
// The special attribute xmlns="URL" sets the default name space
// for unprefixed tags (but not attribute names) to URL.
// Thus:
//
//	<tag xmlns="http://google.com/foo" xmlns:bar="http://google.com/bar">
//		<red bar:attr="value">text1</red>
//		<bar:red>text2</bar:red>
//	</tag>
//
// is another XML document equivalent to the first two, and
//
//	<tag xmlns:bar="http://google.com/foo" xmlns="http://google.com/bar">
//		<bar:red attr="value">text1</bar:red>
//		<red>text2</red>
//	</tag>
//
// would be equivalent, except that `attr' in attr="value" has no
// associated name space, in contrast to the previous three where it
// is in the http://google.com/bar name space.
//
// The XML parser hides these details from the client by passing
// a Name struct (ns + name pair) for tag and attribute names.
// Tags and attributes without a name space have ns == "".
//
// References:
//	Annotated XML spec: http://www.xml.com/axml/testaxml.htm
//	XML name spaces: http://www.w3.org/TR/REC-xml-names/

package xml

import (
	"io";
	"os";
)

// XML name, annotated with name space URL
type Name struct {
	ns, name string;
}

// XML attribute (name=value).
type Attr struct {
	name Name;
	value string;
}

// XML Builder - methods client provides to Parser.
// Parser calls methods on builder as it reads and parses XML.
// If a builder method returns an error, the parse stops.
type Builder interface {
	// Called when an element starts.
	// Attr is list of attributes given in the tag.
	//	<name attr.name=attr.value attr1.name=attr1.value ...>
	//	<name attr.name=attr.value attr1.name=attr1.value ... />
	// xmlns and xmlns:foo attributes are handled internally
	// and not passed through to StartElement.
	StartElement(name Name, attr []Attr) os.Error;

	// Called when an element ends.
	//	</name>
	//	<name ... />
	EndElement(name Name) os.Error;

	// Called for non-empty character data string inside element.
	// Can be called multiple times between elements.
	//	text
	//	<![CDATA[text]]>
	Text(text []byte) os.Error;

	// Called when a comment is found in the XML.
	//	<!-- text -->
	Comment(text []byte) os.Error;

	// Called for a processing instruction
	// <?target text?>
	ProcInst(target string, text []byte) os.Error;
}

// Default builder.  Implements no-op Builder methods.
// Embed this in your own Builders to handle the calls
// you don't care about (e.g., Comment, ProcInst).
type BaseBuilder struct {
}

func (b *BaseBuilder) StartElement(name Name, attr []Attr) os.Error {
	return nil;
}

func (b *BaseBuilder) EndElement(name Name) os.Error {
	return nil;
}

func (b *BaseBuilder) Text(text []byte) os.Error {
	return nil;
}

func (b *BaseBuilder) Comment(text []byte) os.Error {
	return nil;
}

func (b *BaseBuilder) ProcInst(target string, text []byte) os.Error {
	return nil;
}

// XML Parser.  Calls Builder methods as it parses.
func Parse(r io.Read, b Builder) os.Error {
	return os.NewError("unimplemented");
}

// Channel interface to XML parser: create a new channel,
// go ParseTokens(r, c), and then read from the channel
// until TokenEnd.  This variant has the benefit that
// the process reading the channel can be a recursive
// function instead of a set of callbacks, but it has the
// drawback that the channel interface cannot signal an
// error to cause the parser to stop early.

// An XML parsing token.
const (
	TokenStartElement = 1 + iota;
	TokenEndElement;
	TokenText;
	TokenComment;
	TokenProcInst;
	TokenEnd;
)

type Token struct {
	Kind int;		// TokenStartElement, TokenEndElement, etc.
	Name Name;		// name (TokenStartElement, TokenEndElement)
	Attr []Attr;		// attributes (TokenStartElement)
	Target string;		// target (TokenProcessingInstruction)
	Text []byte;		// text (TokenCharData, TokenComment, etc.)
	Err os.Error;		// error (TokenEnd)
}

type ChanBuilder chan Token;

func (c ChanBuilder) StartElement(name Name, attr []Attr) os.Error {
	var t Token;
	t.Kind = TokenStartElement;
	t.Name = name;
	t.Attr = attr;
	c <- t;
	return nil;
}

func (c ChanBuilder) EndElement(name Name) os.Error {
	var t Token;
	t.Kind = TokenEndElement;
	t.Name = name;
	c <- t;
	return nil;
}

func (c ChanBuilder) Text(text []byte) os.Error {
	var t Token;
	t.Kind = TokenText;
	t.Text = text;
	c <- t;
	return nil;
}

func (c ChanBuilder) Comment(text []byte) os.Error {
	var t Token;
	t.Kind = TokenComment;
	t.Text = text;
	c <- t;
	return nil;
}

func (c ChanBuilder) ProcInst(target string, text []byte) os.Error {
	var t Token;
	t.Kind = TokenProcInst;
	t.Target = target;
	t.Text = text;
	c <- t;
	return nil;
}

func ParseToChan(r io.Read, c chan Token) {
	var t Token;
	t.Kind = TokenEnd;
	t.Err = Parse(r, ChanBuilder(c));
	c <- t;
}


// scribbled notes based on XML spec.

// document is
//	xml decl?
// 	doctype decl?
//	element
//
// if xml decl is present, must be first.  after that,
// can have comments and procinsts scattered throughout,
// even after the element is done.
//
// xml decl is:
//
// <\?xml version='[a-zA-Z0-9_.:\-]+'( encoding='[A-Za-z][A-Za-z0-9._\-]*')?
//	( standalone='(yes|no)')? ?\?>
//
// spaces denote [ \r\t\n]+.
// written with '' above but can use "" too.
//
// doctype decl might as well be <!DOCTYPE[^>]*>
//
// procinst is <\?name( .*?)\?>.  name cannot be [Xx][Mm][Ll].
//
// comment is <!--(.*?)-->.
//
// tags are:
//	<name( attrib)* ?>	start tag
//	<name( attrib)* ?/>	combined start/end tag
//	</name ?>		end tag
// (the " ?" is an optional space, not a literal question mark.)
//
// plain text is [^<&]* except cannot contain "]]>".
// can also have escaped characters:
//	&#[0-9]+;
//	&#x[0-9A-Fa-f]+;
//	&name;
//
// can use <![CDATA[.*?]]> to avoid escaping < characters.
//
// must rewrite \r and \r\n into \n in text.
//
// names are Unicode.  valid chars listed below.
//
// attrib is name="value" or name='value'.
// can have spaces around =.
// attribute value text is [^<&"]* for appropriate ".
// can also use the &...; escape sequences above.
// cannot use <![CDATA[...]]>.
//
// xmlns attributes are name=value where name has form xmlns:name
// (i.e., xmlns:123 is not okay, because 123 is not a name; xmlns:a123 is ok).
// sub-name must not start with : either.
//
// name is first(second)*.
//
// first is
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
// second is first plus
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

