// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bytes"
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

type toks struct {
	earlyEOF bool
	t        []Token
}

func (t *toks) Token() (Token, error) {
	if len(t.t) == 0 {
		return nil, io.EOF
	}
	var tok Token
	tok, t.t = t.t[0], t.t[1:]
	if t.earlyEOF && len(t.t) == 0 {
		return tok, io.EOF
	}
	return tok, nil
}

func TestDecodeEOF(t *testing.T) {
	start := StartElement{Name: Name{Local: "test"}}
	tests := []struct {
		name   string
		tokens []Token
		ok     bool
	}{
		{
			name: "OK",
			tokens: []Token{
				start,
				start.End(),
			},
			ok: true,
		},
		{
			name: "Malformed",
			tokens: []Token{
				start,
				StartElement{Name: Name{Local: "bad"}},
				start.End(),
			},
			ok: false,
		},
	}
	for _, tc := range tests {
		for _, eof := range []bool{true, false} {
			name := fmt.Sprintf("%s/earlyEOF=%v", tc.name, eof)
			t.Run(name, func(t *testing.T) {
				d := NewTokenDecoder(&toks{
					earlyEOF: eof,
					t:        tc.tokens,
				})
				err := d.Decode(&struct {
					XMLName Name `xml:"test"`
				}{})
				if tc.ok && err != nil {
					t.Fatalf("d.Decode: expected nil error, got %v", err)
				}
				if _, ok := err.(*SyntaxError); !tc.ok && !ok {
					t.Errorf("d.Decode: expected syntax error, got %v", err)
				}
			})
		}
	}
}

type toksNil struct {
	returnEOF bool
	t         []Token
}

func (t *toksNil) Token() (Token, error) {
	if len(t.t) == 0 {
		if !t.returnEOF {
			// Return nil, nil before returning an EOF. It's legal, but
			// discouraged.
			t.returnEOF = true
			return nil, nil
		}
		return nil, io.EOF
	}
	var tok Token
	tok, t.t = t.t[0], t.t[1:]
	return tok, nil
}

func TestDecodeNilToken(t *testing.T) {
	for _, strict := range []bool{true, false} {
		name := fmt.Sprintf("Strict=%v", strict)
		t.Run(name, func(t *testing.T) {
			start := StartElement{Name: Name{Local: "test"}}
			bad := StartElement{Name: Name{Local: "bad"}}
			d := NewTokenDecoder(&toksNil{
				// Malformed
				t: []Token{start, bad, start.End()},
			})
			d.Strict = strict
			err := d.Decode(&struct {
				XMLName Name `xml:"test"`
			}{})
			if _, ok := err.(*SyntaxError); !ok {
				t.Errorf("d.Decode: expected syntax error, got %v", err)
			}
		})
	}
}

const testInput = `
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<body xmlns:foo="ns1" xmlns="ns2" xmlns:tag="ns3" ` +
	"\r\n\t" + `  >
  <hello lang="en">World &lt;&gt;&apos;&quot; &#x767d;&#40300;翔</hello>
  <query>&何; &is-it;</query>
  <goodbye />
  <outer foo:attr="value" xmlns:tag="ns4">
    <inner/>
  </outer>
  <tag:name>
    <![CDATA[Some text here.]]>
  </tag:name>
</body><!-- missing final newline -->`

var testEntity = map[string]string{"何": "What", "is-it": "is it?"}

var rawTokens = []Token{
	CharData("\n"),
	ProcInst{"xml", []byte(`version="1.0" encoding="UTF-8"`)},
	CharData("\n"),
	Directive(`DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"`),
	CharData("\n"),
	StartElement{Name{"", "body"}, []Attr{{Name{"xmlns", "foo"}, "ns1"}, {Name{"", "xmlns"}, "ns2"}, {Name{"xmlns", "tag"}, "ns3"}}},
	CharData("\n  "),
	StartElement{Name{"", "hello"}, []Attr{{Name{"", "lang"}, "en"}}},
	CharData("World <>'\" 白鵬翔"),
	EndElement{Name{"", "hello"}},
	CharData("\n  "),
	StartElement{Name{"", "query"}, []Attr{}},
	CharData("What is it?"),
	EndElement{Name{"", "query"}},
	CharData("\n  "),
	StartElement{Name{"", "goodbye"}, []Attr{}},
	EndElement{Name{"", "goodbye"}},
	CharData("\n  "),
	StartElement{Name{"", "outer"}, []Attr{{Name{"foo", "attr"}, "value"}, {Name{"xmlns", "tag"}, "ns4"}}},
	CharData("\n    "),
	StartElement{Name{"", "inner"}, []Attr{}},
	EndElement{Name{"", "inner"}},
	CharData("\n  "),
	EndElement{Name{"", "outer"}},
	CharData("\n  "),
	StartElement{Name{"tag", "name"}, []Attr{}},
	CharData("\n    "),
	CharData("Some text here."),
	CharData("\n  "),
	EndElement{Name{"tag", "name"}},
	CharData("\n"),
	EndElement{Name{"", "body"}},
	Comment(" missing final newline "),
}

var cookedTokens = []Token{
	CharData("\n"),
	ProcInst{"xml", []byte(`version="1.0" encoding="UTF-8"`)},
	CharData("\n"),
	Directive(`DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"`),
	CharData("\n"),
	StartElement{Name{"ns2", "body"}, []Attr{{Name{"xmlns", "foo"}, "ns1"}, {Name{"", "xmlns"}, "ns2"}, {Name{"xmlns", "tag"}, "ns3"}}},
	CharData("\n  "),
	StartElement{Name{"ns2", "hello"}, []Attr{{Name{"", "lang"}, "en"}}},
	CharData("World <>'\" 白鵬翔"),
	EndElement{Name{"ns2", "hello"}},
	CharData("\n  "),
	StartElement{Name{"ns2", "query"}, []Attr{}},
	CharData("What is it?"),
	EndElement{Name{"ns2", "query"}},
	CharData("\n  "),
	StartElement{Name{"ns2", "goodbye"}, []Attr{}},
	EndElement{Name{"ns2", "goodbye"}},
	CharData("\n  "),
	StartElement{Name{"ns2", "outer"}, []Attr{{Name{"ns1", "attr"}, "value"}, {Name{"xmlns", "tag"}, "ns4"}}},
	CharData("\n    "),
	StartElement{Name{"ns2", "inner"}, []Attr{}},
	EndElement{Name{"ns2", "inner"}},
	CharData("\n  "),
	EndElement{Name{"ns2", "outer"}},
	CharData("\n  "),
	StartElement{Name{"ns3", "name"}, []Attr{}},
	CharData("\n    "),
	CharData("Some text here."),
	CharData("\n  "),
	EndElement{Name{"ns3", "name"}},
	CharData("\n"),
	EndElement{Name{"ns2", "body"}},
	Comment(" missing final newline "),
}

const testInputAltEncoding = `
<?xml version="1.0" encoding="x-testing-uppercase"?>
<TAG>VALUE</TAG>`

var rawTokensAltEncoding = []Token{
	CharData("\n"),
	ProcInst{"xml", []byte(`version="1.0" encoding="x-testing-uppercase"`)},
	CharData("\n"),
	StartElement{Name{"", "tag"}, []Attr{}},
	CharData("value"),
	EndElement{Name{"", "tag"}},
}

var xmlInput = []string{
	// unexpected EOF cases
	"<",
	"<t",
	"<t ",
	"<t/",
	"<!",
	"<!-",
	"<!--",
	"<!--c-",
	"<!--c--",
	"<!d",
	"<t></",
	"<t></t",
	"<?",
	"<?p",
	"<t a",
	"<t a=",
	"<t a='",
	"<t a=''",
	"<t/><![",
	"<t/><![C",
	"<t/><![CDATA[d",
	"<t/><![CDATA[d]",
	"<t/><![CDATA[d]]",

	// other Syntax errors
	"<>",
	"<t/a",
	"<0 />",
	"<?0 >",
	//	"<!0 >",	// let the Token() caller handle
	"</0>",
	"<t 0=''>",
	"<t a='&'>",
	"<t a='<'>",
	"<t>&nbspc;</t>",
	"<t a>",
	"<t a=>",
	"<t a=v>",
	//	"<![CDATA[d]]>",	// let the Token() caller handle
	"<t></e>",
	"<t></>",
	"<t></t!",
	"<t>cdata]]></t>",
}

func TestRawToken(t *testing.T) {
	d := NewDecoder(strings.NewReader(testInput))
	d.Entity = testEntity
	testRawToken(t, d, testInput, rawTokens)
}

const nonStrictInput = `
<tag>non&entity</tag>
<tag>&unknown;entity</tag>
<tag>&#123</tag>
<tag>&#zzz;</tag>
<tag>&なまえ3;</tag>
<tag>&lt-gt;</tag>
<tag>&;</tag>
<tag>&0a;</tag>
`

var nonStrictTokens = []Token{
	CharData("\n"),
	StartElement{Name{"", "tag"}, []Attr{}},
	CharData("non&entity"),
	EndElement{Name{"", "tag"}},
	CharData("\n"),
	StartElement{Name{"", "tag"}, []Attr{}},
	CharData("&unknown;entity"),
	EndElement{Name{"", "tag"}},
	CharData("\n"),
	StartElement{Name{"", "tag"}, []Attr{}},
	CharData("&#123"),
	EndElement{Name{"", "tag"}},
	CharData("\n"),
	StartElement{Name{"", "tag"}, []Attr{}},
	CharData("&#zzz;"),
	EndElement{Name{"", "tag"}},
	CharData("\n"),
	StartElement{Name{"", "tag"}, []Attr{}},
	CharData("&なまえ3;"),
	EndElement{Name{"", "tag"}},
	CharData("\n"),
	StartElement{Name{"", "tag"}, []Attr{}},
	CharData("&lt-gt;"),
	EndElement{Name{"", "tag"}},
	CharData("\n"),
	StartElement{Name{"", "tag"}, []Attr{}},
	CharData("&;"),
	EndElement{Name{"", "tag"}},
	CharData("\n"),
	StartElement{Name{"", "tag"}, []Attr{}},
	CharData("&0a;"),
	EndElement{Name{"", "tag"}},
	CharData("\n"),
}

func TestNonStrictRawToken(t *testing.T) {
	d := NewDecoder(strings.NewReader(nonStrictInput))
	d.Strict = false
	testRawToken(t, d, nonStrictInput, nonStrictTokens)
}

type downCaser struct {
	t *testing.T
	r io.ByteReader
}

func (d *downCaser) ReadByte() (c byte, err error) {
	c, err = d.r.ReadByte()
	if c >= 'A' && c <= 'Z' {
		c += 'a' - 'A'
	}
	return
}

func (d *downCaser) Read(p []byte) (int, error) {
	d.t.Fatalf("unexpected Read call on downCaser reader")
	panic("unreachable")
}

func TestRawTokenAltEncoding(t *testing.T) {
	d := NewDecoder(strings.NewReader(testInputAltEncoding))
	d.CharsetReader = func(charset string, input io.Reader) (io.Reader, error) {
		if charset != "x-testing-uppercase" {
			t.Fatalf("unexpected charset %q", charset)
		}
		return &downCaser{t, input.(io.ByteReader)}, nil
	}
	testRawToken(t, d, testInputAltEncoding, rawTokensAltEncoding)
}

func TestRawTokenAltEncodingNoConverter(t *testing.T) {
	d := NewDecoder(strings.NewReader(testInputAltEncoding))
	token, err := d.RawToken()
	if token == nil {
		t.Fatalf("expected a token on first RawToken call")
	}
	if err != nil {
		t.Fatal(err)
	}
	token, err = d.RawToken()
	if token != nil {
		t.Errorf("expected a nil token; got %#v", token)
	}
	if err == nil {
		t.Fatalf("expected an error on second RawToken call")
	}
	const encoding = "x-testing-uppercase"
	if !strings.Contains(err.Error(), encoding) {
		t.Errorf("expected error to contain %q; got error: %v",
			encoding, err)
	}
}

func testRawToken(t *testing.T, d *Decoder, raw string, rawTokens []Token) {
	lastEnd := int64(0)
	for i, want := range rawTokens {
		start := d.InputOffset()
		have, err := d.RawToken()
		end := d.InputOffset()
		if err != nil {
			t.Fatalf("token %d: unexpected error: %s", i, err)
		}
		if !reflect.DeepEqual(have, want) {
			var shave, swant string
			if _, ok := have.(CharData); ok {
				shave = fmt.Sprintf("CharData(%q)", have)
			} else {
				shave = fmt.Sprintf("%#v", have)
			}
			if _, ok := want.(CharData); ok {
				swant = fmt.Sprintf("CharData(%q)", want)
			} else {
				swant = fmt.Sprintf("%#v", want)
			}
			t.Errorf("token %d = %s, want %s", i, shave, swant)
		}

		// Check that InputOffset returned actual token.
		switch {
		case start < lastEnd:
			t.Errorf("token %d: position [%d,%d) for %T is before previous token", i, start, end, have)
		case start >= end:
			// Special case: EndElement can be synthesized.
			if start == end && end == lastEnd {
				break
			}
			t.Errorf("token %d: position [%d,%d) for %T is empty", i, start, end, have)
		case end > int64(len(raw)):
			t.Errorf("token %d: position [%d,%d) for %T extends beyond input", i, start, end, have)
		default:
			text := raw[start:end]
			if strings.ContainsAny(text, "<>") && (!strings.HasPrefix(text, "<") || !strings.HasSuffix(text, ">")) {
				t.Errorf("token %d: misaligned raw token %#q for %T", i, text, have)
			}
		}
		lastEnd = end
	}
}

// Ensure that directives (specifically !DOCTYPE) include the complete
// text of any nested directives, noting that < and > do not change
// nesting depth if they are in single or double quotes.

var nestedDirectivesInput = `
<!DOCTYPE [<!ENTITY rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#">]>
<!DOCTYPE [<!ENTITY xlt ">">]>
<!DOCTYPE [<!ENTITY xlt "<">]>
<!DOCTYPE [<!ENTITY xlt '>'>]>
<!DOCTYPE [<!ENTITY xlt '<'>]>
<!DOCTYPE [<!ENTITY xlt '">'>]>
<!DOCTYPE [<!ENTITY xlt "'<">]>
`

var nestedDirectivesTokens = []Token{
	CharData("\n"),
	Directive(`DOCTYPE [<!ENTITY rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#">]`),
	CharData("\n"),
	Directive(`DOCTYPE [<!ENTITY xlt ">">]`),
	CharData("\n"),
	Directive(`DOCTYPE [<!ENTITY xlt "<">]`),
	CharData("\n"),
	Directive(`DOCTYPE [<!ENTITY xlt '>'>]`),
	CharData("\n"),
	Directive(`DOCTYPE [<!ENTITY xlt '<'>]`),
	CharData("\n"),
	Directive(`DOCTYPE [<!ENTITY xlt '">'>]`),
	CharData("\n"),
	Directive(`DOCTYPE [<!ENTITY xlt "'<">]`),
	CharData("\n"),
}

func TestNestedDirectives(t *testing.T) {
	d := NewDecoder(strings.NewReader(nestedDirectivesInput))

	for i, want := range nestedDirectivesTokens {
		have, err := d.Token()
		if err != nil {
			t.Fatalf("token %d: unexpected error: %s", i, err)
		}
		if !reflect.DeepEqual(have, want) {
			t.Errorf("token %d = %#v want %#v", i, have, want)
		}
	}
}

func TestToken(t *testing.T) {
	d := NewDecoder(strings.NewReader(testInput))
	d.Entity = testEntity

	for i, want := range cookedTokens {
		have, err := d.Token()
		if err != nil {
			t.Fatalf("token %d: unexpected error: %s", i, err)
		}
		if !reflect.DeepEqual(have, want) {
			t.Errorf("token %d = %#v want %#v", i, have, want)
		}
	}
}

func TestSyntax(t *testing.T) {
	for i := range xmlInput {
		d := NewDecoder(strings.NewReader(xmlInput[i]))
		var err error
		for _, err = d.Token(); err == nil; _, err = d.Token() {
		}
		if _, ok := err.(*SyntaxError); !ok {
			t.Fatalf(`xmlInput "%s": expected SyntaxError not received`, xmlInput[i])
		}
	}
}

func TestInputLinePos(t *testing.T) {
	testInput := `<root>
<?pi
 ?>  <elt
att
=
"val">
<![CDATA[
]]><!--

--></elt>
</root>`
	linePos := [][]int{
		{1, 7},
		{2, 1},
		{3, 4},
		{3, 6},
		{6, 7},
		{7, 1},
		{8, 4},
		{10, 4},
		{10, 10},
		{11, 1},
		{11, 8},
	}
	dec := NewDecoder(strings.NewReader(testInput))
	for _, want := range linePos {
		if _, err := dec.Token(); err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}

		gotLine, gotCol := dec.InputPos()
		if gotLine != want[0] || gotCol != want[1] {
			t.Errorf("dec.InputPos() = %d,%d, want %d,%d", gotLine, gotCol, want[0], want[1])
		}
	}
}

type allScalars struct {
	True1     bool
	True2     bool
	False1    bool
	False2    bool
	Int       int
	Int8      int8
	Int16     int16
	Int32     int32
	Int64     int64
	Uint      int
	Uint8     uint8
	Uint16    uint16
	Uint32    uint32
	Uint64    uint64
	Uintptr   uintptr
	Float32   float32
	Float64   float64
	String    string
	PtrString *string
}

var all = allScalars{
	True1:     true,
	True2:     true,
	False1:    false,
	False2:    false,
	Int:       1,
	Int8:      -2,
	Int16:     3,
	Int32:     -4,
	Int64:     5,
	Uint:      6,
	Uint8:     7,
	Uint16:    8,
	Uint32:    9,
	Uint64:    10,
	Uintptr:   11,
	Float32:   13.0,
	Float64:   14.0,
	String:    "15",
	PtrString: &sixteen,
}

var sixteen = "16"

const testScalarsInput = `<allscalars>
	<True1>true</True1>
	<True2>1</True2>
	<False1>false</False1>
	<False2>0</False2>
	<Int>1</Int>
	<Int8>-2</Int8>
	<Int16>3</Int16>
	<Int32>-4</Int32>
	<Int64>5</Int64>
	<Uint>6</Uint>
	<Uint8>7</Uint8>
	<Uint16>8</Uint16>
	<Uint32>9</Uint32>
	<Uint64>10</Uint64>
	<Uintptr>11</Uintptr>
	<Float>12.0</Float>
	<Float32>13.0</Float32>
	<Float64>14.0</Float64>
	<String>15</String>
	<PtrString>16</PtrString>
</allscalars>`

func TestAllScalars(t *testing.T) {
	var a allScalars
	err := Unmarshal([]byte(testScalarsInput), &a)

	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, all) {
		t.Errorf("have %+v want %+v", a, all)
	}
}

type item struct {
	FieldA string
}

func TestIssue68387(t *testing.T) {
	data := `<item b=']]>'/>`
	dec := NewDecoder(strings.NewReader(data))
	var tok1, tok2, tok3 Token
	var err error
	if tok1, err = dec.RawToken(); err != nil {
		t.Fatalf("RawToken() failed: %v", err)
	}
	if tok2, err = dec.RawToken(); err != nil {
		t.Fatalf("RawToken() failed: %v", err)
	}
	if tok3, err = dec.RawToken(); err != io.EOF || tok3 != nil {
		t.Fatalf("Missed EOF")
	}
	s := StartElement{Name{"", "item"}, []Attr{Attr{Name{"", "b"}, "]]>"}}}
	if !reflect.DeepEqual(tok1.(StartElement), s) {
		t.Error("Wrong start element")
	}
	e := EndElement{Name{"", "item"}}
	if tok2.(EndElement) != e {
		t.Error("Wrong end element")
	}
}

func TestIssue569(t *testing.T) {
	data := `<item><FieldA>abcd</FieldA></item>`
	var i item
	err := Unmarshal([]byte(data), &i)

	if err != nil || i.FieldA != "abcd" {
		t.Fatal("Expecting abcd")
	}
}

func TestUnquotedAttrs(t *testing.T) {
	data := "<tag attr=azAZ09:-_\t>"
	d := NewDecoder(strings.NewReader(data))
	d.Strict = false
	token, err := d.Token()
	if _, ok := err.(*SyntaxError); ok {
		t.Errorf("Unexpected error: %v", err)
	}
	if token.(StartElement).Name.Local != "tag" {
		t.Errorf("Unexpected tag name: %v", token.(StartElement).Name.Local)
	}
	attr := token.(StartElement).Attr[0]
	if attr.Value != "azAZ09:-_" {
		t.Errorf("Unexpected attribute value: %v", attr.Value)
	}
	if attr.Name.Local != "attr" {
		t.Errorf("Unexpected attribute name: %v", attr.Name.Local)
	}
}

func TestValuelessAttrs(t *testing.T) {
	tests := [][3]string{
		{"<p nowrap>", "p", "nowrap"},
		{"<p nowrap >", "p", "nowrap"},
		{"<input checked/>", "input", "checked"},
		{"<input checked />", "input", "checked"},
	}
	for _, test := range tests {
		d := NewDecoder(strings.NewReader(test[0]))
		d.Strict = false
		token, err := d.Token()
		if _, ok := err.(*SyntaxError); ok {
			t.Errorf("Unexpected error: %v", err)
		}
		if token.(StartElement).Name.Local != test[1] {
			t.Errorf("Unexpected tag name: %v", token.(StartElement).Name.Local)
		}
		attr := token.(StartElement).Attr[0]
		if attr.Value != test[2] {
			t.Errorf("Unexpected attribute value: %v", attr.Value)
		}
		if attr.Name.Local != test[2] {
			t.Errorf("Unexpected attribute name: %v", attr.Name.Local)
		}
	}
}

func TestCopyTokenCharData(t *testing.T) {
	data := []byte("same data")
	var tok1 Token = CharData(data)
	tok2 := CopyToken(tok1)
	if !reflect.DeepEqual(tok1, tok2) {
		t.Error("CopyToken(CharData) != CharData")
	}
	data[1] = 'o'
	if reflect.DeepEqual(tok1, tok2) {
		t.Error("CopyToken(CharData) uses same buffer.")
	}
}

func TestCopyTokenStartElement(t *testing.T) {
	elt := StartElement{Name{"", "hello"}, []Attr{{Name{"", "lang"}, "en"}}}
	var tok1 Token = elt
	tok2 := CopyToken(tok1)
	if tok1.(StartElement).Attr[0].Value != "en" {
		t.Error("CopyToken overwrote Attr[0]")
	}
	if !reflect.DeepEqual(tok1, tok2) {
		t.Error("CopyToken(StartElement) != StartElement")
	}
	tok1.(StartElement).Attr[0] = Attr{Name{"", "lang"}, "de"}
	if reflect.DeepEqual(tok1, tok2) {
		t.Error("CopyToken(CharData) uses same buffer.")
	}
}

func TestCopyTokenComment(t *testing.T) {
	data := []byte("<!-- some comment -->")
	var tok1 Token = Comment(data)
	tok2 := CopyToken(tok1)
	if !reflect.DeepEqual(tok1, tok2) {
		t.Error("CopyToken(Comment) != Comment")
	}
	data[1] = 'o'
	if reflect.DeepEqual(tok1, tok2) {
		t.Error("CopyToken(Comment) uses same buffer.")
	}
}

func TestSyntaxErrorLineNum(t *testing.T) {
	testInput := "<P>Foo<P>\n\n<P>Bar</>\n"
	d := NewDecoder(strings.NewReader(testInput))
	var err error
	for _, err = d.Token(); err == nil; _, err = d.Token() {
	}
	synerr, ok := err.(*SyntaxError)
	if !ok {
		t.Error("Expected SyntaxError.")
	}
	if synerr.Line != 3 {
		t.Error("SyntaxError didn't have correct line number.")
	}
}

func TestTrailingRawToken(t *testing.T) {
	input := `<FOO></FOO>  `
	d := NewDecoder(strings.NewReader(input))
	var err error
	for _, err = d.RawToken(); err == nil; _, err = d.RawToken() {
	}
	if err != io.EOF {
		t.Fatalf("d.RawToken() = _, %v, want _, io.EOF", err)
	}
}

func TestTrailingToken(t *testing.T) {
	input := `<FOO></FOO>  `
	d := NewDecoder(strings.NewReader(input))
	var err error
	for _, err = d.Token(); err == nil; _, err = d.Token() {
	}
	if err != io.EOF {
		t.Fatalf("d.Token() = _, %v, want _, io.EOF", err)
	}
}

func TestEntityInsideCDATA(t *testing.T) {
	input := `<test><![CDATA[ &val=foo ]]></test>`
	d := NewDecoder(strings.NewReader(input))
	var err error
	for _, err = d.Token(); err == nil; _, err = d.Token() {
	}
	if err != io.EOF {
		t.Fatalf("d.Token() = _, %v, want _, io.EOF", err)
	}
}

var characterTests = []struct {
	in  string
	err string
}{
	{"\x12<doc/>", "illegal character code U+0012"},
	{"<?xml version=\"1.0\"?>\x0b<doc/>", "illegal character code U+000B"},
	{"\xef\xbf\xbe<doc/>", "illegal character code U+FFFE"},
	{"<?xml version=\"1.0\"?><doc>\r\n<hiya/>\x07<toots/></doc>", "illegal character code U+0007"},
	{"<?xml version=\"1.0\"?><doc \x12='value'>what's up</doc>", "expected attribute name in element"},
	{"<doc>&abc\x01;</doc>", "invalid character entity &abc (no semicolon)"},
	{"<doc>&\x01;</doc>", "invalid character entity & (no semicolon)"},
	{"<doc>&\xef\xbf\xbe;</doc>", "invalid character entity &\uFFFE;"},
	{"<doc>&hello;</doc>", "invalid character entity &hello;"},
}

func TestDisallowedCharacters(t *testing.T) {

	for i, tt := range characterTests {
		d := NewDecoder(strings.NewReader(tt.in))
		var err error

		for err == nil {
			_, err = d.Token()
		}
		synerr, ok := err.(*SyntaxError)
		if !ok {
			t.Fatalf("input %d d.Token() = _, %v, want _, *SyntaxError", i, err)
		}
		if synerr.Msg != tt.err {
			t.Fatalf("input %d synerr.Msg wrong: want %q, got %q", i, tt.err, synerr.Msg)
		}
	}
}

func TestIsInCharacterRange(t *testing.T) {
	invalid := []rune{
		utf8.MaxRune + 1,
		0xD800, // surrogate min
		0xDFFF, // surrogate max
		-1,
	}
	for _, r := range invalid {
		if isInCharacterRange(r) {
			t.Errorf("rune %U considered valid", r)
		}
	}
}

var procInstTests = []struct {
	input  string
	expect [2]string
}{
	{`version="1.0" encoding="utf-8"`, [2]string{"1.0", "utf-8"}},
	{`version="1.0" encoding='utf-8'`, [2]string{"1.0", "utf-8"}},
	{`version="1.0" encoding='utf-8' `, [2]string{"1.0", "utf-8"}},
	{`version="1.0" encoding=utf-8`, [2]string{"1.0", ""}},
	{`encoding="FOO" `, [2]string{"", "FOO"}},
	{`version=2.0 version="1.0" encoding=utf-7 encoding='utf-8'`, [2]string{"1.0", "utf-8"}},
	{`version= encoding=`, [2]string{"", ""}},
	{`encoding="version=1.0"`, [2]string{"", "version=1.0"}},
	{``, [2]string{"", ""}},
	// TODO: what's the right approach to handle these nested cases?
	{`encoding="version='1.0'"`, [2]string{"1.0", "version='1.0'"}},
	{`version="encoding='utf-8'"`, [2]string{"encoding='utf-8'", "utf-8"}},
}

func TestProcInstEncoding(t *testing.T) {
	for _, test := range procInstTests {
		if got := procInst("version", test.input); got != test.expect[0] {
			t.Errorf("procInst(version, %q) = %q; want %q", test.input, got, test.expect[0])
		}
		if got := procInst("encoding", test.input); got != test.expect[1] {
			t.Errorf("procInst(encoding, %q) = %q; want %q", test.input, got, test.expect[1])
		}
	}
}

// Ensure that directives with comments include the complete
// text of any nested directives.

var directivesWithCommentsInput = `
<!DOCTYPE [<!-- a comment --><!ENTITY rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#">]>
<!DOCTYPE [<!ENTITY go "Golang"><!-- a comment-->]>
<!DOCTYPE <!-> <!> <!----> <!-->--> <!--->--> [<!ENTITY go "Golang"><!-- a comment-->]>
`

var directivesWithCommentsTokens = []Token{
	CharData("\n"),
	Directive(`DOCTYPE [ <!ENTITY rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#">]`),
	CharData("\n"),
	Directive(`DOCTYPE [<!ENTITY go "Golang"> ]`),
	CharData("\n"),
	Directive(`DOCTYPE <!-> <!>       [<!ENTITY go "Golang"> ]`),
	CharData("\n"),
}

func TestDirectivesWithComments(t *testing.T) {
	d := NewDecoder(strings.NewReader(directivesWithCommentsInput))

	for i, want := range directivesWithCommentsTokens {
		have, err := d.Token()
		if err != nil {
			t.Fatalf("token %d: unexpected error: %s", i, err)
		}
		if !reflect.DeepEqual(have, want) {
			t.Errorf("token %d = %#v want %#v", i, have, want)
		}
	}
}

// Writer whose Write method always returns an error.
type errWriter struct{}

func (errWriter) Write(p []byte) (n int, err error) { return 0, fmt.Errorf("unwritable") }

func TestEscapeTextIOErrors(t *testing.T) {
	expectErr := "unwritable"
	err := EscapeText(errWriter{}, []byte{'A'})

	if err == nil || err.Error() != expectErr {
		t.Errorf("have %v, want %v", err, expectErr)
	}
}

func TestEscapeTextInvalidChar(t *testing.T) {
	input := []byte("A \x00 terminated string.")
	expected := "A \uFFFD terminated string."

	buff := new(strings.Builder)
	if err := EscapeText(buff, input); err != nil {
		t.Fatalf("have %v, want nil", err)
	}
	text := buff.String()

	if text != expected {
		t.Errorf("have %v, want %v", text, expected)
	}
}

func TestIssue5880(t *testing.T) {
	type T []byte
	data, err := Marshal(T{192, 168, 0, 1})
	if err != nil {
		t.Errorf("Marshal error: %v", err)
	}
	if !utf8.Valid(data) {
		t.Errorf("Marshal generated invalid UTF-8: %x", data)
	}
}

func TestIssue8535(t *testing.T) {

	type ExampleConflict struct {
		XMLName  Name   `xml:"example"`
		Link     string `xml:"link"`
		AtomLink string `xml:"http://www.w3.org/2005/Atom link"` // Same name in a different name space
	}
	testCase := `<example>
			<title>Example</title>
			<link>http://example.com/default</link> <!-- not assigned -->
			<link>http://example.com/home</link> <!-- not assigned -->
			<ns:link xmlns:ns="http://www.w3.org/2005/Atom">http://example.com/ns</ns:link>
		</example>`

	var dest ExampleConflict
	d := NewDecoder(strings.NewReader(testCase))
	if err := d.Decode(&dest); err != nil {
		t.Fatal(err)
	}
}

func TestEncodeXMLNS(t *testing.T) {
	testCases := []struct {
		f    func() ([]byte, error)
		want string
		ok   bool
	}{
		{encodeXMLNS1, `<Test xmlns="http://example.com/ns"><Body>hello world</Body></Test>`, true},
		{encodeXMLNS2, `<Test><body xmlns="http://example.com/ns">hello world</body></Test>`, true},
		{encodeXMLNS3, `<Test xmlns="http://example.com/ns"><Body>hello world</Body></Test>`, true},
		{encodeXMLNS4, `<Test xmlns="http://example.com/ns"><Body>hello world</Body></Test>`, false},
	}

	for i, tc := range testCases {
		if b, err := tc.f(); err == nil {
			if got, want := string(b), tc.want; got != want {
				t.Errorf("%d: got %s, want %s \n", i, got, want)
			}
		} else {
			t.Errorf("%d: marshal failed with %s", i, err)
		}
	}
}

func encodeXMLNS1() ([]byte, error) {

	type T struct {
		XMLName Name   `xml:"Test"`
		Ns      string `xml:"xmlns,attr"`
		Body    string
	}

	s := &T{Ns: "http://example.com/ns", Body: "hello world"}
	return Marshal(s)
}

func encodeXMLNS2() ([]byte, error) {

	type Test struct {
		Body string `xml:"http://example.com/ns body"`
	}

	s := &Test{Body: "hello world"}
	return Marshal(s)
}

func encodeXMLNS3() ([]byte, error) {

	type Test struct {
		XMLName Name `xml:"http://example.com/ns Test"`
		Body    string
	}

	//s := &Test{XMLName: Name{"http://example.com/ns",""}, Body: "hello world"} is unusable as the "-" is missing
	// as documentation states
	s := &Test{Body: "hello world"}
	return Marshal(s)
}

func encodeXMLNS4() ([]byte, error) {

	type Test struct {
		Ns   string `xml:"xmlns,attr"`
		Body string
	}

	s := &Test{Ns: "http://example.com/ns", Body: "hello world"}
	return Marshal(s)
}

func TestIssue11405(t *testing.T) {
	testCases := []string{
		"<root>",
		"<root><foo>",
		"<root><foo></foo>",
	}
	for _, tc := range testCases {
		d := NewDecoder(strings.NewReader(tc))
		var err error
		for {
			_, err = d.Token()
			if err != nil {
				break
			}
		}
		if _, ok := err.(*SyntaxError); !ok {
			t.Errorf("%s: Token: Got error %v, want SyntaxError", tc, err)
		}
	}
}

func TestIssue12417(t *testing.T) {
	testCases := []struct {
		s  string
		ok bool
	}{
		{`<?xml encoding="UtF-8" version="1.0"?><root/>`, true},
		{`<?xml encoding="UTF-8" version="1.0"?><root/>`, true},
		{`<?xml encoding="utf-8" version="1.0"?><root/>`, true},
		{`<?xml encoding="uuu-9" version="1.0"?><root/>`, false},
	}
	for _, tc := range testCases {
		d := NewDecoder(strings.NewReader(tc.s))
		var err error
		for {
			_, err = d.Token()
			if err != nil {
				if err == io.EOF {
					err = nil
				}
				break
			}
		}
		if err != nil && tc.ok {
			t.Errorf("%q: Encoding charset: expected no error, got %s", tc.s, err)
			continue
		}
		if err == nil && !tc.ok {
			t.Errorf("%q: Encoding charset: expected error, got nil", tc.s)
		}
	}
}

func TestIssue7113(t *testing.T) {
	type C struct {
		XMLName Name `xml:""` // Sets empty namespace
	}

	type D struct {
		XMLName Name `xml:"d"`
	}

	type A struct {
		XMLName Name `xml:""`
		C       C    `xml:""`
		D       D
	}

	var a A
	structSpace := "b"
	xmlTest := `<A xmlns="` + structSpace + `"><C xmlns=""></C><d></d></A>`
	t.Log(xmlTest)
	err := Unmarshal([]byte(xmlTest), &a)
	if err != nil {
		t.Fatal(err)
	}

	if a.XMLName.Space != structSpace {
		t.Errorf("overidding with empty namespace: unmarshaling, got %s, want %s\n", a.XMLName.Space, structSpace)
	}
	if len(a.C.XMLName.Space) != 0 {
		t.Fatalf("overidding with empty namespace: unmarshaling, got %s, want empty\n", a.C.XMLName.Space)
	}

	var b []byte
	b, err = Marshal(&a)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.C.XMLName.Space) != 0 {
		t.Errorf("overidding with empty namespace: marshaling, got %s in C tag which should be empty\n", a.C.XMLName.Space)
	}
	if string(b) != xmlTest {
		t.Fatalf("overidding with empty namespace: marshaling, got %s, want %s\n", b, xmlTest)
	}
	var c A
	err = Unmarshal(b, &c)
	if err != nil {
		t.Fatalf("second Unmarshal failed: %s", err)
	}
	if c.XMLName.Space != "b" {
		t.Errorf("overidding with empty namespace: after marshaling & unmarshaling, XML name space: got %s, want %s\n", a.XMLName.Space, structSpace)
	}
	if len(c.C.XMLName.Space) != 0 {
		t.Errorf("overidding with empty namespace: after marshaling & unmarshaling, got %s, want empty\n", a.C.XMLName.Space)
	}
}

func TestIssue20396(t *testing.T) {

	var attrError = UnmarshalError("XML syntax error on line 1: expected attribute name in element")

	testCases := []struct {
		s       string
		wantErr error
	}{
		{`<a:te:st xmlns:a="abcd"/>`, // Issue 20396
			UnmarshalError("XML syntax error on line 1: expected element name after <")},
		{`<a:te=st xmlns:a="abcd"/>`, attrError},
		{`<a:te&st xmlns:a="abcd"/>`, attrError},
		{`<a:test xmlns:a="abcd"/>`, nil},
		{`<a:te:st xmlns:a="abcd">1</a:te:st>`,
			UnmarshalError("XML syntax error on line 1: expected element name after <")},
		{`<a:te=st xmlns:a="abcd">1</a:te=st>`, attrError},
		{`<a:te&st xmlns:a="abcd">1</a:te&st>`, attrError},
		{`<a:test xmlns:a="abcd">1</a:test>`, nil},
	}

	var dest string
	for _, tc := range testCases {
		if got, want := Unmarshal([]byte(tc.s), &dest), tc.wantErr; got != want {
			if got == nil {
				t.Errorf("%s: Unexpected success, want %v", tc.s, want)
			} else if want == nil {
				t.Errorf("%s: Unexpected error, got %v", tc.s, got)
			} else if got.Error() != want.Error() {
				t.Errorf("%s: got %v, want %v", tc.s, got, want)
			}
		}
	}
}

func TestIssue20685(t *testing.T) {
	testCases := []struct {
		s  string
		ok bool
	}{
		{`<x:book xmlns:x="abcd" xmlns:y="abcd"><unclosetag>one</x:book>`, false},
		{`<x:book xmlns:x="abcd" xmlns:y="abcd">one</x:book>`, true},
		{`<x:book xmlns:x="abcd" xmlns:y="abcd">one</y:book>`, false},
		{`<x:book xmlns:y="abcd" xmlns:x="abcd">one</y:book>`, false},
		{`<x:book xmlns:x="abcd">one</y:book>`, false},
		{`<x:book>one</y:book>`, false},
		{`<xbook>one</ybook>`, false},
	}
	for _, tc := range testCases {
		d := NewDecoder(strings.NewReader(tc.s))
		var err error
		for {
			_, err = d.Token()
			if err != nil {
				if err == io.EOF {
					err = nil
				}
				break
			}
		}
		if err != nil && tc.ok {
			t.Errorf("%q: Closing tag with namespace : expected no error, got %s", tc.s, err)
			continue
		}
		if err == nil && !tc.ok {
			t.Errorf("%q: Closing tag with namespace : expected error, got nil", tc.s)
		}
	}
}

func tokenMap(mapping func(t Token) Token) func(TokenReader) TokenReader {
	return func(src TokenReader) TokenReader {
		return mapper{
			t: src,
			f: mapping,
		}
	}
}

type mapper struct {
	t TokenReader
	f func(Token) Token
}

func (m mapper) Token() (Token, error) {
	tok, err := m.t.Token()
	if err != nil {
		return nil, err
	}
	return m.f(tok), nil
}

func TestNewTokenDecoderIdempotent(t *testing.T) {
	d := NewDecoder(strings.NewReader(`<br>`))
	d2 := NewTokenDecoder(d)
	if d != d2 {
		t.Error("NewTokenDecoder did not detect underlying Decoder")
	}
}

func TestWrapDecoder(t *testing.T) {
	d := NewDecoder(strings.NewReader(`<quote>[Re-enter Clown with a letter, and FABIAN]</quote>`))
	m := tokenMap(func(t Token) Token {
		switch tok := t.(type) {
		case StartElement:
			if tok.Name.Local == "quote" {
				tok.Name.Local = "blocking"
				return tok
			}
		case EndElement:
			if tok.Name.Local == "quote" {
				tok.Name.Local = "blocking"
				return tok
			}
		}
		return t
	})

	d = NewTokenDecoder(m(d))

	o := struct {
		XMLName  Name   `xml:"blocking"`
		Chardata string `xml:",chardata"`
	}{}

	if err := d.Decode(&o); err != nil {
		t.Fatal("Got unexpected error while decoding:", err)
	}

	if o.Chardata != "[Re-enter Clown with a letter, and FABIAN]" {
		t.Fatalf("Got unexpected chardata: `%s`\n", o.Chardata)
	}
}

type tokReader struct{}

func (tokReader) Token() (Token, error) {
	return StartElement{}, nil
}

type Failure struct{}

func (Failure) UnmarshalXML(*Decoder, StartElement) error {
	return nil
}

func TestTokenUnmarshaler(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Error("Unexpected panic using custom token unmarshaler")
		}
	}()

	d := NewTokenDecoder(tokReader{})
	d.Decode(&Failure{})
}

func testRoundTrip(t *testing.T, input string) {
	d := NewDecoder(strings.NewReader(input))
	var tokens []Token
	var buf bytes.Buffer
	e := NewEncoder(&buf)
	for {
		tok, err := d.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("invalid input: %v", err)
		}
		if err := e.EncodeToken(tok); err != nil {
			t.Fatalf("failed to re-encode input: %v", err)
		}
		tokens = append(tokens, CopyToken(tok))
	}
	if err := e.Flush(); err != nil {
		t.Fatal(err)
	}

	d = NewDecoder(&buf)
	for {
		tok, err := d.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("failed to decode output: %v", err)
		}
		if len(tokens) == 0 {
			t.Fatalf("unexpected token: %#v", tok)
		}
		a, b := tokens[0], tok
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("token mismatch: %#v vs %#v", a, b)
		}
		tokens = tokens[1:]
	}
	if len(tokens) > 0 {
		t.Fatalf("lost tokens: %#v", tokens)
	}
}

func TestRoundTrip(t *testing.T) {
	tests := map[string]string{
		"trailing colon":         `<foo abc:="x"></foo>`,
		"comments in directives": `<!ENTITY x<!<!-- c1 [ " -->--x --> > <e></e> <!DOCTYPE xxx [ x<!-- c2 " -->--x ]>`,
	}
	for name, input := range tests {
		t.Run(name, func(t *testing.T) { testRoundTrip(t, input) })
	}
}

func TestParseErrors(t *testing.T) {
	withDefaultHeader := func(s string) string {
		return `<?xml version="1.0" encoding="UTF-8"?>` + s
	}
	tests := []struct {
		src string
		err string
	}{
		{withDefaultHeader(`</foo>`), `unexpected end element </foo>`},
		{withDefaultHeader(`<x:foo></y:foo>`), `element <foo> in space x closed by </foo> in space y`},
		{withDefaultHeader(`<? not ok ?>`), `expected target name after <?`},
		{withDefaultHeader(`<!- not ok -->`), `invalid sequence <!- not part of <!--`},
		{withDefaultHeader(`<!-? not ok -->`), `invalid sequence <!- not part of <!--`},
		{withDefaultHeader(`<![not ok]>`), `invalid <![ sequence`},
		{withDefaultHeader(`<zzz:foo xmlns:zzz="http://example.com"><bar>baz</bar></foo>`),
			`element <foo> in space zzz closed by </foo> in space ""`},
		{withDefaultHeader("\xf1"), `invalid UTF-8`},

		// Header-related errors.
		{`<?xml version="1.1" encoding="UTF-8"?>`, `unsupported version "1.1"; only version 1.0 is supported`},

		// Cases below are for "no errors".
		{withDefaultHeader(`<?ok?>`), ``},
		{withDefaultHeader(`<?ok version="ok"?>`), ``},
	}

	for _, test := range tests {
		d := NewDecoder(strings.NewReader(test.src))
		var err error
		for {
			_, err = d.Token()
			if err != nil {
				break
			}
		}
		if test.err == "" {
			if err != io.EOF {
				t.Errorf("parse %s: have %q error, expected none", test.src, err)
			}
			continue
		}
		// Inv: err != nil
		if err == io.EOF {
			t.Errorf("parse %s: unexpected EOF", test.src)
			continue
		}
		if !strings.Contains(err.Error(), test.err) {
			t.Errorf("parse %s: can't find %q error substring\nerror: %q", test.src, test.err, err)
			continue
		}
	}
}

const testInputHTMLAutoClose = `<?xml version="1.0" encoding="UTF-8"?>
<br>
<br/><br/>
<br><br>
<br></br>
<BR>
<BR/><BR/>
<Br></Br>
<BR><span id="test">abc</span><br/><br/>`

func BenchmarkHTMLAutoClose(b *testing.B) {
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			d := NewDecoder(strings.NewReader(testInputHTMLAutoClose))
			d.Strict = false
			d.AutoClose = HTMLAutoClose
			d.Entity = HTMLEntity
			for {
				_, err := d.Token()
				if err != nil {
					if err == io.EOF {
						break
					}
					b.Fatalf("unexpected error: %v", err)
				}
			}
		}
	})
}

func TestHTMLAutoClose(t *testing.T) {
	wantTokens := []Token{
		ProcInst{"xml", []byte(`version="1.0" encoding="UTF-8"`)},
		CharData("\n"),
		StartElement{Name{"", "br"}, []Attr{}},
		EndElement{Name{"", "br"}},
		CharData("\n"),
		StartElement{Name{"", "br"}, []Attr{}},
		EndElement{Name{"", "br"}},
		StartElement{Name{"", "br"}, []Attr{}},
		EndElement{Name{"", "br"}},
		CharData("\n"),
		StartElement{Name{"", "br"}, []Attr{}},
		EndElement{Name{"", "br"}},
		StartElement{Name{"", "br"}, []Attr{}},
		EndElement{Name{"", "br"}},
		CharData("\n"),
		StartElement{Name{"", "br"}, []Attr{}},
		EndElement{Name{"", "br"}},
		CharData("\n"),
		StartElement{Name{"", "BR"}, []Attr{}},
		EndElement{Name{"", "BR"}},
		CharData("\n"),
		StartElement{Name{"", "BR"}, []Attr{}},
		EndElement{Name{"", "BR"}},
		StartElement{Name{"", "BR"}, []Attr{}},
		EndElement{Name{"", "BR"}},
		CharData("\n"),
		StartElement{Name{"", "Br"}, []Attr{}},
		EndElement{Name{"", "Br"}},
		CharData("\n"),
		StartElement{Name{"", "BR"}, []Attr{}},
		EndElement{Name{"", "BR"}},
		StartElement{Name{"", "span"}, []Attr{{Name: Name{"", "id"}, Value: "test"}}},
		CharData("abc"),
		EndElement{Name{"", "span"}},
		StartElement{Name{"", "br"}, []Attr{}},
		EndElement{Name{"", "br"}},
		StartElement{Name{"", "br"}, []Attr{}},
		EndElement{Name{"", "br"}},
	}

	d := NewDecoder(strings.NewReader(testInputHTMLAutoClose))
	d.Strict = false
	d.AutoClose = HTMLAutoClose
	d.Entity = HTMLEntity
	var haveTokens []Token
	for {
		tok, err := d.Token()
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("unexpected error: %v", err)
		}
		haveTokens = append(haveTokens, CopyToken(tok))
	}
	if len(haveTokens) != len(wantTokens) {
		t.Errorf("tokens count mismatch: have %d, want %d", len(haveTokens), len(wantTokens))
	}
	for i, want := range wantTokens {
		if i >= len(haveTokens) {
			t.Errorf("token[%d] expected %#v, have no token", i, want)
		} else {
			have := haveTokens[i]
			if !reflect.DeepEqual(have, want) {
				t.Errorf("token[%d] mismatch:\nhave: %#v\nwant: %#v", i, have, want)
			}
		}
	}
}
