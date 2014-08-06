// Copyright 2009 The Go Authors.  All rights reserved.
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

var nonStringEntity = map[string]string{"": "oops!", "0a": "oops!"}

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
	Field_a string
}

func TestIssue569(t *testing.T) {
	data := `<item><Field_a>abcd</Field_a></item>`
	var i item
	err := Unmarshal([]byte(data), &i)

	if err != nil || i.Field_a != "abcd" {
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

type procInstEncodingTest struct {
	expect, got string
}

var procInstTests = []struct {
	input, expect string
}{
	{`version="1.0" encoding="utf-8"`, "utf-8"},
	{`version="1.0" encoding='utf-8'`, "utf-8"},
	{`version="1.0" encoding='utf-8' `, "utf-8"},
	{`version="1.0" encoding=utf-8`, ""},
	{`encoding="FOO" `, "FOO"},
}

func TestProcInstEncoding(t *testing.T) {
	for _, test := range procInstTests {
		got := procInstEncoding(test.input)
		if got != test.expect {
			t.Errorf("procInstEncoding(%q) = %q; want %q", test.input, got, test.expect)
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
	Directive(`DOCTYPE [<!ENTITY rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#">]`),
	CharData("\n"),
	Directive(`DOCTYPE [<!ENTITY go "Golang">]`),
	CharData("\n"),
	Directive(`DOCTYPE <!-> <!>    [<!ENTITY go "Golang">]`),
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

	buff := new(bytes.Buffer)
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
