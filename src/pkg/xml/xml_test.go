// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bytes"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
)

const testInput = `
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<body xmlns:foo="ns1" xmlns="ns2" xmlns:tag="ns3" ` +
	"\r\n\t" + `  >
  <hello lang="en">World &lt;&gt;&apos;&quot; &#x767d;&#40300;翔</hello>
  <goodbye />
  <outer foo:attr="value" xmlns:tag="ns4">
    <inner/>
  </outer>
  <tag:name>
    <![CDATA[Some text here.]]>
  </tag:name>
</body><!-- missing final newline -->`

var rawTokens = []Token{
	CharData([]byte("\n")),
	ProcInst{"xml", []byte(`version="1.0" encoding="UTF-8"`)},
	CharData([]byte("\n")),
	Directive([]byte(`DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"`),
	),
	CharData([]byte("\n")),
	StartElement{Name{"", "body"}, []Attr{{Name{"xmlns", "foo"}, "ns1"}, {Name{"", "xmlns"}, "ns2"}, {Name{"xmlns", "tag"}, "ns3"}}},
	CharData([]byte("\n  ")),
	StartElement{Name{"", "hello"}, []Attr{{Name{"", "lang"}, "en"}}},
	CharData([]byte("World <>'\" 白鵬翔")),
	EndElement{Name{"", "hello"}},
	CharData([]byte("\n  ")),
	StartElement{Name{"", "goodbye"}, nil},
	EndElement{Name{"", "goodbye"}},
	CharData([]byte("\n  ")),
	StartElement{Name{"", "outer"}, []Attr{{Name{"foo", "attr"}, "value"}, {Name{"xmlns", "tag"}, "ns4"}}},
	CharData([]byte("\n    ")),
	StartElement{Name{"", "inner"}, nil},
	EndElement{Name{"", "inner"}},
	CharData([]byte("\n  ")),
	EndElement{Name{"", "outer"}},
	CharData([]byte("\n  ")),
	StartElement{Name{"tag", "name"}, nil},
	CharData([]byte("\n    ")),
	CharData([]byte("Some text here.")),
	CharData([]byte("\n  ")),
	EndElement{Name{"tag", "name"}},
	CharData([]byte("\n")),
	EndElement{Name{"", "body"}},
	Comment([]byte(" missing final newline ")),
}

var cookedTokens = []Token{
	CharData([]byte("\n")),
	ProcInst{"xml", []byte(`version="1.0" encoding="UTF-8"`)},
	CharData([]byte("\n")),
	Directive([]byte(`DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"`),
	),
	CharData([]byte("\n")),
	StartElement{Name{"ns2", "body"}, []Attr{{Name{"xmlns", "foo"}, "ns1"}, {Name{"", "xmlns"}, "ns2"}, {Name{"xmlns", "tag"}, "ns3"}}},
	CharData([]byte("\n  ")),
	StartElement{Name{"ns2", "hello"}, []Attr{{Name{"", "lang"}, "en"}}},
	CharData([]byte("World <>'\" 白鵬翔")),
	EndElement{Name{"ns2", "hello"}},
	CharData([]byte("\n  ")),
	StartElement{Name{"ns2", "goodbye"}, nil},
	EndElement{Name{"ns2", "goodbye"}},
	CharData([]byte("\n  ")),
	StartElement{Name{"ns2", "outer"}, []Attr{{Name{"ns1", "attr"}, "value"}, {Name{"xmlns", "tag"}, "ns4"}}},
	CharData([]byte("\n    ")),
	StartElement{Name{"ns2", "inner"}, nil},
	EndElement{Name{"ns2", "inner"}},
	CharData([]byte("\n  ")),
	EndElement{Name{"ns2", "outer"}},
	CharData([]byte("\n  ")),
	StartElement{Name{"ns3", "name"}, nil},
	CharData([]byte("\n    ")),
	CharData([]byte("Some text here.")),
	CharData([]byte("\n  ")),
	EndElement{Name{"ns3", "name"}},
	CharData([]byte("\n")),
	EndElement{Name{"ns2", "body"}},
	Comment([]byte(" missing final newline ")),
}

const testInputAltEncoding = `
<?xml version="1.0" encoding="x-testing-uppercase"?>
<TAG>VALUE</TAG>`

var rawTokensAltEncoding = []Token{
	CharData([]byte("\n")),
	ProcInst{"xml", []byte(`version="1.0" encoding="x-testing-uppercase"`)},
	CharData([]byte("\n")),
	StartElement{Name{"", "tag"}, nil},
	CharData([]byte("value")),
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

type stringReader struct {
	s   string
	off int
}

func (r *stringReader) Read(b []byte) (n int, err os.Error) {
	if r.off >= len(r.s) {
		return 0, os.EOF
	}
	for r.off < len(r.s) && n < len(b) {
		b[n] = r.s[r.off]
		n++
		r.off++
	}
	return
}

func (r *stringReader) ReadByte() (b byte, err os.Error) {
	if r.off >= len(r.s) {
		return 0, os.EOF
	}
	b = r.s[r.off]
	r.off++
	return
}

func StringReader(s string) io.Reader { return &stringReader{s, 0} }

func TestRawToken(t *testing.T) {
	p := NewParser(StringReader(testInput))
	testRawToken(t, p, rawTokens)
}

type downCaser struct {
	t *testing.T
	r io.ByteReader
}

func (d *downCaser) ReadByte() (c byte, err os.Error) {
	c, err = d.r.ReadByte()
	if c >= 'A' && c <= 'Z' {
		c += 'a' - 'A'
	}
	return
}

func (d *downCaser) Read(p []byte) (int, os.Error) {
	d.t.Fatalf("unexpected Read call on downCaser reader")
	return 0, os.EINVAL
}

func TestRawTokenAltEncoding(t *testing.T) {
	sawEncoding := ""
	p := NewParser(StringReader(testInputAltEncoding))
	p.CharsetReader = func(charset string, input io.Reader) (io.Reader, os.Error) {
		sawEncoding = charset
		if charset != "x-testing-uppercase" {
			t.Fatalf("unexpected charset %q", charset)
		}
		return &downCaser{t, input.(io.ByteReader)}, nil
	}
	testRawToken(t, p, rawTokensAltEncoding)
}

func TestRawTokenAltEncodingNoConverter(t *testing.T) {
	p := NewParser(StringReader(testInputAltEncoding))
	token, err := p.RawToken()
	if token == nil {
		t.Fatalf("expected a token on first RawToken call")
	}
	if err != nil {
		t.Fatal(err)
	}
	token, err = p.RawToken()
	if token != nil {
		t.Errorf("expected a nil token; got %#v", token)
	}
	if err == nil {
		t.Fatalf("expected an error on second RawToken call")
	}
	const encoding = "x-testing-uppercase"
	if !strings.Contains(err.String(), encoding) {
		t.Errorf("expected error to contain %q; got error: %v",
			encoding, err)
	}
}

func testRawToken(t *testing.T, p *Parser, rawTokens []Token) {
	for i, want := range rawTokens {
		have, err := p.RawToken()
		if err != nil {
			t.Fatalf("token %d: unexpected error: %s", i, err)
		}
		if !reflect.DeepEqual(have, want) {
			t.Errorf("token %d = %#v want %#v", i, have, want)
		}
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
	CharData([]byte("\n")),
	Directive([]byte(`DOCTYPE [<!ENTITY rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#">]`)),
	CharData([]byte("\n")),
	Directive([]byte(`DOCTYPE [<!ENTITY xlt ">">]`)),
	CharData([]byte("\n")),
	Directive([]byte(`DOCTYPE [<!ENTITY xlt "<">]`)),
	CharData([]byte("\n")),
	Directive([]byte(`DOCTYPE [<!ENTITY xlt '>'>]`)),
	CharData([]byte("\n")),
	Directive([]byte(`DOCTYPE [<!ENTITY xlt '<'>]`)),
	CharData([]byte("\n")),
	Directive([]byte(`DOCTYPE [<!ENTITY xlt '">'>]`)),
	CharData([]byte("\n")),
	Directive([]byte(`DOCTYPE [<!ENTITY xlt "'<">]`)),
	CharData([]byte("\n")),
}

func TestNestedDirectives(t *testing.T) {
	p := NewParser(StringReader(nestedDirectivesInput))

	for i, want := range nestedDirectivesTokens {
		have, err := p.Token()
		if err != nil {
			t.Fatalf("token %d: unexpected error: %s", i, err)
		}
		if !reflect.DeepEqual(have, want) {
			t.Errorf("token %d = %#v want %#v", i, have, want)
		}
	}
}

func TestToken(t *testing.T) {
	p := NewParser(StringReader(testInput))

	for i, want := range cookedTokens {
		have, err := p.Token()
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
		p := NewParser(StringReader(xmlInput[i]))
		var err os.Error
		for _, err = p.Token(); err == nil; _, err = p.Token() {
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
	<true1>true</true1>
	<true2>1</true2>
	<false1>false</false1>
	<false2>0</false2>
	<int>1</int>
	<int8>-2</int8>
	<int16>3</int16>
	<int32>-4</int32>
	<int64>5</int64>
	<uint>6</uint>
	<uint8>7</uint8>
	<uint16>8</uint16>
	<uint32>9</uint32>
	<uint64>10</uint64>
	<uintptr>11</uintptr>
	<float>12.0</float>
	<float32>13.0</float32>
	<float64>14.0</float64>
	<string>15</string>
	<ptrstring>16</ptrstring>
</allscalars>`

func TestAllScalars(t *testing.T) {
	var a allScalars
	buf := bytes.NewBufferString(testScalarsInput)
	err := Unmarshal(buf, &a)

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
	data := `<item><field_a>abcd</field_a></item>`
	var i item
	buf := bytes.NewBufferString(data)
	err := Unmarshal(buf, &i)

	if err != nil || i.Field_a != "abcd" {
		t.Fatal("Expecting abcd")
	}
}

func TestUnquotedAttrs(t *testing.T) {
	data := "<tag attr=azAZ09:-_\t>"
	p := NewParser(StringReader(data))
	p.Strict = false
	token, err := p.Token()
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
		p := NewParser(StringReader(test[0]))
		p.Strict = false
		token, err := p.Token()
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
	if !reflect.DeepEqual(tok1, tok2) {
		t.Error("CopyToken(StartElement) != StartElement")
	}
	elt.Attr[0] = Attr{Name{"", "lang"}, "de"}
	if reflect.DeepEqual(tok1, tok2) {
		t.Error("CopyToken(CharData) uses same buffer.")
	}
}

func TestSyntaxErrorLineNum(t *testing.T) {
	testInput := "<P>Foo<P>\n\n<P>Bar</>\n"
	p := NewParser(StringReader(testInput))
	var err os.Error
	for _, err = p.Token(); err == nil; _, err = p.Token() {
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
	p := NewParser(StringReader(input))
	var err os.Error
	for _, err = p.RawToken(); err == nil; _, err = p.RawToken() {
	}
	if err != os.EOF {
		t.Fatalf("p.RawToken() = _, %v, want _, os.EOF", err)
	}
}

func TestTrailingToken(t *testing.T) {
	input := `<FOO></FOO>  `
	p := NewParser(StringReader(input))
	var err os.Error
	for _, err = p.Token(); err == nil; _, err = p.Token() {
	}
	if err != os.EOF {
		t.Fatalf("p.Token() = _, %v, want _, os.EOF", err)
	}
}

func TestEntityInsideCDATA(t *testing.T) {
	input := `<test><![CDATA[ &val=foo ]]></test>`
	p := NewParser(StringReader(input))
	var err os.Error
	for _, err = p.Token(); err == nil; _, err = p.Token() {
	}
	if err != os.EOF {
		t.Fatalf("p.Token() = _, %v, want _, os.EOF", err)
	}
}

// The last three tests (respectively one for characters in attribute
// names and two for character entities) pass not because of code
// changed for issue 1259, but instead pass with the given messages
// from other parts of xml.Parser.  I provide these to note the
// current behavior of situations where one might think that character
// range checking would detect the error, but it does not in fact.

var characterTests = []struct {
	in  string
	err string
}{
	{"\x12<doc/>", "illegal character code U+0012"},
	{"<?xml version=\"1.0\"?>\x0b<doc/>", "illegal character code U+000B"},
	{"\xef\xbf\xbe<doc/>", "illegal character code U+FFFE"},
	{"<?xml version=\"1.0\"?><doc>\r\n<hiya/>\x07<toots/></doc>", "illegal character code U+0007"},
	{"<?xml version=\"1.0\"?><doc \x12='value'>what's up</doc>", "expected attribute name in element"},
	{"<doc>&\x01;</doc>", "invalid character entity &;"},
	{"<doc>&\xef\xbf\xbe;</doc>", "invalid character entity &;"},
}

func TestDisallowedCharacters(t *testing.T) {

	for i, tt := range characterTests {
		p := NewParser(StringReader(tt.in))
		var err os.Error

		for err == nil {
			_, err = p.Token()
		}
		synerr, ok := err.(*SyntaxError)
		if !ok {
			t.Fatalf("input %d p.Token() = _, %v, want _, *SyntaxError", i, err)
		}
		if synerr.Msg != tt.err {
			t.Fatalf("input %d synerr.Msg wrong: want '%s', got '%s'", i, tt.err, synerr.Msg)
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
