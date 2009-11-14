// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"io";
	"os";
	"reflect";
	"strings";
	"testing";
)

const testInput = `
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<body xmlns:foo="ns1" xmlns="ns2" xmlns:tag="ns3" `
	"\r\n\t" `  >
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
	CharData(strings.Bytes("\n")),
	ProcInst{"xml", strings.Bytes(`version="1.0" encoding="UTF-8"`)},
	CharData(strings.Bytes("\n")),
	Directive(strings.Bytes(`DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"`)),
	CharData(strings.Bytes("\n")),
	StartElement{Name{"", "body"}, []Attr{Attr{Name{"xmlns", "foo"}, "ns1"}, Attr{Name{"", "xmlns"}, "ns2"}, Attr{Name{"xmlns", "tag"}, "ns3"}}},
	CharData(strings.Bytes("\n  ")),
	StartElement{Name{"", "hello"}, []Attr{Attr{Name{"", "lang"}, "en"}}},
	CharData(strings.Bytes("World <>'\" 白鵬翔")),
	EndElement{Name{"", "hello"}},
	CharData(strings.Bytes("\n  ")),
	StartElement{Name{"", "goodbye"}, nil},
	EndElement{Name{"", "goodbye"}},
	CharData(strings.Bytes("\n  ")),
	StartElement{Name{"", "outer"}, []Attr{Attr{Name{"foo", "attr"}, "value"}, Attr{Name{"xmlns", "tag"}, "ns4"}}},
	CharData(strings.Bytes("\n    ")),
	StartElement{Name{"", "inner"}, nil},
	EndElement{Name{"", "inner"}},
	CharData(strings.Bytes("\n  ")),
	EndElement{Name{"", "outer"}},
	CharData(strings.Bytes("\n  ")),
	StartElement{Name{"tag", "name"}, nil},
	CharData(strings.Bytes("\n    ")),
	CharData(strings.Bytes("Some text here.")),
	CharData(strings.Bytes("\n  ")),
	EndElement{Name{"tag", "name"}},
	CharData(strings.Bytes("\n")),
	EndElement{Name{"", "body"}},
	Comment(strings.Bytes(" missing final newline ")),
}

var cookedTokens = []Token{
	CharData(strings.Bytes("\n")),
	ProcInst{"xml", strings.Bytes(`version="1.0" encoding="UTF-8"`)},
	CharData(strings.Bytes("\n")),
	Directive(strings.Bytes(`DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"`)),
	CharData(strings.Bytes("\n")),
	StartElement{Name{"ns2", "body"}, []Attr{Attr{Name{"xmlns", "foo"}, "ns1"}, Attr{Name{"", "xmlns"}, "ns2"}, Attr{Name{"xmlns", "tag"}, "ns3"}}},
	CharData(strings.Bytes("\n  ")),
	StartElement{Name{"ns2", "hello"}, []Attr{Attr{Name{"", "lang"}, "en"}}},
	CharData(strings.Bytes("World <>'\" 白鵬翔")),
	EndElement{Name{"ns2", "hello"}},
	CharData(strings.Bytes("\n  ")),
	StartElement{Name{"ns2", "goodbye"}, nil},
	EndElement{Name{"ns2", "goodbye"}},
	CharData(strings.Bytes("\n  ")),
	StartElement{Name{"ns2", "outer"}, []Attr{Attr{Name{"ns1", "attr"}, "value"}, Attr{Name{"xmlns", "tag"}, "ns4"}}},
	CharData(strings.Bytes("\n    ")),
	StartElement{Name{"ns2", "inner"}, nil},
	EndElement{Name{"ns2", "inner"}},
	CharData(strings.Bytes("\n  ")),
	EndElement{Name{"ns2", "outer"}},
	CharData(strings.Bytes("\n  ")),
	StartElement{Name{"ns3", "name"}, nil},
	CharData(strings.Bytes("\n    ")),
	CharData(strings.Bytes("Some text here.")),
	CharData(strings.Bytes("\n  ")),
	EndElement{Name{"ns3", "name"}},
	CharData(strings.Bytes("\n")),
	EndElement{Name{"ns2", "body"}},
	Comment(strings.Bytes(" missing final newline ")),
}

type stringReader struct {
	s	string;
	off	int;
}

func (r *stringReader) Read(b []byte) (n int, err os.Error) {
	if r.off >= len(r.s) {
		return 0, os.EOF
	}
	for r.off < len(r.s) && n < len(b) {
		b[n] = r.s[r.off];
		n++;
		r.off++;
	}
	return;
}

func (r *stringReader) ReadByte() (b byte, err os.Error) {
	if r.off >= len(r.s) {
		return 0, os.EOF
	}
	b = r.s[r.off];
	r.off++;
	return;
}

func StringReader(s string) io.Reader	{ return &stringReader{s, 0} }

func TestRawToken(t *testing.T) {
	p := NewParser(StringReader(testInput));

	for i, want := range rawTokens {
		have, err := p.RawToken();
		if err != nil {
			t.Fatalf("token %d: unexpected error: %s", i, err)
		}
		if !reflect.DeepEqual(have, want) {
			t.Errorf("token %d = %#v want %#v", i, have, want)
		}
	}
}

func TestToken(t *testing.T) {
	p := NewParser(StringReader(testInput));

	for i, want := range cookedTokens {
		have, err := p.Token();
		if err != nil {
			t.Fatalf("token %d: unexpected error: %s", i, err)
		}
		if !reflect.DeepEqual(have, want) {
			t.Errorf("token %d = %#v want %#v", i, have, want)
		}
	}
}
