// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"reflect"
	"testing"
)

func TestConsumeToken(t *testing.T) {
	tests := [...][3]string{
		{"foo bar", "foo", " bar"},
		{"bar", "bar", ""},
		{"", "", ""},
		{" foo", "", " foo"},
	}
	for _, test := range tests {
		token, rest := consumeToken(test[0])
		expectedToken := test[1]
		expectedRest := test[2]
		if token != expectedToken {
			t.Errorf("expected to consume token '%s', not '%s' from '%s'",
				expectedToken, token, test[0])
		} else if rest != expectedRest {
			t.Errorf("expected to have left '%s', not '%s' after reading token '%s' from '%s'",
				expectedRest, rest, token, test[0])
		}
	}
}

func TestConsumeValue(t *testing.T) {
	tests := [...][3]string{
		{"foo bar", "foo", " bar"},
		{"bar", "bar", ""},
		{" bar ", "", " bar "},
		{`"My value"end`, "My value", "end"},
		{`"My value" end`, "My value", " end"},
		{`"\\" rest`, "\\", " rest"},
		{`"My \" value"end`, "My \" value", "end"},
		{`"\" rest`, "", `"\" rest`},
	}
	for _, test := range tests {
		value, rest := consumeValue(test[0])
		expectedValue := test[1]
		expectedRest := test[2]
		if value != expectedValue {
			t.Errorf("expected to consume value [%s], not [%s] from [%s]",
				expectedValue, value, test[0])
		} else if rest != expectedRest {
			t.Errorf("expected to have left [%s], not [%s] after reading value [%s] from [%s]",
				expectedRest, rest, value, test[0])
		}
	}
}

func TestConsumeMediaParam(t *testing.T) {
	tests := [...][4]string{
		{" ; foo=bar", "foo", "bar", ""},
		{"; foo=bar", "foo", "bar", ""},
		{";foo=bar", "foo", "bar", ""},
		{`;foo="bar"`, "foo", "bar", ""},
		{`;foo="bar"; `, "foo", "bar", "; "},
		{`;foo="bar"; foo=baz`, "foo", "bar", "; foo=baz"},
		{` ; boundary=----CUT;`, "boundary", "----CUT", ";"},
		{` ; key=value;  blah="value";name="foo" `, "key", "value", `;  blah="value";name="foo" `},
		{`;  blah="value";name="foo" `, "blah", "value", `;name="foo" `},
		{`;name="foo" `, "name", "foo", ` `},
	}
	for _, test := range tests {
		param, value, rest := consumeMediaParam(test[0])
		expectedParam := test[1]
		expectedValue := test[2]
		expectedRest := test[3]
		if param != expectedParam {
			t.Errorf("expected to consume param [%s], not [%s] from [%s]",
				expectedParam, param, test[0])
		} else if value != expectedValue {
			t.Errorf("expected to consume value [%s], not [%s] from [%s]",
				expectedValue, value, test[0])
		} else if rest != expectedRest {
			t.Errorf("expected to have left [%s], not [%s] after reading [%s/%s] from [%s]",
				expectedRest, rest, param, value, test[0])
		}
	}
}

type mediaTypeTest struct {
	in string
	t  string
	p  map[string]string
}

func TestParseMediaType(t *testing.T) {
	// Convenience map initializer
	m := func(s ...string) map[string]string {
		sm := make(map[string]string)
		for i := 0; i < len(s); i += 2 {
			sm[s[i]] = s[i+1]
		}
		return sm
	}

	nameFoo := map[string]string{"name": "foo"}
	tests := []mediaTypeTest{
		{`form-data; name="foo"`, "form-data", nameFoo},
		{` form-data ; name=foo`, "form-data", nameFoo},
		{`FORM-DATA;name="foo"`, "form-data", nameFoo},
		{` FORM-DATA ; name="foo"`, "form-data", nameFoo},
		{` FORM-DATA ; name="foo"`, "form-data", nameFoo},

		{`form-data; key=value;  blah="value";name="foo" `,
			"form-data",
			m("key", "value", "blah", "value", "name", "foo")},

		// Tests from http://greenbytes.de/tech/tc2231/
		// TODO(bradfitz): add the rest of the tests from that site.
		{`attachment; filename="f\oo.html"`,
			"attachment",
			m("filename", "foo.html")},
		{`attachment; filename="\"quoting\" tested.html"`,
			"attachment",
			m("filename", `"quoting" tested.html`)},
		{`attachment; filename="Here's a semicolon;.html"`,
			"attachment",
			m("filename", "Here's a semicolon;.html")},
		{`attachment; foo="\"\\";filename="foo.html"`,
			"attachment",
			m("foo", "\"\\", "filename", "foo.html")},
		{`attachment; filename=foo.html`,
			"attachment",
			m("filename", "foo.html")},
		{`attachment; filename=foo.html ;`,
			"attachment",
			m("filename", "foo.html")},
		{`attachment; filename='foo.html'`,
			"attachment",
			m("filename", "foo.html")},
		{`attachment; filename="foo-%41.html"`,
			"attachment",
			m("filename", "foo-%41.html")},
		{`attachment; filename="foo-%\41.html"`,
			"attachment",
			m("filename", "foo-%41.html")},
		{`filename=foo.html`,
			"", m()},
		{`x=y; filename=foo.html`,
			"", m()},
		{`"foo; filename=bar;baz"; filename=qux`,
			"", m()},
		{`inline; attachment; filename=foo.html`,
			"", m()},
		{`attachment; filename="foo.html".txt`,
			"", m()},
		{`attachment; filename="bar`,
			"", m()},
		{`attachment; creation-date="Wed, 12 Feb 1997 16:29:51 -0500"`,
			"attachment",
			m("creation-date", "Wed, 12 Feb 1997 16:29:51 -0500")},
		{`foobar`, "foobar", m()},
		// TODO(bradfitz): rest of them, including RFC2231 encoded UTF-8 and
		// other charsets.
	}
	for _, test := range tests {
		mt, params := ParseMediaType(test.in)
		if g, e := mt, test.t; g != e {
			t.Errorf("for input %q, expected type %q, got %q",
				test.in, e, g)
			continue
		}
		if len(params) == 0 && len(test.p) == 0 {
			continue
		}
		if !reflect.DeepEqual(params, test.p) {
			t.Errorf("for input %q, wrong params.\n"+
				"expected: %#v\n"+
				"     got: %#v",
				test.in, test.p, params)
		}
	}
}

func TestParseMediaTypeBogus(t *testing.T) {
	mt, params := ParseMediaType("bogus ;=========")
	if mt != "" {
		t.Error("expected empty type")
	}
	if params != nil {
		t.Error("expected nil params")
	}
}
