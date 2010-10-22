// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
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

func TestParseMediaType(t *testing.T) {
	tests := [...]string{
		`form-data; name="foo"`,
		` form-data ; name=foo`,
		`FORM-DATA;name="foo"`,
		` FORM-DATA ; name="foo"`,
		` FORM-DATA ; name="foo"`,
		`form-data; key=value;  blah="value";name="foo" `,
	}
	for _, test := range tests {
		mt, params := ParseMediaType(test)
		if mt != "form-data" {
			t.Errorf("expected type form-data for %s, got [%s]", test, mt)
			continue
		}
		if params["name"] != "foo" {
			t.Errorf("expected name=foo for %s", test)
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
