// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"reflect"
	"strings"
	"testing"
)

func TestParameters(t *testing.T) {
	p := NewParams()

	add := []struct {
		in       string
		expected interface{}
		valid    bool
	}{
		{"f_o1o3-", 10.0, true},
		{"deleteme", "", true},
		{"*f0.o*", "", true},
		{"t", true, true},
		{"f", false, true},
		{"b", []byte{0, 1}, true},
		{"0foo", "", false},
		{"mAj", "", false},
		{"_foo", "", false},
		{"foo", "é", false},
	}

	var b strings.Builder

	for _, d := range add {
		vParams := NewParams()
		vParams.Add(d.in, d.expected)

		b.Reset()

		if valid := vParams.marshalSFV(&b) == nil; valid != d.valid {
			t.Errorf("(%v, %v).isValid() = %v; %v expected", d.in, d.expected, valid, d.valid)
		}

		if d.valid {
			p.Add(d.in, d.expected)
		}
	}

	p.Add("f_o1o3-", 123.0)

	newValue, _ := p.Get("f_o1o3-")
	if newValue != 123.0 {
		t.Errorf(`Add("f_o1o3-") must overwrite the existing value`)
	}

	if !p.Del("deleteme") {
		t.Errorf(`Del("deleteme") must return true`)
	}

	if p.Del("deleteme") {
		t.Errorf(`the second call to Del("deleteme") must return false`)
	}

	if v, ok := p.Get("*f0.o*"); v != "" || !ok {
		t.Errorf(`Get("*f0.o*") = %v, %v; "", true expected`, v, ok)
	}

	if v, ok := p.Get("notexist"); v != nil || ok {
		t.Errorf(`Get("notexist") = %v, %v; nil, false expected`, v, ok)
	}

	k := p.Names()
	if len(k) != 5 {
		t.Errorf(`Names() = %v; {"f_o1o3-", "*f0.o*"} expected`, k)
	}

	b.Reset()
	err := p.marshalSFV(&b)

	if b.String() != `;f_o1o3-=123.0;*f0.o*="";t;f=?0;b=:AAE=:` {
		t.Errorf(`marshalSFV(): invalid serialization: %v (%v)`, b.String(), err)
	}
}

func TestParseParameters(t *testing.T) {
	p0 := NewParams()
	p0.Add("foo", true)
	p0.Add("*bar", "baz")

	data := []struct {
		in  string
		out *Params
		err bool
	}{
		{`;foo=?1;*bar="baz" foo`, p0, false},
		{`;foo;*bar="baz" foo`, p0, false},
		{`;é=?0`, p0, true},
		{`;foo=é`, p0, true},
	}

	for _, d := range data {
		s := &scanner{data: d.in}

		p, err := parseParams(s)
		if d.err && err == nil {
			t.Errorf("parseParameters(%s): error expected", d.in)
		}

		if !d.err && !reflect.DeepEqual(p, d.out) {
			t.Errorf("parseParameters(%s) = %v, %v; %v, <nil> expected", d.in, p, err, d.out)
		}
	}
}
