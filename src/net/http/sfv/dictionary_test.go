// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"reflect"
	"strings"
	"testing"
)

func TestDictionnary(t *testing.T) {
	dict := NewDictionary()

	add := []struct {
		in       string
		expected Member
		valid    bool
	}{
		{"f_o1o3-", NewItem(10.0), true},
		{"deleteme", NewItem(""), true},
		{"*f0.o*", NewItem(""), true},
		{"t", NewItem(true), true},
		{"f", NewItem(false), true},
		{"b", NewItem([]byte{0, 1}), true},
		{"0foo", NewItem(""), false},
		{"mAj", NewItem(""), false},
		{"_foo", NewItem(""), false},
		{"foo", NewItem(Token("é")), false},
	}

	var b strings.Builder

	for _, d := range add {
		vDict := NewDictionary()
		vDict.Add(d.in, d.expected)

		b.Reset()

		if valid := vDict.marshalSFV(&b) == nil; valid != d.valid {
			t.Errorf("(%v, %v).isValid() = %v; %v expected", d.in, d.expected, valid, d.valid)
		}

		if d.valid {
			dict.Add(d.in, d.expected)
		}
	}

	i := NewItem(123.0)
	dict.Add("f_o1o3-", i)

	newValue, _ := dict.Get("f_o1o3-")
	if newValue != i {
		t.Errorf(`Add("f_o1o3-") must overwrite the existing value`)
	}

	if !dict.Del("deleteme") {
		t.Errorf(`Del("deleteme") must return true`)
	}

	if dict.Del("deleteme") {
		t.Errorf(`the second call to Del("deleteme") must return false`)
	}

	if v, ok := dict.Get("*f0.o*"); v.(Item).Value != "" || !ok {
		t.Errorf(`Get("*f0.o*") = %v, %v; "", true expected`, v, ok)
	}

	if v, ok := dict.Get("notexist"); v != nil || ok {
		t.Errorf(`Get("notexist") = %v, %v; nil, false expected`, v, ok)
	}

	k := dict.Names()
	if len(k) != 5 {
		t.Errorf(`Names() = %v; {"f_o1o3-", "*f0.o*"} expected`, k)
	}

	m, _ := dict.Get("f_o1o3-")
	i = m.(Item)
	i.Params.Add("foo", 9.5)

	b.Reset()
	_ = dict.marshalSFV(&b)

	if b.String() != `f_o1o3-=123.0;foo=9.5, *f0.o*="", t, f=?0, b=:AAE=:` {
		t.Errorf(`Dictionnary.marshalSFV(): invalid serialization: %v`, b.String())
	}
}

func TestUnmarshalDictionary(t *testing.T) {
	d1 := NewDictionary()
	d1.Add("a", NewItem(false))
	d1.Add("b", NewItem(true))

	c := NewItem(true)
	c.Params.Add("foo", Token("bar"))
	d1.Add("c", c)

	data := []struct {
		in       []string
		expected *Dictionary
		valid    bool
	}{
		{[]string{"a=?0, b, c; foo=bar"}, d1, false},
		{[]string{"a=?0, b", "c; foo=bar"}, d1, false},
		{[]string{""}, NewDictionary(), false},
		{[]string{"é"}, nil, true},
		{[]string{`foo="é"`}, nil, true},
		{[]string{`foo;é`}, nil, true},
		{[]string{`f="foo" é`}, nil, true},
		{[]string{`f="foo",`}, nil, true},
	}

	for _, d := range data {
		l, err := UnmarshalDictionary(d.in)
		if d.valid && err == nil {
			t.Errorf("UnmarshalDictionary(%s): error expected", d.in)
		}

		if !d.valid && !reflect.DeepEqual(d.expected, l) {
			t.Errorf("UnmarshalDictionary(%s) = %v, %v; %v, <nil> expected", d.in, l, err, d.expected)
		}
	}
}
