// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"reflect"
	"strings"
	"testing"
)

func TestMarshalItem(t *testing.T) {
	data := []struct {
		in       Item
		expected string
		valid    bool
	}{
		{NewItem(0), "0", true},
		{NewItem(int8(-42)), "-42", true},
		{NewItem(int16(-42)), "-42", true},
		{NewItem(int32(-42)), "-42", true},
		{NewItem(int64(-42)), "-42", true},
		{NewItem(uint(42)), "42", true},
		{NewItem(uint8(42)), "42", true},
		{NewItem(uint16(42)), "42", true},
		{NewItem(uint32(42)), "42", true},
		{NewItem(uint64(42)), "42", true},
		{NewItem(1.1), "1.1", true},
		{NewItem(""), `""`, true},
		{NewItem(Token("foo")), "foo", true},
		{NewItem([]byte{0, 1}), ":AAE=:", true},
		{NewItem(false), "?0", true},
		{NewItem(int64(9999999999999999)), "", false},
		{NewItem(9999999999999999.22), "", false},
		{NewItem("Kévin"), "", false},
		{NewItem(Token("/foo")), "", false},
		{Item{}, "", false},
	}

	for _, d := range data {
		r, err := Marshal(d.in)
		if d.valid && err != nil {
			t.Errorf("error not expected for %v, got %v", d.in, err)
		} else if !d.valid && err == nil {
			t.Errorf("error expected for %v, got %v", d.in, err)
		}

		if r != d.expected {
			t.Errorf("got %v; want %v", r, d.expected)
		}
	}
}

func TestItemParamsMarshalSFV(t *testing.T) {
	i := NewItem(Token("bar"))
	i.Params.Add("foo", 0.0)
	i.Params.Add("baz", true)

	var b strings.Builder
	_ = i.marshalSFV(&b)

	if b.String() != "bar;foo=0.0;baz" {
		t.Error("marshalSFV(): invalid")
	}
}

func TestUnmarshalItem(t *testing.T) {
	i1 := NewItem(true)
	i1.Params.Add("foo", true)
	i1.Params.Add("*bar", Token("tok"))

	data := []struct {
		in       []string
		expected Item
		valid    bool
	}{
		{[]string{"?1;foo;*bar=tok"}, i1, false},
		{[]string{"  ?1;foo;*bar=tok  "}, i1, false},
		{[]string{`"foo`, `bar"`}, NewItem("foo,bar"), false},
		{[]string{"é", ""}, Item{}, true},
		{[]string{"tok;é"}, Item{}, true},
		{[]string{"  ?1;foo;*bar=tok  é"}, Item{}, true},
	}

	for _, d := range data {
		i, err := UnmarshalItem(d.in)
		if d.valid && err == nil {
			t.Errorf("UnmarshalItem(%s): error expected", d.in)
		}

		if !d.valid && !reflect.DeepEqual(d.expected, i) {
			t.Errorf("UnmarshalItem(%s) = %v, %v; %v, <nil> expected", d.in, i, err, d.expected)
		}
	}
}
