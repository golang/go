// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"strings"
	"testing"
)

func TestKeyMarshalSFV(t *testing.T) {
	data := []struct {
		in       string
		expected string
		valid    bool
	}{
		{"f1oo", "f1oo", true},
		{"*foo0", "*foo0", true},
		{"", "", false},
		{"1foo", "", false},
		{"fOo", "", false},
	}

	var b strings.Builder

	for _, d := range data {
		b.Reset()

		err := marshalKey(&b, d.in)
		if d.valid && err != nil {
			t.Errorf("error not expected for %v, got %v", d.in, err)
		} else if !d.valid && err == nil {
			t.Errorf("error expected for %v, got %v", d.in, err)
		}

		if b.String() != d.expected {
			t.Errorf("got %v; want %v", b.String(), d.expected)
		}
	}
}

func TestParseKey(t *testing.T) {
	data := []struct {
		in       string
		expected string
		err      bool
	}{
		{"t", "t", false},
		{"tok", "tok", false},
		{"*k-.*", "*k-.*", false},
		{"k=", "k", false},
		{"", "", true},
		{"é", "", true},
	}

	for _, d := range data {
		s := &scanner{data: d.in}

		i, err := parseKey(s)
		if d.err && err == nil {
			t.Errorf("parseKey(%s): error expected", d.in)
		}

		if !d.err && d.expected != i {
			t.Errorf("parseKey(%s) = %v, %v; %v, <nil> expected", d.in, i, err, d.expected)
		}
	}
}
