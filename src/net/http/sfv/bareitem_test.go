// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestParseBareItem(t *testing.T) {
	data := []struct {
		in  string
		out interface{}
		err bool
	}{
		{"?1", true, false},
		{"?0", false, false},
		{"22", int64(22), false},
		{"-2.2", -2.2, false},
		{`"foo"`, "foo", false},
		{"abc", Token("abc"), false},
		{"*abc", Token("*abc"), false},
		{":YWJj:", []byte("abc"), false},
		{"", nil, true},
		{"~", nil, true},
	}

	for _, d := range data {
		s := &scanner{data: d.in}

		i, err := parseBareItem(s)
		if d.err && err == nil {
			t.Errorf("parseBareItem(%s): error expected", d.in)
		}

		if !d.err && !reflect.DeepEqual(d.out, i) {
			t.Errorf("parseBareItem(%s) = %v, %v; %v, <nil> expected", d.in, i, err, d.out)
		}
	}
}

func TestMarshalBareItem(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	var b strings.Builder
	_ = marshalBareItem(&b, time.Second)
}

func TestAssertBareItem(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	assertBareItem(time.Second)
}
