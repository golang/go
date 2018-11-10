// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package driver

import (
	"reflect"
	"testing"
	"time"
)

type valueConverterTest struct {
	c   ValueConverter
	in  interface{}
	out interface{}
	err string
}

var now = time.Now()
var answer int64 = 42

type (
	i  int64
	f  float64
	b  bool
	bs []byte
	s  string
	t  time.Time
	is []int
)

var valueConverterTests = []valueConverterTest{
	{Bool, "true", true, ""},
	{Bool, "True", true, ""},
	{Bool, []byte("t"), true, ""},
	{Bool, true, true, ""},
	{Bool, "1", true, ""},
	{Bool, 1, true, ""},
	{Bool, int64(1), true, ""},
	{Bool, uint16(1), true, ""},
	{Bool, "false", false, ""},
	{Bool, false, false, ""},
	{Bool, "0", false, ""},
	{Bool, 0, false, ""},
	{Bool, int64(0), false, ""},
	{Bool, uint16(0), false, ""},
	{c: Bool, in: "foo", err: "sql/driver: couldn't convert \"foo\" into type bool"},
	{c: Bool, in: 2, err: "sql/driver: couldn't convert 2 into type bool"},
	{DefaultParameterConverter, now, now, ""},
	{DefaultParameterConverter, (*int64)(nil), nil, ""},
	{DefaultParameterConverter, &answer, answer, ""},
	{DefaultParameterConverter, &now, now, ""},
	{DefaultParameterConverter, i(9), int64(9), ""},
	{DefaultParameterConverter, f(0.1), float64(0.1), ""},
	{DefaultParameterConverter, b(true), true, ""},
	{DefaultParameterConverter, bs{1}, []byte{1}, ""},
	{DefaultParameterConverter, s("a"), "a", ""},
	{DefaultParameterConverter, is{1}, nil, "unsupported type driver.is, a slice of int"},
}

func TestValueConverters(t *testing.T) {
	for i, tt := range valueConverterTests {
		out, err := tt.c.ConvertValue(tt.in)
		goterr := ""
		if err != nil {
			goterr = err.Error()
		}
		if goterr != tt.err {
			t.Errorf("test %d: %T(%T(%v)) error = %q; want error = %q",
				i, tt.c, tt.in, tt.in, goterr, tt.err)
		}
		if tt.err != "" {
			continue
		}
		if !reflect.DeepEqual(out, tt.out) {
			t.Errorf("test %d: %T(%T(%v)) = %v (%T); want %v (%T)",
				i, tt.c, tt.in, tt.in, out, out, tt.out, tt.out)
		}
	}
}
