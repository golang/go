// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"strings"
	"testing"
)

func TestDecimalMarshalSFV(t *testing.T) {
	data := []struct {
		in       float64
		expected string
		valid    bool
	}{
		{10.0, "10.0", true},
		{-10.123, "-10.123", true},
		{10.1236, "10.124", true},
		{-10.0, "-10.0", true},
		{0, "0.0", true},
		{-999999999999.0, "-999999999999.0", true},
		{999999999999.0, "999999999999.0", true},
		{9999999999999, "", false},
		{-9999999999999.0, "", false},
		{9999999999999.0, "", false},
	}

	var b strings.Builder

	for _, d := range data {
		b.Reset()

		err := marshalDecimal(&b, d.in)
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
