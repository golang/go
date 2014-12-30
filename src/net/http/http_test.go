// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests of internal functions with no better homes.

package http

import (
	"reflect"
	"testing"
)

func TestForeachHeaderElement(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"Foo", []string{"Foo"}},
		{" Foo", []string{"Foo"}},
		{"Foo ", []string{"Foo"}},
		{" Foo ", []string{"Foo"}},

		{"foo", []string{"foo"}},
		{"anY-cAsE", []string{"anY-cAsE"}},

		{"", nil},
		{",,,,  ,  ,,   ,,, ,", nil},

		{" Foo,Bar, Baz,lower,,Quux ", []string{"Foo", "Bar", "Baz", "lower", "Quux"}},
	}
	for _, tt := range tests {
		var got []string
		foreachHeaderElement(tt.in, func(v string) {
			got = append(got, v)
		})
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("foreachHeaderElement(%q) = %q; want %q", tt.in, got, tt.want)
		}
	}
}
