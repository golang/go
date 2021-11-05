// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.ß

package strings_test

import (
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

var emptyString string

func TestClone(t *testing.T) {
	var cloneTests = []string{
		"",
		strings.Clone(""),
		strings.Repeat("a", 42)[:0],
		"short",
		strings.Repeat("a", 42),
	}
	for _, input := range cloneTests {
		clone := strings.Clone(input)
		if clone != input {
			t.Errorf("Clone(%q) = %q; want %q", input, clone, input)
		}

		inputHeader := (*reflect.StringHeader)(unsafe.Pointer(&input))
		cloneHeader := (*reflect.StringHeader)(unsafe.Pointer(&clone))
		if len(input) != 0 && cloneHeader.Data == inputHeader.Data {
			t.Errorf("Clone(%q) return value should not reference inputs backing memory.", input)
		}

		emptyHeader := (*reflect.StringHeader)(unsafe.Pointer(&emptyString))
		if len(input) == 0 && cloneHeader.Data != emptyHeader.Data {
			t.Errorf("Clone(%#v) return value should be equal to empty string.", inputHeader)
		}
	}
}

func BenchmarkClone(b *testing.B) {
	var str = strings.Repeat("a", 42)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stringSink = strings.Clone(str)
	}
}
