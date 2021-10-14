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

func TestClone(t *testing.T) {
	var cloneTests = []string{
		"",
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
		if inputHeader.Data == cloneHeader.Data {
			t.Errorf("Clone(%q) return value should not reference inputs backing memory.", input)
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
