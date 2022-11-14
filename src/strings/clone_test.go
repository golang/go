// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.ß

package strings_test

import (
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

		if len(input) != 0 && unsafe.StringData(clone) == unsafe.StringData(input) {
			t.Errorf("Clone(%q) return value should not reference inputs backing memory.", input)
		}

		if len(input) == 0 && unsafe.StringData(clone) != unsafe.StringData(emptyString) {
			t.Errorf("Clone(%#v) return value should be equal to empty string.", unsafe.StringData(input))
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
