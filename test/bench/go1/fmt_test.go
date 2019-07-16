// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go1

// benchmark based on fmt/fmt_test.go

import (
	"bytes"
	"fmt"
	"testing"
)

func BenchmarkFmtFprintfEmpty(b *testing.B) {
	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		fmt.Fprintf(&buf, "")
	}
}

func BenchmarkFmtFprintfString(b *testing.B) {
	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		buf.Reset()
		fmt.Fprintf(&buf, "%s", "hello")
	}
}

func BenchmarkFmtFprintfInt(b *testing.B) {
	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		buf.Reset()
		fmt.Fprintf(&buf, "%d", 5)
	}
}

func BenchmarkFmtFprintfIntInt(b *testing.B) {
	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		buf.Reset()
		fmt.Fprintf(&buf, "%d %d", 5, 6)
	}
}

func BenchmarkFmtFprintfPrefixedInt(b *testing.B) {
	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		buf.Reset()
		fmt.Fprintf(&buf, "This is some meaningless prefix text that needs to be scanned %d", 6)
	}
}

func BenchmarkFmtFprintfFloat(b *testing.B) {
	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		buf.Reset()
		fmt.Fprintf(&buf, "%g", 5.23184)
	}
}

func BenchmarkFmtManyArgs(b *testing.B) {
	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		buf.Reset()
		fmt.Fprintf(&buf, "%2d/%2d/%2d %d:%d:%d %s %s\n", 3, 4, 5, 11, 12, 13, "hello", "world")
	}
}
