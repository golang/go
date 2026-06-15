// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	. "fmt"
	"io"
	"testing"
)

// BenchmarkSprint exercises doPrint (no format string path).
func BenchmarkSprintInts(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Sprint(1, 2, 3, 4, 5)
	}
}

func BenchmarkSprintlnInts(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Sprintln(1, 2, 3, 4, 5)
	}
}

func BenchmarkSprintStrings(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Sprint("a", "b", "c", "d", "e")
	}
}

func BenchmarkSprintMixed(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Sprint(1, "hello", 3.14, true, "world")
	}
}

func BenchmarkPrintInts(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Print(1, 2, 3, 4, 5)
	}
}

func BenchmarkPrintlnInts(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Println(1, 2, 3, 4, 5)
	}
}

// discard is a Writer that discards all writes.
type discard struct{}

func (discard) Write(p []byte) (int, error) { return len(p), nil }

func BenchmarkFprintInts(b *testing.B) {
	w := io.Writer(discard{})
	for i := 0; i < b.N; i++ {
		Fprint(w, 1, 2, 3, 4, 5)
	}
}

func BenchmarkFprintlnInts(b *testing.B) {
	w := io.Writer(discard{})
	for i := 0; i < b.N; i++ {
		Fprintln(w, 1, 2, 3, 4, 5)
	}
}
