// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	. "strings"
	"testing"
)

func BenchmarkSplitSeqEmptySeparator(b *testing.B) {
	for range b.N {
		for range SplitSeq(benchInputHard, "") {
		}
	}
}

func BenchmarkSplitSeqSingleByteSeparator(b *testing.B) {
	for range b.N {
		for range SplitSeq(benchInputHard, "/") {
		}
	}
}

func BenchmarkSplitSeqMultiByteSeparator(b *testing.B) {
	for range b.N {
		for range SplitSeq(benchInputHard, "hello") {
		}
	}
}

func BenchmarkSplitAfterSeqEmptySeparator(b *testing.B) {
	for range b.N {
		for range SplitAfterSeq(benchInputHard, "") {
		}
	}
}

func BenchmarkSplitAfterSeqSingleByteSeparator(b *testing.B) {
	for range b.N {
		for range SplitAfterSeq(benchInputHard, "/") {
		}
	}
}

func BenchmarkSplitAfterSeqMultiByteSeparator(b *testing.B) {
	for range b.N {
		for range SplitAfterSeq(benchInputHard, "hello") {
		}
	}
}
