// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes_test

import (
	. "bytes"
	"testing"
)

func BenchmarkSplitSeqEmptySeparator(b *testing.B) {
	for range b.N {
		for range SplitSeq(benchInputHard, nil) {
		}
	}
}

func BenchmarkSplitSeqSingleByteSeparator(b *testing.B) {
	sep := []byte("/")
	for range b.N {
		for range SplitSeq(benchInputHard, sep) {
		}
	}
}

func BenchmarkSplitSeqMultiByteSeparator(b *testing.B) {
	sep := []byte("hello")
	for range b.N {
		for range SplitSeq(benchInputHard, sep) {
		}
	}
}

func BenchmarkSplitAfterSeqEmptySeparator(b *testing.B) {
	for range b.N {
		for range SplitAfterSeq(benchInputHard, nil) {
		}
	}
}

func BenchmarkSplitAfterSeqSingleByteSeparator(b *testing.B) {
	sep := []byte("/")
	for range b.N {
		for range SplitAfterSeq(benchInputHard, sep) {
		}
	}
}

func BenchmarkSplitAfterSeqMultiByteSeparator(b *testing.B) {
	sep := []byte("hello")
	for range b.N {
		for range SplitAfterSeq(benchInputHard, sep) {
		}
	}
}
