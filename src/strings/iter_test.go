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

func findKvBySplit(s string, k string) string {
	for _, kv := range Split(s, ",") {
		if HasPrefix(kv, k) {
			return kv
		}
	}
	return ""
}

func findKvBySplitSeq(s string, k string) string {
	for kv := range SplitSeq(s, ",") {
		if HasPrefix(kv, k) {
			return kv
		}
	}
	return ""
}

var testSplitString = "k1=v1,k2=v2,k3=v3,k4=v4"

func BenchmarkSplitAndSplitSeqKeyFound(b *testing.B) {
	b.Run("findKvBySplit", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for b.Loop() {
			findKvBySplit(testSplitString, "k3")
		}
	})

	b.Run("findKvBySplitSeq", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for b.Loop() {
			findKvBySplitSeq(testSplitString, "k3")
		}
	})
}

func BenchmarkSplitAndSplitSeqKeyNotFound(b *testing.B) {
	b.Run("findKvBySplit", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for b.Loop() {
			findKvBySplit(testSplitString, "notFoundKey")
		}
	})

	b.Run("findKvBySplitSeq", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for b.Loop() {
			findKvBySplitSeq(testSplitString, "notFoundKey")
		}
	})
}
