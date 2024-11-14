// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique

import (
	"fmt"
	"runtime"
	"testing"
)

func BenchmarkMake(b *testing.B) {
	benchmarkMake(b, []string{"foo"})
}

func BenchmarkMakeMany(b *testing.B) {
	benchmarkMake(b, testData[:])
}

func BenchmarkMakeManyMany(b *testing.B) {
	benchmarkMake(b, testDataLarge[:])
}

func benchmarkMake(b *testing.B, testData []string) {
	handles := make([]Handle[string], 0, len(testData))
	for i := range testData {
		handles = append(handles, Make(testData[i]))
	}

	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func { pb ->
		i := 0
		for pb.Next() {
			_ = Make(testData[i])
			i++
			if i >= len(testData) {
				i = 0
			}
		}
	})

	b.StopTimer()

	runtime.GC()
	runtime.GC()
}

var (
	testData      [128]string
	testDataLarge [128 << 10]string
)

func init() {
	for i := range testData {
		testData[i] = fmt.Sprintf("%b", i)
	}
	for i := range testDataLarge {
		testDataLarge[i] = fmt.Sprintf("%b", i)
	}
}
