// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"runtime"
	. "strconv"
	"strings"
	"testing"
)

var (
	globalBuf [64]byte
	nextToOne = "1.00000000000000011102230246251565404236316680908203125" + strings.Repeat("0", 10000) + "1"

	mallocTest = []struct {
		count int
		desc  string
		fn    func()
	}{
		{0, `AppendInt(localBuf[:0], 123, 10)`, func() {
			var localBuf [64]byte
			AppendInt(localBuf[:0], 123, 10)
		}},
		{0, `AppendInt(globalBuf[:0], 123, 10)`, func() { AppendInt(globalBuf[:0], 123, 10) }},
		{0, `AppendFloat(localBuf[:0], 1.23, 'g', 5, 64)`, func() {
			var localBuf [64]byte
			AppendFloat(localBuf[:0], 1.23, 'g', 5, 64)
		}},
		{0, `AppendFloat(globalBuf[:0], 1.23, 'g', 5, 64)`, func() { AppendFloat(globalBuf[:0], 1.23, 'g', 5, 64) }},
		{0, `ParseFloat("123.45", 64)`, func() { ParseFloat("123.45", 64) }},
		{0, `ParseFloat("123.456789123456789", 64)`, func() { ParseFloat("123.456789123456789", 64) }},
		{0, `ParseFloat("1.000000000000000111022302462515654042363166809082031251", 64)`, func() {
			ParseFloat("1.000000000000000111022302462515654042363166809082031251", 64)
		}},
		{0, `ParseFloat("1.0000000000000001110223024625156540423631668090820312500...001", 64)`, func() {
			ParseFloat(nextToOne, 64)
		}},
	}
)

func TestCountMallocs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	if runtime.GOMAXPROCS(0) > 1 {
		t.Skip("skipping; GOMAXPROCS>1")
	}
	for _, mt := range mallocTest {
		allocs := testing.AllocsPerRun(100, mt.fn)
		if max := float64(mt.count); allocs > max {
			t.Errorf("%s: %v allocs, want <=%v", mt.desc, allocs, max)
		}
	}
}
