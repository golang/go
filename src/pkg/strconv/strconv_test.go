// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"runtime"
	. "strconv"
	"testing"
)

var (
	globalBuf [64]byte

	mallocTest = []struct {
		count int
		desc  string
		fn    func()
	}{
		// TODO(bradfitz): this might be 0, once escape analysis is better
		{1, `AppendInt(localBuf[:0], 123, 10)`, func() {
			var localBuf [64]byte
			AppendInt(localBuf[:0], 123, 10)
		}},
		{0, `AppendInt(globalBuf[:0], 123, 10)`, func() { AppendInt(globalBuf[:0], 123, 10) }},
		// TODO(bradfitz): this might be 0, once escape analysis is better
		{1, `AppendFloat(localBuf[:0], 1.23, 'g', 5, 64)`, func() {
			var localBuf [64]byte
			AppendFloat(localBuf[:0], 1.23, 'g', 5, 64)
		}},
		{0, `AppendFloat(globalBuf[:0], 1.23, 'g', 5, 64)`, func() { AppendFloat(globalBuf[:0], 1.23, 'g', 5, 64) }},
	}
)

func TestCountMallocs(t *testing.T) {
	for _, mt := range mallocTest {
		const N = 100
		memstats := new(runtime.MemStats)
		runtime.ReadMemStats(memstats)
		mallocs := 0 - memstats.Mallocs
		for i := 0; i < N; i++ {
			mt.fn()
		}
		runtime.ReadMemStats(memstats)
		mallocs += memstats.Mallocs
		if mallocs/N > uint64(mt.count) {
			t.Errorf("%s: expected %d mallocs, got %d", mt.desc, mt.count, mallocs/N)
		}
	}
}
