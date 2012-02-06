// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
)

func TestGcSys(t *testing.T) {
	memstats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(memstats)
	sys := memstats.Sys

	for i := 0; i < 1000000; i++ {
		workthegc()
	}

	// Should only be using a few MB.
	runtime.ReadMemStats(memstats)
	if sys > memstats.Sys {
		sys = 0
	} else {
		sys = memstats.Sys - sys
	}
	t.Logf("used %d extra bytes", sys)
	if sys > 4<<20 {
		t.Fatalf("using too much memory: %d bytes", sys)
	}
}

func workthegc() []byte {
	return make([]byte, 1029)
}
