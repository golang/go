// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime_test

import (
	"runtime"
	"testing"
	"time"
)

func TestLogosRegionParkReleasesEmptyBehindLive(t *testing.T) {
	inFreshGoroutine(t, func() {
		h, gen := runtime.LogosRegionOpen()
		capacity := uintptr(runtime.LogosArenaCapacity)
		maxBytes := uintptr(runtime.LogosRegionMaxBytes)
		runtime.LogosRegionAllocNoscan(h, gen, maxBytes, 8)
		runtime.LogosRegionAllocNoscan(h, gen, capacity-maxBytes, 8)
		runtime.LogosRegionAllocNoscan(h, gen, 1, 8)
		live := runtime.LogosRegionAllocNoscan(h, gen, 64, 8)
		fillBytes(live, 64, 0xA5)
		chunks := runtime.LogosRegionChunks()
		if len(chunks) != 2 {
			t.Fatalf("region chunks = %d, want 2", len(chunks))
		}
		if !runtime.LogosRegionEmptyChunk(chunks[1]) {
			t.Fatal("could not empty the older region chunk")
		}
		wantBytes := runtime.LogosRegionBytes()
		release := make(chan struct{})
		go func() {
			time.Sleep(50 * time.Millisecond)
			close(release)
		}()
		<-release
		chunks = runtime.LogosRegionChunks()
		if len(chunks) != 1 {
			t.Fatalf("region chunks after park = %d, want 1 (live only)", len(chunks))
		}
		if !checkBytes(live, 64, 0xA5) {
			t.Fatal("live region object moved or changed over park")
		}
		if got := runtime.LogosRegionBytes(); got != wantBytes {
			t.Fatalf("RegionBytes = %d after park, want %d", got, wantBytes)
		}
		runtime.LogosRegionClose(h, gen)
	})
}
