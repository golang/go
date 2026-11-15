// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64 || 386 || ppc64le || s390x

package rt

import (
	"runtime"
	"sync/atomic"
	"testing"
)

func pinnedViewasCleanup(cleaned *atomic.Bool) (*viewasHeader, viewasHeader) {
	backing := new([64]byte)
	for i := range backing {
		backing[i] = byte(i + 17)
	}
	runtime.AddCleanup(backing, func(b *atomic.Bool) { b.Store(true) }, cleaned)
	return ViewAs[viewasHeader](backing[:16]), viewasNative(backing[:16])
}

func TestViewAsKeepsAliveCleanup(t *testing.T) {
	cleaned := new(atomic.Bool)
	v, want := pinnedViewasCleanup(cleaned)
	for range 20 {
		runtime.GC()
		if cleaned.Load() {
			t.Fatalf("cleanup ran while view live")
		}
	}
	if *v != want {
		t.Fatalf("view after GC = %+v, want %+v", *v, want)
	}
	runtime.KeepAlive(v)
	v = nil
	for range 200 {
		runtime.GC()
		if cleaned.Load() {
			return
		}
	}
	t.Fatalf("cleanup never ran after view dropped")
}
