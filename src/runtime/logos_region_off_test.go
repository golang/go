// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.logosarena

package runtime_test

import (
	"runtime"
	"testing"
)

func TestLogosRegionOpenOffReturnsNil(t *testing.T) {
	h, gen := runtime.LogosRegionOpen()
	if h != nil || gen != 0 {
		t.Fatalf("open with experiment off = %v, %d, want nil, 0", h, gen)
	}
	if n := runtime.LogosRegionBytes(); n != 0 {
		t.Fatalf("region bytes with experiment off = %d, want 0", n)
	}
}
