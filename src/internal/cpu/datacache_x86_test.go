// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64

package cpu_test

import (
	"internal/cpu"
	"testing"
)

// Tests fetching data cache sizes. This test only checks that DataCacheSizes
// won't explode. Otherwise it's just informational, and dumps the current
// data cache sizes.
func TestDataCacheSizes(t *testing.T) {
	// N.B. Don't try to check these values because we don't know what
	// kind of environment we're running in. We don't want this test to
	// fail on some random x86 chip that happens to not support the right
	// CPUID bits for some reason.
	caches := cpu.DataCacheSizes()
	for i, size := range caches {
		t.Logf("L%d: %d", i+1, size)
	}
}
