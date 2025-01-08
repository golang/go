// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"crypto/internal/boring"
	"internal/asan"
	"internal/msan"
	"internal/race"
	"internal/testenv"
	"runtime"
	"testing"
)

// SkipTestAllocations skips the test if there are any factors that interfere
// with allocation optimizations.
func SkipTestAllocations(t *testing.T) {
	// Go+BoringCrypto uses cgo.
	if boring.Enabled {
		t.Skip("skipping allocations test with BoringCrypto")
	}

	// The sanitizers sometimes cause allocations.
	if race.Enabled || msan.Enabled || asan.Enabled {
		t.Skip("skipping allocations test with sanitizers")
	}

	// The plan9 crypto/rand allocates.
	if runtime.GOOS == "plan9" {
		t.Skip("skipping allocations test on plan9")
	}

	// s390x deviates from other assembly implementations and is very hard to
	// test due to the lack of LUCI builders. See #67307.
	if runtime.GOARCH == "s390x" {
		t.Skip("skipping allocations test on s390x")
	}

	// Some APIs rely on inliner and devirtualization to allocate on the stack.
	testenv.SkipIfOptimizationOff(t)
}
