// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64

package scan_test

import (
	"internal/cpu"
	"internal/runtime/gc/scan"
	"testing"
)

func TestFilterNilRVV(t *testing.T) {
	if !cpu.RISCV64.HasV {
		t.Skip("V extension is required for TestFilterNilRVV")
	}
	runTestFilterNil(t, scan.FilterNilRVV)
}

func filterNilPlatform() func(*uintptr, int32) int32 {
	if cpu.RISCV64.HasV {
		return scan.FilterNilRVV
	}
	return nil
}
