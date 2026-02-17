// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64

package scan_test

import (
	"internal/runtime/gc/scan"
	"testing"
)

func TestScanSpanPackedAVX512(t *testing.T) {
	if !scan.CanAVX512() {
		t.Skip("no AVX512")
	}
	testScanSpanPacked(t, scan.ScanSpanPackedAVX512)
}
