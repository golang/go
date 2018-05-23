// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu_test

import (
	. "internal/cpu"
	"runtime"
	"testing"
)

func TestARM64minimalFeatures(t *testing.T) {
	switch runtime.GOOS {
	case "linux", "android":
	default:
		t.Skipf("%s/arm64 is not supported", runtime.GOOS)
	}

	if !ARM64.HasASIMD {
		t.Fatalf("HasASIMD expected true, got false")
	}
	if !ARM64.HasFP {
		t.Fatalf("HasFP expected true, got false")
	}
}
