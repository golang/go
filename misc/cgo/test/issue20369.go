// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#define UINT64_MAX        18446744073709551615ULL
*/
import "C"
import (
	"math"
	"testing"
)

func test20369(t *testing.T) {
	if C.UINT64_MAX != math.MaxUint64 {
		t.Fatalf("got %v, want %v", uint64(C.UINT64_MAX), uint64(math.MaxUint64))
	}
}
