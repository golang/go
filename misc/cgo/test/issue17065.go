// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
// Test that C symbols larger than a page play nicely with the race detector.
// See issue 17065.

int ii[65537];
*/
import "C"

import (
	"runtime"
	"testing"
)

var sink C.int

func test17065(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("broken on darwin; issue 17065")
	}
	for i := range C.ii {
		sink = C.ii[i]
	}
}
