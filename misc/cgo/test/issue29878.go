// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// #include <stdint.h>
// uint64_t issue29878exported(int8_t);  // prototype must match
// int16_t issue29878function(uint32_t arg) { return issue29878exported(arg); }
import "C"

import "testing"

func test29878(t *testing.T) {
	const arg uint32 = 123                    // fits into all integer types
	var ret int16 = C.issue29878function(arg) // no conversions needed
	if int64(ret) != int64(arg) {
		t.Errorf("return value unexpected: got %d, want %d", ret, arg)
	}
}
