// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Wrong type of constant with GCC 8 and newer.

package cgotest

// const unsigned long long int issue26066 = (const unsigned long long) -1;
import "C"

import "testing"

func test26066(t *testing.T) {
	var i = int64(C.issue26066)
	if i != -1 {
		t.Errorf("got %d, want -1", i)
	}
}
