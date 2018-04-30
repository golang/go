// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "C"
import "testing"

func issue25143sum(ns ...C.int) C.int {
	total := C.int(0)
	for _, n := range ns {
		total += n
	}
	return total
}

func test25143(t *testing.T) {
	if got, want := issue25143sum(1, 2, 3), C.int(6); got != want {
		t.Errorf("issue25143sum(1, 2, 3) == %v, expected %v", got, want)
	}
}
