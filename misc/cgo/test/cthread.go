// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// extern void doAdd(int, int);
import "C"

import (
	"sync"
	"testing"
)

var sum struct {
	sync.Mutex
	i int
}

//export Add
func Add(x int) {
	defer func() {
		recover()
	}()
	sum.Lock()
	sum.i += x
	sum.Unlock()
	var p *int
	*p = 2
}

func testCthread(t *testing.T) {
	sum.i = 0
	C.doAdd(10, 6)

	want := 10 * (10 - 1) / 2 * 6
	if sum.i != want {
		t.Fatalf("sum=%d, want %d", sum.i, want)
	}
}
