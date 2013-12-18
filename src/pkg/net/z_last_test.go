// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"testing"
)

func TestDNSThreadLimit(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	const N = 10000
	c := make(chan int, N)
	for i := 0; i < N; i++ {
		go func(i int) {
			LookupIP(fmt.Sprintf("%d.net-test.golang.org", i))
			c <- 1
		}(i)
	}
	// Don't bother waiting for the stragglers; stop at 0.9 N.
	for i := 0; i < N*9/10; i++ {
		if i%100 == 0 {
			//println("TestDNSThreadLimit:", i)
		}
		<-c
	}

	// If we're still here, it worked.
}
