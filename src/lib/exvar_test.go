// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exvar

import (
	"exvar";
	"fmt";
	"testing";
)

func TestSimpleCounter(t *testing.T) {
	// Unknown exvar should be zero, and return !ok.
	x, ok := Get("requests");
	if x != 0 || ok {
		t.Errorf("Get(nonexistent) = (%v, %v), want (%v, %v)",
		         x, ok, 0, false)
	}

	Increment("requests", 1);
	Increment("requests", 3);
	x, ok = Get("requests");
	if x != 4 || !ok {
		t.Errorf("Get('requests') = (%v, %v), want (%v, %v)",
		         x, ok, 4, true)
	}

	out := String();
	if out != "requests 4\n" {
		t.Errorf("String() = \"%v\", want \"requests 4\n\"",
		         out);
	}
}

func hammer(name string, total int, done chan <- int) {
	for i := 0; i < total; i++ {
		Increment(name, 1)
	}
	done <- 1
}

func TestHammer(t *testing.T) {
	Set("hammer-times", 0);
	sync := make(chan int);
	hammer_times := int(1e5);
	go hammer("hammer-times", hammer_times, sync);
	go hammer("hammer-times", hammer_times, sync);
	<-sync;
	<-sync;
	if final, ok := Get("hammer-times"); final != 2 * hammer_times {
		t.Errorf("hammer-times = %v, want %v", final, 2 * hammer_times)
	}
}
