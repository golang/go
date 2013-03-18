// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux freebsd

package runtime_test

import (
	. "runtime"
	"testing"
	"time"
)

func TestFutexsleep(t *testing.T) {
	ch := make(chan bool, 1)
	var dummy uint32
	start := time.Now()
	go func() {
		Entersyscall()
		Futexsleep(&dummy, 0, (1<<31+100)*1e9)
		Exitsyscall()
		ch <- true
	}()
	select {
	case <-ch:
		t.Errorf("futexsleep finished early after %s!", time.Since(start))
	case <-time.After(time.Second):
		Futexwakeup(&dummy, 1)
	}
}
