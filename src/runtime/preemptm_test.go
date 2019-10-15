// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package runtime_test

import (
	"runtime"
	"sync"
	"testing"
)

func TestPreemptM(t *testing.T) {
	var want, got int64
	var wg sync.WaitGroup
	ready := make(chan *runtime.M)
	wg.Add(1)
	go func() {
		runtime.LockOSThread()
		want, got = runtime.WaitForSigusr1(func(mp *runtime.M) {
			ready <- mp
		}, 1e9)
		runtime.UnlockOSThread()
		wg.Done()
	}()
	runtime.SendSigusr1(<-ready)
	wg.Wait()
	if got == -1 {
		t.Fatal("preemptM signal not received")
	} else if want != got {
		t.Fatalf("signal sent to M %d, but received on M %d", want, got)
	}
}
