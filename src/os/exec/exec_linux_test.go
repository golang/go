// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && cgo

// On systems that use glibc, calling malloc can create a new arena,
// and creating a new arena can read /sys/devices/system/cpu/online.
// If we are using cgo, we will call malloc when creating a new thread.
// That can break TestExtraFiles if we create a new thread that creates
// a new arena and opens the /sys file while we are checking for open
// file descriptors. Work around the problem by creating threads up front.
// See issue 25628.

package exec_test

import (
	"os"
	"sync"
	"syscall"
	"time"
)

func init() {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	// Start some threads. 10 is arbitrary but intended to be enough
	// to ensure that the code won't have to create any threads itself.
	// In particular this should be more than the number of threads
	// the garbage collector might create.
	const threads = 10

	var wg sync.WaitGroup
	wg.Add(threads)
	ts := syscall.NsecToTimespec((100 * time.Microsecond).Nanoseconds())
	for i := 0; i < threads; i++ {
		go func() {
			defer wg.Done()
			syscall.Nanosleep(&ts, nil)
		}()
	}
	wg.Wait()
}
