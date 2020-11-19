// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"internal/poll"
	"runtime"
	"syscall"
	"testing"
	"time"
)

// checkPipes returns true if all pipes are closed properly, false otherwise.
func checkPipes(fds []int) bool {
	for _, fd := range fds {
		// Check if each pipe fd has been closed.
		err := syscall.FcntlFlock(uintptr(fd), syscall.F_GETFD, nil)
		if err == nil {
			return false
		}
	}
	return true
}

func TestSplicePipePool(t *testing.T) {
	const N = 64
	var (
		p   *poll.SplicePipe
		ps  []*poll.SplicePipe
		fds []int
		err error
	)
	for i := 0; i < N; i++ {
		p, _, err = poll.GetPipe()
		if err != nil {
			t.Skip("failed to create pipe, skip this test")
		}
		prfd, pwfd := poll.GetPipeFds(p)
		fds = append(fds, prfd, pwfd)
		ps = append(ps, p)
	}
	for _, p = range ps {
		poll.PutPipe(p)
	}
	ps = nil

	var ok bool
	// Trigger garbage collection to free the pipes in sync.Pool and check whether or not
	// those pipe buffers have been closed as we expected.
	for i := 0; i < 5; i++ {
		runtime.GC()
		time.Sleep(time.Duration(i*100+10) * time.Millisecond)
		if ok = checkPipes(fds); ok {
			break
		}
	}

	if !ok {
		t.Fatal("at least one pipe is still open")
	}
}

func BenchmarkSplicePipe(b *testing.B) {
	b.Run("SplicePipeWithPool", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p, _, _ := poll.GetPipe()
			poll.PutPipe(p)
		}
	})
	b.Run("SplicePipeWithoutPool", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p := poll.NewPipe()
			poll.DestroyPipe(p)
		}
	})
}

func BenchmarkSplicePipePoolParallel(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p, _, _ := poll.GetPipe()
			poll.PutPipe(p)
		}
	})
}

func BenchmarkSplicePipeNativeParallel(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p := poll.NewPipe()
			poll.DestroyPipe(p)
		}
	})
}
