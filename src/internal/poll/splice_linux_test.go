// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"internal/poll"
	"internal/syscall/unix"
	"runtime"
	"syscall"
	"testing"
	"time"
)

// checkPipes returns true if all pipes are closed properly, false otherwise.
func checkPipes(fds []int) bool {
	for _, fd := range fds {
		// Check if each pipe fd has been closed.
		_, _, errno := syscall.Syscall(unix.FcntlSyscall, uintptr(fd), syscall.F_GETPIPE_SZ, 0)
		if errno == 0 {
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
		_, pwfd := poll.GetPipeFds(p)
		fds = append(fds, pwfd)
		ps = append(ps, p)
	}
	for _, p = range ps {
		poll.PutPipe(p)
	}
	ps = nil
	p = nil

	// Exploit the timeout of "go test" as a timer for the subsequent verification.
	timeout := 5 * time.Minute
	if deadline, ok := t.Deadline(); ok {
		timeout = deadline.Sub(time.Now())
		timeout -= timeout / 10 // Leave 10% headroom for cleanup.
	}
	expiredTime := time.NewTimer(timeout)
	defer expiredTime.Stop()

	// Trigger garbage collection repeatedly, waiting for all pipes in sync.Pool
	// to either be deallocated and closed, or to time out.
	for {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
		if checkPipes(fds) {
			break
		}
		select {
		case <-expiredTime.C:
			t.Fatal("at least one pipe is still open")
		default:
		}
	}
}

func BenchmarkSplicePipe(b *testing.B) {
	b.Run("SplicePipeWithPool", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p, _, err := poll.GetPipe()
			if err != nil {
				continue
			}
			poll.PutPipe(p)
		}
	})
	b.Run("SplicePipeWithoutPool", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p := poll.NewPipe()
			if p == nil {
				b.Skip("newPipe returned nil")
			}
			poll.DestroyPipe(p)
		}
	})
}

func BenchmarkSplicePipePoolParallel(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p, _, err := poll.GetPipe()
			if err != nil {
				continue
			}
			poll.PutPipe(p)
		}
	})
}

func BenchmarkSplicePipeNativeParallel(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p := poll.NewPipe()
			if p == nil {
				b.Skip("newPipe returned nil")
			}
			poll.DestroyPipe(p)
		}
	})
}
