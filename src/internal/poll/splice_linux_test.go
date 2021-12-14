// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"internal/poll"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var closeHook atomic.Value // func(fd int)

func init() {
	closeFunc := poll.CloseFunc
	poll.CloseFunc = func(fd int) (err error) {
		if v := closeHook.Load(); v != nil {
			if hook := v.(func(int)); hook != nil {
				hook(fd)
			}
		}
		return closeFunc(fd)
	}
}

func TestSplicePipePool(t *testing.T) {
	const N = 64
	var (
		p          *poll.SplicePipe
		ps         []*poll.SplicePipe
		allFDs     []int
		pendingFDs sync.Map // fd → struct{}{}
		err        error
	)

	closeHook.Store(func(fd int) { pendingFDs.Delete(fd) })
	t.Cleanup(func() { closeHook.Store((func(int))(nil)) })

	for i := 0; i < N; i++ {
		p, _, err = poll.GetPipe()
		if err != nil {
			t.Skipf("failed to create pipe due to error(%v), skip this test", err)
		}
		_, pwfd := poll.GetPipeFds(p)
		allFDs = append(allFDs, pwfd)
		pendingFDs.Store(pwfd, struct{}{})
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

		// Detect whether all pipes are closed properly.
		var leakedFDs []int
		pendingFDs.Range(func(k, v interface{}) bool {
			leakedFDs = append(leakedFDs, k.(int))
			return true
		})
		if len(leakedFDs) == 0 {
			break
		}

		select {
		case <-expiredTime.C:
			t.Logf("all descriptors: %v", allFDs)
			t.Fatalf("leaked descriptors: %v", leakedFDs)
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
