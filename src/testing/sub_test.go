// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"time"
)

func TestTestContext(t *T) {
	const (
		add1 = 0
		done = 1
	)
	// After each of the calls are applied to the context, the
	type call struct {
		typ int // run or done
		// result from applying the call
		running int
		waiting int
		started bool
	}
	testCases := []struct {
		max int
		run []call
	}{{
		max: 1,
		run: []call{
			{typ: add1, running: 1, waiting: 0, started: true},
			{typ: done, running: 0, waiting: 0, started: false},
		},
	}, {
		max: 1,
		run: []call{
			{typ: add1, running: 1, waiting: 0, started: true},
			{typ: add1, running: 1, waiting: 1, started: false},
			{typ: done, running: 1, waiting: 0, started: true},
			{typ: done, running: 0, waiting: 0, started: false},
			{typ: add1, running: 1, waiting: 0, started: true},
		},
	}, {
		max: 3,
		run: []call{
			{typ: add1, running: 1, waiting: 0, started: true},
			{typ: add1, running: 2, waiting: 0, started: true},
			{typ: add1, running: 3, waiting: 0, started: true},
			{typ: add1, running: 3, waiting: 1, started: false},
			{typ: add1, running: 3, waiting: 2, started: false},
			{typ: add1, running: 3, waiting: 3, started: false},
			{typ: done, running: 3, waiting: 2, started: true},
			{typ: add1, running: 3, waiting: 3, started: false},
			{typ: done, running: 3, waiting: 2, started: true},
			{typ: done, running: 3, waiting: 1, started: true},
			{typ: done, running: 3, waiting: 0, started: true},
			{typ: done, running: 2, waiting: 0, started: false},
			{typ: done, running: 1, waiting: 0, started: false},
			{typ: done, running: 0, waiting: 0, started: false},
		},
	}}
	for i, tc := range testCases {
		ctx := &testContext{
			startParallel: make(chan bool),
			maxParallel:   tc.max,
		}
		for j, call := range tc.run {
			doCall := func(f func()) chan bool {
				done := make(chan bool)
				go func() {
					f()
					done <- true
				}()
				return done
			}
			started := false
			switch call.typ {
			case add1:
				signal := doCall(ctx.waitParallel)
				select {
				case <-signal:
					started = true
				case ctx.startParallel <- true:
					<-signal
				}
			case done:
				signal := doCall(ctx.release)
				select {
				case <-signal:
				case <-ctx.startParallel:
					started = true
					<-signal
				}
			}
			if started != call.started {
				t.Errorf("%d:%d:started: got %v; want %v", i, j, started, call.started)
			}
			if ctx.running != call.running {
				t.Errorf("%d:%d:running: got %v; want %v", i, j, ctx.running, call.running)
			}
			if ctx.numWaiting != call.waiting {
				t.Errorf("%d:%d:waiting: got %v; want %v", i, j, ctx.numWaiting, call.waiting)
			}
		}
	}
}

// TODO: remove this stub when API is exposed
func (b *B) Run(name string, f func(b *B)) bool { return b.runBench(name, f) }

func TestBRun(t *T) {
	work := func(b *B) {
		for i := 0; i < b.N; i++ {
			time.Sleep(time.Nanosecond)
		}
	}
	testCases := []struct {
		desc   string
		failed bool
		f      func(*B)
	}{{
		desc: "simulate sequential run of subbenchmarks.",
		f: func(b *B) {
			b.Run("", func(b *B) { work(b) })
			time1 := b.result.NsPerOp()
			b.Run("", func(b *B) { work(b) })
			time2 := b.result.NsPerOp()
			if time1 >= time2 {
				t.Errorf("no time spent in benchmark t1 >= t2 (%d >= %d)", time1, time2)
			}
		},
	}, {
		desc: "bytes set by all benchmarks",
		f: func(b *B) {
			b.Run("", func(b *B) { b.SetBytes(10); work(b) })
			b.Run("", func(b *B) { b.SetBytes(10); work(b) })
			if b.result.Bytes != 20 {
				t.Errorf("bytes: got: %d; want 20", b.result.Bytes)
			}
		},
	}, {
		desc: "bytes set by some benchmarks",
		// In this case the bytes result is meaningless, so it must be 0.
		f: func(b *B) {
			b.Run("", func(b *B) { b.SetBytes(10); work(b) })
			b.Run("", func(b *B) { work(b) })
			b.Run("", func(b *B) { b.SetBytes(10); work(b) })
			if b.result.Bytes != 0 {
				t.Errorf("bytes: got: %d; want 0", b.result.Bytes)
			}
		},
	}, {
		desc:   "failure carried over to root",
		failed: true,
		f:      func(b *B) { b.Fail() },
	}, {
		desc: "memory allocation",
		f: func(b *B) {
			const bufSize = 256
			alloc := func(b *B) {
				var buf [bufSize]byte
				for i := 0; i < b.N; i++ {
					_ = append([]byte(nil), buf[:]...)
				}
			}
			b.Run("", func(b *B) { alloc(b) })
			b.Run("", func(b *B) { alloc(b) })
			if got := b.result.MemAllocs; got != 2 {
				t.Errorf("MemAllocs was %v; want 2", got)
			}
			if got := b.result.MemBytes; got != 2*bufSize {
				t.Errorf("MemBytes was %v; want %v", got, 2*bufSize)
			}
		},
	}}
	for _, tc := range testCases {
		var ok bool
		// This is almost like the Benchmark function, except that we override
		// the benchtime and catch the failure result of the subbenchmark.
		root := &B{
			common: common{
				signal: make(chan bool),
			},
			benchFunc: func(b *B) { ok = b.Run("test", tc.f) }, // Use Run to catch failure.
			benchTime: time.Microsecond,
		}
		root.run()
		if ok != !tc.failed {
			t.Errorf("%s:ok: got %v; want %v", tc.desc, ok, !tc.failed)
		}
		if !ok != root.Failed() {
			t.Errorf("%s:root failed: got %v; want %v", tc.desc, !ok, root.Failed())
		}
		// All tests are run as subtests
		if root.result.N != 1 {
			t.Errorf("%s: N for parent benchmark was %d; want 1", tc.desc, root.result.N)
		}
	}
}
