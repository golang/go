// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"regexp"
	"strings"
	"sync/atomic"
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

func TestTRun(t *T) {
	realTest := t
	testCases := []struct {
		desc   string
		ok     bool
		maxPar int
		chatty bool
		output string
		f      func(*T)
	}{{
		desc:   "failnow skips future sequential and parallel tests at same level",
		ok:     false,
		maxPar: 1,
		output: `
--- FAIL: failnow skips future sequential and parallel tests at same level (0.00s)
    --- FAIL: failnow skips future sequential and parallel tests at same level/#00 (0.00s)
    `,
		f: func(t *T) {
			ranSeq := false
			ranPar := false
			t.Run("", func(t *T) {
				t.Run("par", func(t *T) {
					t.Parallel()
					ranPar = true
				})
				t.Run("seq", func(t *T) {
					ranSeq = true
				})
				t.FailNow()
				t.Run("seq", func(t *T) {
					realTest.Error("test must be skipped")
				})
				t.Run("par", func(t *T) {
					t.Parallel()
					realTest.Error("test must be skipped.")
				})
			})
			if !ranPar {
				realTest.Error("parallel test was not run")
			}
			if !ranSeq {
				realTest.Error("sequential test was not run")
			}
		},
	}, {
		desc:   "failure in parallel test propagates upwards",
		ok:     false,
		maxPar: 1,
		output: `
--- FAIL: failure in parallel test propagates upwards (0.00s)
    --- FAIL: failure in parallel test propagates upwards/#00 (0.00s)
        --- FAIL: failure in parallel test propagates upwards/#00/par (0.00s)
		`,
		f: func(t *T) {
			t.Run("", func(t *T) {
				t.Parallel()
				t.Run("par", func(t *T) {
					t.Parallel()
					t.Fail()
				})
			})
		},
	}, {
		desc:   "skipping without message, chatty",
		ok:     true,
		chatty: true,
		output: `
=== RUN   skipping without message, chatty
--- SKIP: skipping without message, chatty (0.00s)`,
		f: func(t *T) { t.SkipNow() },
	}, {
		desc: "skipping without message, not chatty",
		ok:   true,
		f:    func(t *T) { t.SkipNow() },
	}, {
		desc: "skipping after error",
		output: `
--- FAIL: skipping after error (0.00s)
	sub_test.go:nnn: an error
	sub_test.go:nnn: skipped`,
		f: func(t *T) {
			t.Error("an error")
			t.Skip("skipped")
		},
	}, {
		desc:   "use Run to locally synchronize parallelism",
		ok:     true,
		maxPar: 1,
		f: func(t *T) {
			var count uint32
			t.Run("waitGroup", func(t *T) {
				for i := 0; i < 4; i++ {
					t.Run("par", func(t *T) {
						t.Parallel()
						atomic.AddUint32(&count, 1)
					})
				}
			})
			if count != 4 {
				t.Errorf("count was %d; want 4", count)
			}
		},
	}, {
		desc:   "run no more than *parallel tests concurrently",
		ok:     true,
		maxPar: 4,
		f: func(t *T) {
			max := 0
			in := make(chan int)
			out := make(chan int)
			ctx := t.context
			t.Run("wait", func(t *T) {
				t.Run("controller", func(t *T) {
					// Verify sequential tests don't skew counts.
					t.Run("seq1", func(t *T) {})
					t.Run("seq2", func(t *T) {})
					t.Run("seq3", func(t *T) {})
					t.Parallel()
					for i := 0; i < 80; i++ {
						ctx.mu.Lock()
						if ctx.running > max {
							max = ctx.running
						}
						ctx.mu.Unlock()
						<-in
						// force a minimum to avoid a race, although it works
						// without it.
						if i >= ctx.maxParallel-2 { // max - this - 1
							out <- i
						}
					}
					close(out)
				})
				// Ensure we don't exceed the maximum even with nested parallelism.
				for i := 0; i < 2; i++ {
					t.Run("", func(t *T) {
						t.Parallel()
						for j := 0; j < 40; j++ {
							t.Run("", func(t *T) {
								t.Run("seq1", func(t *T) {})
								t.Run("seq2", func(t *T) {})
								t.Parallel()
								in <- j
								<-out
							})
						}
					})
				}
			})
			if max != ctx.maxParallel {
				realTest.Errorf("max: got %d; want: %d", max, ctx.maxParallel)
			}
		},
	}, {
		desc: "alternate sequential and parallel",
		// Sequential tests should partake in the counting of running threads.
		// Otherwise, if one runs parallel subtests in sequential tests that are
		// itself subtests of parallel tests, the counts can get askew.
		ok:     true,
		maxPar: 1,
		f: func(t *T) {
			t.Run("a", func(t *T) {
				t.Parallel()
				t.Run("b", func(t *T) {
					// Sequential: ensure running count is decremented.
					t.Run("c", func(t *T) {
						t.Parallel()
					})

				})
			})
		},
	}, {
		desc: "alternate sequential and parallel 2",
		// Sequential tests should partake in the counting of running threads.
		// Otherwise, if one runs parallel subtests in sequential tests that are
		// itself subtests of parallel tests, the counts can get askew.
		ok:     true,
		maxPar: 2,
		f: func(t *T) {
			for i := 0; i < 2; i++ {
				t.Run("a", func(t *T) {
					t.Parallel()
					time.Sleep(time.Nanosecond)
					for i := 0; i < 2; i++ {
						t.Run("b", func(t *T) {
							time.Sleep(time.Nanosecond)
							for i := 0; i < 2; i++ {
								t.Run("c", func(t *T) {
									t.Parallel()
									time.Sleep(time.Nanosecond)
								})
							}

						})
					}
				})
			}
		},
	}, {
		desc:   "stress test",
		ok:     true,
		maxPar: 4,
		f: func(t *T) {
			t.Parallel()
			for i := 0; i < 12; i++ {
				t.Run("a", func(t *T) {
					t.Parallel()
					time.Sleep(time.Nanosecond)
					for i := 0; i < 12; i++ {
						t.Run("b", func(t *T) {
							time.Sleep(time.Nanosecond)
							for i := 0; i < 12; i++ {
								t.Run("c", func(t *T) {
									t.Parallel()
									time.Sleep(time.Nanosecond)
									t.Run("d1", func(t *T) {})
									t.Run("d2", func(t *T) {})
									t.Run("d3", func(t *T) {})
									t.Run("d4", func(t *T) {})
								})
							}
						})
					}
				})
			}
		},
	}, {
		desc:   "skip output",
		ok:     true,
		maxPar: 4,
		f: func(t *T) {
			t.Skip()
		},
	}}
	for _, tc := range testCases {
		ctx := newTestContext(tc.maxPar, newMatcher(regexp.MatchString, "", ""))
		buf := &bytes.Buffer{}
		root := &T{
			common: common{
				signal: make(chan bool),
				name:   "Test",
				w:      buf,
				chatty: tc.chatty,
			},
			context: ctx,
		}
		ok := root.Run(tc.desc, tc.f)
		ctx.release()

		if ok != tc.ok {
			t.Errorf("%s:ok: got %v; want %v", tc.desc, ok, tc.ok)
		}
		if ok != !root.Failed() {
			t.Errorf("%s:root failed: got %v; want %v", tc.desc, !ok, root.Failed())
		}
		if ctx.running != 0 || ctx.numWaiting != 0 {
			t.Errorf("%s:running and waiting non-zero: got %d and %d", tc.desc, ctx.running, ctx.numWaiting)
		}
		got := sanitizeLog(buf.String())
		want := sanitizeLog(tc.output)
		if got != want {
			t.Errorf("%s:ouput:\ngot:\n%s\nwant:\n%s", tc.desc, got, want)
		}
	}
}

func TestBRun(t *T) {
	work := func(b *B) {
		for i := 0; i < b.N; i++ {
			time.Sleep(time.Nanosecond)
		}
	}
	testCases := []struct {
		desc   string
		failed bool
		chatty bool
		output string
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
		output: "--- FAIL: root",
		f:      func(b *B) { b.Fail() },
	}, {
		desc:   "skipping without message, chatty",
		chatty: true,
		output: "--- SKIP: root",
		f:      func(b *B) { b.SkipNow() },
	}, {
		desc:   "skipping with message, chatty",
		chatty: true,
		output: `
--- SKIP: root
	sub_test.go:: skipping`,
		f: func(b *B) { b.Skip("skipping") },
	}, {
		desc: "skipping without message, not chatty",
		f:    func(b *B) { b.SkipNow() },
	}, {
		desc:   "skipping after error",
		failed: true,
		output: `
--- FAIL: root
	sub_test.go:nnn: an error
	sub_test.go:nnn: skipped`,
		f: func(b *B) {
			b.Error("an error")
			b.Skip("skipped")
		},
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
			// runtime.MemStats sometimes reports more allocations than the
			// benchmark is responsible for. Luckily the point of this test is
			// to ensure that the results are not underreported, so we can
			// simply verify the lower bound.
			if got := b.result.MemAllocs; got < 2 {
				t.Errorf("MemAllocs was %v; want 2", got)
			}
			if got := b.result.MemBytes; got < 2*bufSize {
				t.Errorf("MemBytes was %v; want %v", got, 2*bufSize)
			}
		},
	}}
	for _, tc := range testCases {
		var ok bool
		buf := &bytes.Buffer{}
		// This is almost like the Benchmark function, except that we override
		// the benchtime and catch the failure result of the subbenchmark.
		root := &B{
			common: common{
				signal: make(chan bool),
				name:   "root",
				w:      buf,
				chatty: tc.chatty,
			},
			benchFunc: func(b *B) { ok = b.Run("test", tc.f) }, // Use Run to catch failure.
			benchTime: time.Microsecond,
		}
		root.runN(1)
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
		got := sanitizeLog(buf.String())
		want := sanitizeLog(tc.output)
		if got != want {
			t.Errorf("%s:ouput:\ngot:\n%s\nwant:\n%s", tc.desc, got, want)
		}
	}
}

// sanitizeLog removes line numbers from log entries.
func sanitizeLog(s string) string {
	s = strings.TrimSpace(s)
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		p := strings.IndexByte(line, ':')
		if p > 0 && line[p+4] == ':' { // assuming 3-digit file positions
			lines[i] = line[:p+1] + line[p+4:]
		}
	}
	return strings.Join(lines, "\n")
}

func TestBenchmarkOutput(t *T) {
	// Ensure Benchmark initialized common.w by invoking it with an error and
	// normal case.
	Benchmark(func(b *B) { b.Error("do not print this output") })
	Benchmark(func(b *B) {})
}
