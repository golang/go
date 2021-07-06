// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"fmt"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func init() {
	// Make benchmark tests run 10x faster.
	benchTime.d = 100 * time.Millisecond
}

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
--- FAIL: failnow skips future sequential and parallel tests at same level (N.NNs)
    --- FAIL: failnow skips future sequential and parallel tests at same level/#00 (N.NNs)
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
--- FAIL: failure in parallel test propagates upwards (N.NNs)
    --- FAIL: failure in parallel test propagates upwards/#00 (N.NNs)
        --- FAIL: failure in parallel test propagates upwards/#00/par (N.NNs)
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
--- SKIP: skipping without message, chatty (N.NNs)`,
		f: func(t *T) { t.SkipNow() },
	}, {
		desc:   "chatty with recursion",
		ok:     true,
		chatty: true,
		output: `
=== RUN   chatty with recursion
=== RUN   chatty with recursion/#00
=== RUN   chatty with recursion/#00/#00
--- PASS: chatty with recursion (N.NNs)
    --- PASS: chatty with recursion/#00 (N.NNs)
        --- PASS: chatty with recursion/#00/#00 (N.NNs)`,
		f: func(t *T) {
			t.Run("", func(t *T) {
				t.Run("", func(t *T) {})
			})
		},
	}, {
		desc: "skipping without message, not chatty",
		ok:   true,
		f:    func(t *T) { t.SkipNow() },
	}, {
		desc: "skipping after error",
		output: `
--- FAIL: skipping after error (N.NNs)
    sub_test.go:NNN: an error
    sub_test.go:NNN: skipped`,
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
	}, {
		desc: "subtest calls error on parent",
		ok:   false,
		output: `
--- FAIL: subtest calls error on parent (N.NNs)
    sub_test.go:NNN: first this
    sub_test.go:NNN: and now this!
    sub_test.go:NNN: oh, and this too`,
		maxPar: 1,
		f: func(t *T) {
			t.Errorf("first this")
			outer := t
			t.Run("", func(t *T) {
				outer.Errorf("and now this!")
			})
			t.Errorf("oh, and this too")
		},
	}, {
		desc: "subtest calls fatal on parent",
		ok:   false,
		output: `
--- FAIL: subtest calls fatal on parent (N.NNs)
    sub_test.go:NNN: first this
    sub_test.go:NNN: and now this!
    --- FAIL: subtest calls fatal on parent/#00 (N.NNs)
        testing.go:NNN: test executed panic(nil) or runtime.Goexit: subtest may have called FailNow on a parent test`,
		maxPar: 1,
		f: func(t *T) {
			outer := t
			t.Errorf("first this")
			t.Run("", func(t *T) {
				outer.Fatalf("and now this!")
			})
			t.Errorf("Should not reach here.")
		},
	}, {
		desc: "subtest calls error on ancestor",
		ok:   false,
		output: `
--- FAIL: subtest calls error on ancestor (N.NNs)
    sub_test.go:NNN: Report to ancestor
    --- FAIL: subtest calls error on ancestor/#00 (N.NNs)
        sub_test.go:NNN: Still do this
    sub_test.go:NNN: Also do this`,
		maxPar: 1,
		f: func(t *T) {
			outer := t
			t.Run("", func(t *T) {
				t.Run("", func(t *T) {
					outer.Errorf("Report to ancestor")
				})
				t.Errorf("Still do this")
			})
			t.Errorf("Also do this")
		},
	}, {
		desc: "subtest calls fatal on ancestor",
		ok:   false,
		output: `
--- FAIL: subtest calls fatal on ancestor (N.NNs)
    sub_test.go:NNN: Nope`,
		maxPar: 1,
		f: func(t *T) {
			outer := t
			t.Run("", func(t *T) {
				for i := 0; i < 4; i++ {
					t.Run("", func(t *T) {
						outer.Fatalf("Nope")
					})
					t.Errorf("Don't do this")
				}
				t.Errorf("And neither do this")
			})
			t.Errorf("Nor this")
		},
	}, {
		desc:   "panic on goroutine fail after test exit",
		ok:     false,
		maxPar: 4,
		f: func(t *T) {
			ch := make(chan bool)
			t.Run("", func(t *T) {
				go func() {
					<-ch
					defer func() {
						if r := recover(); r == nil {
							realTest.Errorf("expected panic")
						}
						ch <- true
					}()
					t.Errorf("failed after success")
				}()
			})
			ch <- true
			<-ch
		},
	}, {
		desc: "log in finished sub test logs to parent",
		ok:   false,
		output: `
		--- FAIL: log in finished sub test logs to parent (N.NNs)
    sub_test.go:NNN: message2
    sub_test.go:NNN: message1
    sub_test.go:NNN: error`,
		maxPar: 1,
		f: func(t *T) {
			ch := make(chan bool)
			t.Run("sub", func(t2 *T) {
				go func() {
					<-ch
					t2.Log("message1")
					ch <- true
				}()
			})
			t.Log("message2")
			ch <- true
			<-ch
			t.Errorf("error")
		},
	}, {
		// A chatty test should always log with fmt.Print, even if the
		// parent test has completed.
		desc:   "log in finished sub test with chatty",
		ok:     false,
		chatty: true,
		output: `
		--- FAIL: log in finished sub test with chatty (N.NNs)`,
		maxPar: 1,
		f: func(t *T) {
			ch := make(chan bool)
			t.Run("sub", func(t2 *T) {
				go func() {
					<-ch
					t2.Log("message1")
					ch <- true
				}()
			})
			t.Log("message2")
			ch <- true
			<-ch
			t.Errorf("error")
		},
	}, {
		// If a subtest panics we should run cleanups.
		desc:   "cleanup when subtest panics",
		ok:     false,
		chatty: false,
		output: `
--- FAIL: cleanup when subtest panics (N.NNs)
    --- FAIL: cleanup when subtest panics/sub (N.NNs)
    sub_test.go:NNN: running cleanup`,
		f: func(t *T) {
			t.Cleanup(func() { t.Log("running cleanup") })
			t.Run("sub", func(t2 *T) {
				t2.FailNow()
			})
		},
	}}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *T) {
			ctx := newTestContext(tc.maxPar, newMatcher(regexp.MatchString, "", ""))
			buf := &bytes.Buffer{}
			root := &T{
				common: common{
					signal: make(chan bool),
					name:   "Test",
					w:      buf,
				},
				context: ctx,
			}
			if tc.chatty {
				root.chatty = newChattyPrinter(root.w)
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
			got := strings.TrimSpace(buf.String())
			want := strings.TrimSpace(tc.output)
			re := makeRegexp(want)
			if ok, err := regexp.MatchString(re, got); !ok || err != nil {
				t.Errorf("%s:output:\ngot:\n%s\nwant:\n%s", tc.desc, got, want)
			}
		})
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
		desc:   "chatty with recursion",
		chatty: true,
		f: func(b *B) {
			b.Run("", func(b *B) {
				b.Run("", func(b *B) {})
			})
		},
	}, {
		desc: "skipping without message, not chatty",
		f:    func(b *B) { b.SkipNow() },
	}, {
		desc:   "skipping after error",
		failed: true,
		output: `
--- FAIL: root
    sub_test.go:NNN: an error
    sub_test.go:NNN: skipped`,
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
			b.Run("", func(b *B) {
				alloc(b)
				b.ReportAllocs()
			})
			b.Run("", func(b *B) {
				alloc(b)
				b.ReportAllocs()
			})
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
	}, {
		desc: "cleanup is called",
		f: func(b *B) {
			var calls, cleanups, innerCalls, innerCleanups int
			b.Run("", func(b *B) {
				calls++
				b.Cleanup(func() {
					cleanups++
				})
				b.Run("", func(b *B) {
					b.Cleanup(func() {
						innerCleanups++
					})
					innerCalls++
				})
				work(b)
			})
			if calls == 0 || calls != cleanups {
				t.Errorf("mismatched cleanups; got %d want %d", cleanups, calls)
			}
			if innerCalls == 0 || innerCalls != innerCleanups {
				t.Errorf("mismatched cleanups; got %d want %d", cleanups, calls)
			}
		},
	}, {
		desc:   "cleanup is called on failure",
		failed: true,
		f: func(b *B) {
			var calls, cleanups int
			b.Run("", func(b *B) {
				calls++
				b.Cleanup(func() {
					cleanups++
				})
				b.Fatalf("failure")
			})
			if calls == 0 || calls != cleanups {
				t.Errorf("mismatched cleanups; got %d want %d", cleanups, calls)
			}
		},
	}}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *T) {
			var ok bool
			buf := &bytes.Buffer{}
			// This is almost like the Benchmark function, except that we override
			// the benchtime and catch the failure result of the subbenchmark.
			root := &B{
				common: common{
					signal: make(chan bool),
					name:   "root",
					w:      buf,
				},
				benchFunc: func(b *B) { ok = b.Run("test", tc.f) }, // Use Run to catch failure.
				benchTime: benchTimeFlag{d: 1 * time.Microsecond},
			}
			if tc.chatty {
				root.chatty = newChattyPrinter(root.w)
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
			got := strings.TrimSpace(buf.String())
			want := strings.TrimSpace(tc.output)
			re := makeRegexp(want)
			if ok, err := regexp.MatchString(re, got); !ok || err != nil {
				t.Errorf("%s:output:\ngot:\n%s\nwant:\n%s", tc.desc, got, want)
			}
		})
	}
}

func makeRegexp(s string) string {
	s = regexp.QuoteMeta(s)
	s = strings.ReplaceAll(s, ":NNN:", `:\d\d\d\d?:`)
	s = strings.ReplaceAll(s, "N\\.NNs", `\d*\.\d*s`)
	return s
}

func TestBenchmarkOutput(t *T) {
	// Ensure Benchmark initialized common.w by invoking it with an error and
	// normal case.
	Benchmark(func(b *B) { b.Error("do not print this output") })
	Benchmark(func(b *B) {})
}

func TestBenchmarkStartsFrom1(t *T) {
	var first = true
	Benchmark(func(b *B) {
		if first && b.N != 1 {
			panic(fmt.Sprintf("Benchmark() first N=%v; want 1", b.N))
		}
		first = false
	})
}

func TestBenchmarkReadMemStatsBeforeFirstRun(t *T) {
	var first = true
	Benchmark(func(b *B) {
		if first && (b.startAllocs == 0 || b.startBytes == 0) {
			panic(fmt.Sprintf("ReadMemStats not called before first run"))
		}
		first = false
	})
}

func TestParallelSub(t *T) {
	c := make(chan int)
	block := make(chan int)
	for i := 0; i < 10; i++ {
		go func(i int) {
			<-block
			t.Run(fmt.Sprint(i), func(t *T) {})
			c <- 1
		}(i)
	}
	close(block)
	for i := 0; i < 10; i++ {
		<-c
	}
}

type funcWriter struct {
	write func([]byte) (int, error)
}

func (fw *funcWriter) Write(b []byte) (int, error) {
	return fw.write(b)
}

func TestRacyOutput(t *T) {
	var runs int32  // The number of running Writes
	var races int32 // Incremented for each race detected
	raceDetector := func(b []byte) (int, error) {
		// Check if some other goroutine is concurrently calling Write.
		if atomic.LoadInt32(&runs) > 0 {
			atomic.AddInt32(&races, 1) // Race detected!
		}
		atomic.AddInt32(&runs, 1)
		defer atomic.AddInt32(&runs, -1)
		runtime.Gosched() // Increase probability of a race
		return len(b), nil
	}

	var wg sync.WaitGroup
	root := &T{
		common:  common{w: &funcWriter{raceDetector}},
		context: newTestContext(1, newMatcher(regexp.MatchString, "", "")),
	}
	root.chatty = newChattyPrinter(root.w)
	root.Run("", func(t *T) {
		for i := 0; i < 100; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				t.Run(fmt.Sprint(i), func(t *T) {
					t.Logf("testing run %d", i)
				})
			}(i)
		}
	})
	wg.Wait()

	if races > 0 {
		t.Errorf("detected %d racy Writes", races)
	}
}

// The late log message did not include the test name.  Issue 29388.
func TestLogAfterComplete(t *T) {
	ctx := newTestContext(1, newMatcher(regexp.MatchString, "", ""))
	var buf bytes.Buffer
	t1 := &T{
		common: common{
			// Use a buffered channel so that tRunner can write
			// to it although nothing is reading from it.
			signal: make(chan bool, 1),
			w:      &buf,
		},
		context: ctx,
	}

	c1 := make(chan bool)
	c2 := make(chan string)
	tRunner(t1, func(t *T) {
		t.Run("TestLateLog", func(t *T) {
			go func() {
				defer close(c2)
				defer func() {
					p := recover()
					if p == nil {
						c2 <- "subtest did not panic"
						return
					}
					s, ok := p.(string)
					if !ok {
						c2 <- fmt.Sprintf("subtest panic with unexpected value %v", p)
						return
					}
					const want = "Log in goroutine after TestLateLog has completed: log after test"
					if !strings.Contains(s, want) {
						c2 <- fmt.Sprintf("subtest panic %q does not contain %q", s, want)
					}
				}()

				<-c1
				t.Log("log after test")
			}()
		})
	})
	close(c1)

	if s := <-c2; s != "" {
		t.Error(s)
	}
}

func TestBenchmark(t *T) {
	if Short() {
		t.Skip("skipping in short mode")
	}
	res := Benchmark(func(b *B) {
		for i := 0; i < 5; i++ {
			b.Run("", func(b *B) {
				for i := 0; i < b.N; i++ {
					time.Sleep(time.Millisecond)
				}
			})
		}
	})
	if res.NsPerOp() < 4000000 {
		t.Errorf("want >5ms; got %v", time.Duration(res.NsPerOp()))
	}
}

func TestCleanup(t *T) {
	var cleanups []int
	t.Run("test", func(t *T) {
		t.Cleanup(func() { cleanups = append(cleanups, 1) })
		t.Cleanup(func() { cleanups = append(cleanups, 2) })
	})
	if got, want := cleanups, []int{2, 1}; !reflect.DeepEqual(got, want) {
		t.Errorf("unexpected cleanup record; got %v want %v", got, want)
	}
}

func TestConcurrentCleanup(t *T) {
	cleanups := 0
	t.Run("test", func(t *T) {
		done := make(chan struct{})
		for i := 0; i < 2; i++ {
			i := i
			go func() {
				t.Cleanup(func() {
					cleanups |= 1 << i
				})
				done <- struct{}{}
			}()
		}
		<-done
		<-done
	})
	if cleanups != 1|2 {
		t.Errorf("unexpected cleanup; got %d want 3", cleanups)
	}
}

func TestCleanupCalledEvenAfterGoexit(t *T) {
	cleanups := 0
	t.Run("test", func(t *T) {
		t.Cleanup(func() {
			cleanups++
		})
		t.Cleanup(func() {
			runtime.Goexit()
		})
	})
	if cleanups != 1 {
		t.Errorf("unexpected cleanup count; got %d want 1", cleanups)
	}
}

func TestRunCleanup(t *T) {
	outerCleanup := 0
	innerCleanup := 0
	t.Run("test", func(t *T) {
		t.Cleanup(func() { outerCleanup++ })
		t.Run("x", func(t *T) {
			t.Cleanup(func() { innerCleanup++ })
		})
	})
	if innerCleanup != 1 {
		t.Errorf("unexpected inner cleanup count; got %d want 1", innerCleanup)
	}
	if outerCleanup != 1 {
		t.Errorf("unexpected outer cleanup count; got %d want 0", outerCleanup)
	}
}

func TestCleanupParallelSubtests(t *T) {
	ranCleanup := 0
	t.Run("test", func(t *T) {
		t.Cleanup(func() { ranCleanup++ })
		t.Run("x", func(t *T) {
			t.Parallel()
			if ranCleanup > 0 {
				t.Error("outer cleanup ran before parallel subtest")
			}
		})
	})
	if ranCleanup != 1 {
		t.Errorf("unexpected cleanup count; got %d want 1", ranCleanup)
	}
}

func TestNestedCleanup(t *T) {
	ranCleanup := 0
	t.Run("test", func(t *T) {
		t.Cleanup(func() {
			if ranCleanup != 2 {
				t.Errorf("unexpected cleanup count in first cleanup: got %d want 2", ranCleanup)
			}
			ranCleanup++
		})
		t.Cleanup(func() {
			if ranCleanup != 0 {
				t.Errorf("unexpected cleanup count in second cleanup: got %d want 0", ranCleanup)
			}
			ranCleanup++
			t.Cleanup(func() {
				if ranCleanup != 1 {
					t.Errorf("unexpected cleanup count in nested cleanup: got %d want 1", ranCleanup)
				}
				ranCleanup++
			})
		})
	})
	if ranCleanup != 3 {
		t.Errorf("unexpected cleanup count: got %d want 3", ranCleanup)
	}
}
