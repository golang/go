// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package synctest_test

import (
	"fmt"
	"internal/synctest"
	"internal/testenv"
	"iter"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"weak"
)

func TestNow(t *testing.T) {
	start := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC).In(time.Local)
	synctest.Run(func() {
		// Time starts at 2000-1-1 00:00:00.
		if got, want := time.Now(), start; !got.Equal(want) {
			t.Errorf("at start: time.Now = %v, want %v", got, want)
		}
		go func() {
			// New goroutines see the same fake clock.
			if got, want := time.Now(), start; !got.Equal(want) {
				t.Errorf("time.Now = %v, want %v", got, want)
			}
		}()
		// Time advances after a sleep.
		time.Sleep(1 * time.Second)
		if got, want := time.Now(), start.Add(1*time.Second); !got.Equal(want) {
			t.Errorf("after sleep: time.Now = %v, want %v", got, want)
		}
	})
}

// TestMonotonicClock exercises comparing times from within a bubble
// with ones from outside the bubble.
func TestMonotonicClock(t *testing.T) {
	start := time.Now()
	synctest.Run(func() {
		time.Sleep(time.Until(start.Round(0)))
		if got, want := time.Now().In(time.UTC), start.In(time.UTC); !got.Equal(want) {
			t.Fatalf("time.Now() = %v, want %v", got, want)
		}

		wait := 1 * time.Second
		time.Sleep(wait)
		if got := time.Since(start); got != wait {
			t.Fatalf("time.Since(start) = %v, want %v", got, wait)
		}
		if got := time.Now().Sub(start); got != wait {
			t.Fatalf("time.Now().Sub(start) = %v, want %v", got, wait)
		}
	})
}

func TestRunEmpty(t *testing.T) {
	synctest.Run(func() {
	})
}

func TestSimpleWait(t *testing.T) {
	synctest.Run(func() {
		synctest.Wait()
	})
}

func TestGoroutineWait(t *testing.T) {
	synctest.Run(func() {
		go func() {}()
		synctest.Wait()
	})
}

// TestWait starts a collection of goroutines.
// It checks that synctest.Wait waits for all goroutines to exit before returning.
func TestWait(t *testing.T) {
	synctest.Run(func() {
		done := false
		ch := make(chan int)
		var f func()
		f = func() {
			count := <-ch
			if count == 0 {
				done = true
			} else {
				go f()
				ch <- count - 1
			}
		}
		go f()
		ch <- 100
		synctest.Wait()
		if !done {
			t.Fatalf("done = false, want true")
		}
	})
}

func TestMallocs(t *testing.T) {
	for i := 0; i < 100; i++ {
		synctest.Run(func() {
			done := false
			ch := make(chan []byte)
			var f func()
			f = func() {
				b := <-ch
				if len(b) == 0 {
					done = true
				} else {
					go f()
					ch <- make([]byte, len(b)-1)
				}
			}
			go f()
			ch <- make([]byte, 100)
			synctest.Wait()
			if !done {
				t.Fatalf("done = false, want true")
			}
		})
	}
}

func TestTimerReadBeforeDeadline(t *testing.T) {
	synctest.Run(func() {
		start := time.Now()
		tm := time.NewTimer(5 * time.Second)
		<-tm.C
		if got, want := time.Since(start), 5*time.Second; got != want {
			t.Errorf("after sleep: time.Since(start) = %v, want %v", got, want)
		}
	})
}

func TestTimerReadAfterDeadline(t *testing.T) {
	synctest.Run(func() {
		delay := 1 * time.Second
		want := time.Now().Add(delay)
		tm := time.NewTimer(delay)
		time.Sleep(2 * delay)
		got := <-tm.C
		if got != want {
			t.Errorf("<-tm.C = %v, want %v", got, want)
		}
	})
}

func TestTimerReset(t *testing.T) {
	synctest.Run(func() {
		start := time.Now()
		tm := time.NewTimer(1 * time.Second)
		if got, want := <-tm.C, start.Add(1*time.Second); got != want {
			t.Errorf("first sleep: <-tm.C = %v, want %v", got, want)
		}

		tm.Reset(2 * time.Second)
		if got, want := <-tm.C, start.Add((1+2)*time.Second); got != want {
			t.Errorf("second sleep: <-tm.C = %v, want %v", got, want)
		}

		tm.Reset(3 * time.Second)
		time.Sleep(1 * time.Second)
		tm.Reset(3 * time.Second)
		if got, want := <-tm.C, start.Add((1+2+4)*time.Second); got != want {
			t.Errorf("third sleep: <-tm.C = %v, want %v", got, want)
		}
	})
}

func TestTimeAfter(t *testing.T) {
	synctest.Run(func() {
		i := 0
		time.AfterFunc(1*time.Second, func() {
			// Ensure synctest group membership propagates through the AfterFunc.
			i++ // 1
			go func() {
				time.Sleep(1 * time.Second)
				i++ // 2
			}()
		})
		time.Sleep(3 * time.Second)
		synctest.Wait()
		if got, want := i, 2; got != want {
			t.Errorf("after sleep and wait: i = %v, want %v", got, want)
		}
	})
}

func TestTimerAfterBubbleExit(t *testing.T) {
	run := false
	synctest.Run(func() {
		time.AfterFunc(1*time.Second, func() {
			run = true
		})
	})
	if run {
		t.Errorf("timer ran before bubble exit")
	}
}

func TestTimerFromOutsideBubble(t *testing.T) {
	tm := time.NewTimer(10 * time.Millisecond)
	synctest.Run(func() {
		<-tm.C
	})
	if tm.Stop() {
		t.Errorf("synctest.Run unexpectedly returned before timer fired")
	}
}

// TestTimerNondeterminism verifies that timers firing at the same instant
// don't always fire in exactly the same order.
func TestTimerNondeterminism(t *testing.T) {
	synctest.Run(func() {
		const iterations = 1000
		var seen1, seen2 bool
		for range iterations {
			tm1 := time.NewTimer(1)
			tm2 := time.NewTimer(1)
			select {
			case <-tm1.C:
				seen1 = true
			case <-tm2.C:
				seen2 = true
			}
			if seen1 && seen2 {
				return
			}
			synctest.Wait()
		}
		t.Errorf("after %v iterations, seen timer1:%v, timer2:%v; want both", iterations, seen1, seen2)
	})
}

// TestSleepNondeterminism verifies that goroutines sleeping to the same instant
// don't always schedule in exactly the same order.
func TestSleepNondeterminism(t *testing.T) {
	synctest.Run(func() {
		const iterations = 1000
		var seen1, seen2 bool
		for range iterations {
			var first atomic.Int32
			go func() {
				time.Sleep(1)
				first.CompareAndSwap(0, 1)
			}()
			go func() {
				time.Sleep(1)
				first.CompareAndSwap(0, 2)
			}()
			time.Sleep(1)
			synctest.Wait()
			switch v := first.Load(); v {
			case 1:
				seen1 = true
			case 2:
				seen2 = true
			default:
				t.Fatalf("first = %v, want 1 or 2", v)
			}
			if seen1 && seen2 {
				return
			}
			synctest.Wait()
		}
		t.Errorf("after %v iterations, seen goroutine 1:%v, 2:%v; want both", iterations, seen1, seen2)
	})
}

// TestTimerRunsImmediately verifies that a 0-duration timer sends on its channel
// without waiting for the bubble to block.
func TestTimerRunsImmediately(t *testing.T) {
	synctest.Run(func() {
		start := time.Now()
		tm := time.NewTimer(0)
		select {
		case got := <-tm.C:
			if !got.Equal(start) {
				t.Errorf("<-tm.C = %v, want %v", got, start)
			}
		default:
			t.Errorf("0-duration timer channel is not readable; want it to be")
		}
	})
}

// TestTimerRunsLater verifies that reading from a timer's channel receives the
// timer fired, even when that time is in reading from a timer's channel receives the
// time the timer fired, even when that time is in the past.
func TestTimerRanInPast(t *testing.T) {
	synctest.Run(func() {
		delay := 1 * time.Second
		want := time.Now().Add(delay)
		tm := time.NewTimer(delay)
		time.Sleep(2 * delay)
		select {
		case got := <-tm.C:
			if !got.Equal(want) {
				t.Errorf("<-tm.C = %v, want %v", got, want)
			}
		default:
			t.Errorf("0-duration timer channel is not readable; want it to be")
		}
	})
}

// TestAfterFuncRunsImmediately verifies that a 0-duration AfterFunc is scheduled
// without waiting for the bubble to block.
func TestAfterFuncRunsImmediately(t *testing.T) {
	synctest.Run(func() {
		var b atomic.Bool
		time.AfterFunc(0, func() {
			b.Store(true)
		})
		for !b.Load() {
			runtime.Gosched()
		}
	})
}

func TestChannelFromOutsideBubble(t *testing.T) {
	choutside := make(chan struct{})
	for _, test := range []struct {
		desc    string
		outside func(ch chan int)
		inside  func(ch chan int)
	}{{
		desc:    "read closed",
		outside: func(ch chan int) { close(ch) },
		inside:  func(ch chan int) { <-ch },
	}, {
		desc:    "read value",
		outside: func(ch chan int) { ch <- 0 },
		inside:  func(ch chan int) { <-ch },
	}, {
		desc:    "write value",
		outside: func(ch chan int) { <-ch },
		inside:  func(ch chan int) { ch <- 0 },
	}, {
		desc:    "select outside only",
		outside: func(ch chan int) { close(ch) },
		inside: func(ch chan int) {
			select {
			case <-ch:
			case <-choutside:
			}
		},
	}, {
		desc:    "select mixed",
		outside: func(ch chan int) { close(ch) },
		inside: func(ch chan int) {
			ch2 := make(chan struct{})
			select {
			case <-ch:
			case <-ch2:
			}
		},
	}} {
		t.Run(test.desc, func(t *testing.T) {
			ch := make(chan int)
			time.AfterFunc(1*time.Millisecond, func() {
				test.outside(ch)
			})
			synctest.Run(func() {
				test.inside(ch)
			})
		})
	}
}

func TestChannelMovedOutOfBubble(t *testing.T) {
	for _, test := range []struct {
		desc      string
		f         func(chan struct{})
		wantPanic string
	}{{
		desc: "receive",
		f: func(ch chan struct{}) {
			<-ch
		},
		wantPanic: "receive on synctest channel from outside bubble",
	}, {
		desc: "send",
		f: func(ch chan struct{}) {
			ch <- struct{}{}
		},
		wantPanic: "send on synctest channel from outside bubble",
	}, {
		desc: "close",
		f: func(ch chan struct{}) {
			close(ch)
		},
		wantPanic: "close of synctest channel from outside bubble",
	}} {
		t.Run(test.desc, func(t *testing.T) {
			// Bubbled channel accessed from outside any bubble.
			t.Run("outside_bubble", func(t *testing.T) {
				donec := make(chan struct{})
				ch := make(chan chan struct{})
				go func() {
					defer close(donec)
					defer wantPanic(t, test.wantPanic)
					test.f(<-ch)
				}()
				synctest.Run(func() {
					ch <- make(chan struct{})
				})
				<-donec
			})
			// Bubbled channel accessed from a different bubble.
			t.Run("different_bubble", func(t *testing.T) {
				donec := make(chan struct{})
				ch := make(chan chan struct{})
				go func() {
					defer close(donec)
					c := <-ch
					synctest.Run(func() {
						defer wantPanic(t, test.wantPanic)
						test.f(c)
					})
				}()
				synctest.Run(func() {
					ch <- make(chan struct{})
				})
				<-donec
			})
		})
	}
}

func TestTimerFromInsideBubble(t *testing.T) {
	for _, test := range []struct {
		desc      string
		f         func(tm *time.Timer)
		wantPanic string
	}{{
		desc: "read channel",
		f: func(tm *time.Timer) {
			<-tm.C
		},
		wantPanic: "receive on synctest channel from outside bubble",
	}, {
		desc: "Reset",
		f: func(tm *time.Timer) {
			tm.Reset(1 * time.Second)
		},
		wantPanic: "reset of synctest timer from outside bubble",
	}, {
		desc: "Stop",
		f: func(tm *time.Timer) {
			tm.Stop()
		},
		wantPanic: "stop of synctest timer from outside bubble",
	}} {
		t.Run(test.desc, func(t *testing.T) {
			donec := make(chan struct{})
			ch := make(chan *time.Timer)
			go func() {
				defer close(donec)
				defer wantPanic(t, test.wantPanic)
				test.f(<-ch)
			}()
			synctest.Run(func() {
				tm := time.NewTimer(1 * time.Second)
				ch <- tm
			})
			<-donec
		})
	}
}

func TestDeadlockRoot(t *testing.T) {
	defer wantPanic(t, "deadlock: all goroutines in bubble are blocked")
	synctest.Run(func() {
		select {}
	})
}

func TestDeadlockChild(t *testing.T) {
	defer wantPanic(t, "deadlock: main bubble goroutine has exited but blocked goroutines remain")
	synctest.Run(func() {
		go func() {
			select {}
		}()
	})
}

func TestDeadlockTicker(t *testing.T) {
	defer wantPanic(t, "deadlock: main bubble goroutine has exited but blocked goroutines remain")
	synctest.Run(func() {
		go func() {
			for range time.Tick(1 * time.Second) {
				t.Errorf("ticker unexpectedly ran")
				return
			}
		}()
	})
}

func TestCond(t *testing.T) {
	synctest.Run(func() {
		var mu sync.Mutex
		cond := sync.NewCond(&mu)
		start := time.Now()
		const waitTime = 1 * time.Millisecond

		go func() {
			// Signal the cond.
			time.Sleep(waitTime)
			mu.Lock()
			cond.Signal()
			mu.Unlock()

			// Broadcast to the cond.
			time.Sleep(waitTime)
			mu.Lock()
			cond.Broadcast()
			mu.Unlock()
		}()

		// Wait for cond.Signal.
		mu.Lock()
		cond.Wait()
		mu.Unlock()
		if got, want := time.Since(start), waitTime; got != want {
			t.Errorf("after cond.Signal: time elapsed = %v, want %v", got, want)
		}

		// Wait for cond.Broadcast in two goroutines.
		waiterDone := false
		go func() {
			mu.Lock()
			cond.Wait()
			mu.Unlock()
			waiterDone = true
		}()
		mu.Lock()
		cond.Wait()
		mu.Unlock()
		synctest.Wait()
		if !waiterDone {
			t.Errorf("after cond.Broadcast: waiter not done")
		}
		if got, want := time.Since(start), 2*waitTime; got != want {
			t.Errorf("after cond.Broadcast: time elapsed = %v, want %v", got, want)
		}
	})
}

func TestIteratorPush(t *testing.T) {
	synctest.Run(func() {
		seq := func(yield func(time.Time) bool) {
			for yield(time.Now()) {
				time.Sleep(1 * time.Second)
			}
		}
		var got []time.Time
		go func() {
			for now := range seq {
				got = append(got, now)
				if len(got) >= 3 {
					break
				}
			}
		}()
		want := []time.Time{
			time.Now(),
			time.Now().Add(1 * time.Second),
			time.Now().Add(2 * time.Second),
		}
		time.Sleep(5 * time.Second)
		synctest.Wait()
		if !slices.Equal(got, want) {
			t.Errorf("got: %v; want: %v", got, want)
		}
	})
}

func TestIteratorPull(t *testing.T) {
	synctest.Run(func() {
		seq := func(yield func(time.Time) bool) {
			for yield(time.Now()) {
				time.Sleep(1 * time.Second)
			}
		}
		var got []time.Time
		go func() {
			next, stop := iter.Pull(seq)
			defer stop()
			for len(got) < 3 {
				now, _ := next()
				got = append(got, now)
			}
		}()
		want := []time.Time{
			time.Now(),
			time.Now().Add(1 * time.Second),
			time.Now().Add(2 * time.Second),
		}
		time.Sleep(5 * time.Second)
		synctest.Wait()
		if !slices.Equal(got, want) {
			t.Errorf("got: %v; want: %v", got, want)
		}
	})
}

func TestReflectFuncOf(t *testing.T) {
	mkfunc := func(name string, i int) {
		reflect.FuncOf([]reflect.Type{
			reflect.StructOf([]reflect.StructField{{
				Name: name + strconv.Itoa(i),
				Type: reflect.TypeOf(0),
			}}),
		}, nil, false)
	}
	go func() {
		for i := 0; i < 100000; i++ {
			mkfunc("A", i)
		}
	}()
	synctest.Run(func() {
		for i := 0; i < 100000; i++ {
			mkfunc("A", i)
		}
	})
}

func TestWaitGroupInBubble(t *testing.T) {
	synctest.Run(func() {
		var wg sync.WaitGroup
		wg.Add(1)
		const delay = 1 * time.Second
		go func() {
			time.Sleep(delay)
			wg.Done()
		}()
		start := time.Now()
		wg.Wait()
		if got := time.Since(start); got != delay {
			t.Fatalf("WaitGroup.Wait() took %v, want %v", got, delay)
		}
	})
}

func TestWaitGroupOutOfBubble(t *testing.T) {
	var wg sync.WaitGroup
	wg.Add(1)
	donec := make(chan struct{})
	go synctest.Run(func() {
		// Since wg.Add was called outside the bubble, Wait is not durably blocking
		// and this waits until wg.Done is called below.
		wg.Wait()
		close(donec)
	})
	select {
	case <-donec:
		t.Fatalf("synctest.Run finished before WaitGroup.Done called")
	case <-time.After(1 * time.Millisecond):
	}
	wg.Done()
	<-donec
}

func TestWaitGroupMovedIntoBubble(t *testing.T) {
	wantFatal(t, "fatal error: sync: WaitGroup.Add called from inside and outside synctest bubble", func() {
		var wg sync.WaitGroup
		wg.Add(1)
		synctest.Run(func() {
			wg.Add(1)
		})
	})
}

func TestWaitGroupMovedOutOfBubble(t *testing.T) {
	wantFatal(t, "fatal error: sync: WaitGroup.Add called from inside and outside synctest bubble", func() {
		var wg sync.WaitGroup
		synctest.Run(func() {
			wg.Add(1)
		})
		wg.Add(1)
	})
}

func TestWaitGroupMovedBetweenBubblesWithNonZeroCount(t *testing.T) {
	wantFatal(t, "fatal error: sync: WaitGroup.Add called from multiple synctest bubbles", func() {
		var wg sync.WaitGroup
		synctest.Run(func() {
			wg.Add(1)
		})
		synctest.Run(func() {
			wg.Add(1)
		})
	})
}

func TestWaitGroupMovedBetweenBubblesWithZeroCount(t *testing.T) {
	var wg sync.WaitGroup
	synctest.Run(func() {
		wg.Add(1)
		wg.Done()
	})
	synctest.Run(func() {
		// Reusing the WaitGroup is safe, because its count is zero.
		wg.Add(1)
		wg.Done()
	})
}

func TestWaitGroupMovedBetweenBubblesAfterWait(t *testing.T) {
	var wg sync.WaitGroup
	synctest.Run(func() {
		wg.Go(func() {})
		wg.Wait()
	})
	synctest.Run(func() {
		// Reusing the WaitGroup is safe, because its count is zero.
		wg.Go(func() {})
		wg.Wait()
	})
}

var testWaitGroupLinkerAllocatedWG sync.WaitGroup

func TestWaitGroupLinkerAllocated(t *testing.T) {
	synctest.Run(func() {
		// This WaitGroup is probably linker-allocated and has no span,
		// so we won't be able to add a special to it associating it with
		// this bubble.
		//
		// Operations on it may not be durably blocking,
		// but they shouldn't fail.
		testWaitGroupLinkerAllocatedWG.Go(func() {})
		testWaitGroupLinkerAllocatedWG.Wait()
	})
}

var testWaitGroupHeapAllocatedWG = new(sync.WaitGroup)

func TestWaitGroupHeapAllocated(t *testing.T) {
	synctest.Run(func() {
		// This package-scoped WaitGroup var should have been heap-allocated,
		// so we can associate it with a bubble.
		testWaitGroupHeapAllocatedWG.Add(1)
		go testWaitGroupHeapAllocatedWG.Wait()
		synctest.Wait()
		testWaitGroupHeapAllocatedWG.Done()
	})
}

func TestHappensBefore(t *testing.T) {
	// Use two parallel goroutines accessing different vars to ensure that
	// we correctly account for multiple goroutines in the bubble.
	var v1 int
	var v2 int
	synctest.Run(func() {
		v1++ // 1
		v2++ // 1

		// Wait returns after these goroutines exit.
		go func() {
			v1++ // 2
		}()
		go func() {
			v2++ // 2
		}()
		synctest.Wait()

		v1++ // 3
		v2++ // 3

		// Wait returns after these goroutines block.
		ch1 := make(chan struct{})
		go func() {
			v1++ // 4
			<-ch1
		}()
		go func() {
			v2++ // 4
			<-ch1
		}()
		synctest.Wait()

		v1++ // 5
		v2++ // 5
		close(ch1)

		// Wait returns after these timers run.
		time.AfterFunc(0, func() {
			v1++ // 6
		})
		time.AfterFunc(0, func() {
			v2++ // 6
		})
		synctest.Wait()

		v1++ // 7
		v2++ // 7

		// Wait returns after these timer goroutines block.
		ch2 := make(chan struct{})
		time.AfterFunc(0, func() {
			v1++ // 8
			<-ch2
		})
		time.AfterFunc(0, func() {
			v2++ // 8
			<-ch2
		})
		synctest.Wait()

		v1++ // 9
		v2++ // 9
		close(ch2)
	})
	// This Run happens after the previous Run returns.
	synctest.Run(func() {
		go func() {
			go func() {
				v1++ // 10
			}()
		}()
		go func() {
			go func() {
				v2++ // 10
			}()
		}()
	})
	// These tests happen after Run returns.
	if got, want := v1, 10; got != want {
		t.Errorf("v1 = %v, want %v", got, want)
	}
	if got, want := v2, 10; got != want {
		t.Errorf("v2 = %v, want %v", got, want)
	}
}

// https://go.dev/issue/73817
func TestWeak(t *testing.T) {
	synctest.Run(func() {
		for range 5 {
			runtime.GC()
			b := make([]byte, 1024)
			weak.Make(&b)
		}
	})
}

func wantPanic(t *testing.T, want string) {
	if e := recover(); e != nil {
		if got := fmt.Sprint(e); got != want {
			t.Errorf("got panic message %q, want %q", got, want)
		}
	} else {
		t.Errorf("got no panic, want one")
	}
}

func wantFatal(t *testing.T, want string, f func()) {
	t.Helper()

	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		f()
		return
	}

	cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^"+t.Name()+"$")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Errorf("expected test function to panic, but test returned successfully")
	}
	if !strings.Contains(string(out), want) {
		t.Errorf("wanted test output contaiing %q; got %q", want, string(out))
	}
}
