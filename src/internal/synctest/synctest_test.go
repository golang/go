// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package synctest_test

import (
	"fmt"
	"internal/synctest"
	"iter"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"
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

func TestTimerFromOutsideBubble(t *testing.T) {
	tm := time.NewTimer(10 * time.Millisecond)
	synctest.Run(func() {
		<-tm.C
	})
	if tm.Stop() {
		t.Errorf("synctest.Run unexpectedly returned before timer fired")
	}
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
	defer wantPanic(t, "deadlock: all goroutines in bubble are blocked")
	synctest.Run(func() {
		go func() {
			select {}
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

func TestWaitGroup(t *testing.T) {
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

func wantPanic(t *testing.T, want string) {
	if e := recover(); e != nil {
		if got := fmt.Sprint(e); got != want {
			t.Errorf("got panic message %q, want %q", got, want)
		}
	} else {
		t.Errorf("got no panic, want one")
	}
}
