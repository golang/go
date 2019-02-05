// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !nacl
// +build !js
// +build !plan9
// +build !windows

package os_test

import (
	"fmt"
	"internal/poll"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"testing"
	"time"
)

func TestNonpollableDeadline(t *testing.T) {
	// On BSD systems regular files seem to be pollable,
	// so just run this test on Linux.
	if runtime.GOOS != "linux" {
		t.Skipf("skipping on %s", runtime.GOOS)
	}

	f, err := ioutil.TempFile("", "ostest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()
	deadline := time.Now().Add(10 * time.Second)
	if err := f.SetDeadline(deadline); err != os.ErrNoDeadline {
		t.Errorf("SetDeadline on file returned %v, wanted %v", err, os.ErrNoDeadline)
	}
	if err := f.SetReadDeadline(deadline); err != os.ErrNoDeadline {
		t.Errorf("SetReadDeadline on file returned %v, wanted %v", err, os.ErrNoDeadline)
	}
	if err := f.SetWriteDeadline(deadline); err != os.ErrNoDeadline {
		t.Errorf("SetWriteDeadline on file returned %v, wanted %v", err, os.ErrNoDeadline)
	}
}

// noDeadline is a zero time.Time value, which cancels a deadline.
var noDeadline time.Time

var readTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that read deadlines work, even if there's data ready
	// to be read.
	{-5 * time.Second, [2]error{poll.ErrTimeout, poll.ErrTimeout}},

	{50 * time.Millisecond, [2]error{nil, poll.ErrTimeout}},
}

func TestReadTimeout(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	if _, err := w.Write([]byte("READ TIMEOUT TEST")); err != nil {
		t.Fatal(err)
	}

	for i, tt := range readTimeoutTests {
		if err := r.SetReadDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		var b [1]byte
		for j, xerr := range tt.xerrs {
			for {
				n, err := r.Read(b[:])
				if xerr != nil {
					if !os.IsTimeout(err) {
						t.Fatalf("#%d/%d: %v", i, j, err)
					}
				}
				if err == nil {
					time.Sleep(tt.timeout / 3)
					continue
				}
				if n != 0 {
					t.Fatalf("#%d/%d: read %d; want 0", i, j, n)
				}
				break
			}
		}
	}
}

func TestReadTimeoutMustNotReturn(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	max := time.NewTimer(100 * time.Millisecond)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := r.SetDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := r.SetWriteDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := r.SetReadDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		var b [1]byte
		_, err := r.Read(b[:])
		ch <- err
	}()

	select {
	case err := <-ch:
		t.Fatalf("expected Read to not return, but it returned with %v", err)
	case <-max.C:
		w.Close()
		err := <-ch // wait for tester goroutine to stop
		if os.IsTimeout(err) {
			t.Fatal(err)
		}
	}
}

var writeTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that write deadlines work, even if there's buffer
	// space available to write.
	{-5 * time.Second, [2]error{poll.ErrTimeout, poll.ErrTimeout}},

	{10 * time.Millisecond, [2]error{nil, poll.ErrTimeout}},
}

func TestWriteTimeout(t *testing.T) {
	t.Parallel()

	for i, tt := range writeTimeoutTests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			r, w, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			defer w.Close()

			if err := w.SetWriteDeadline(time.Now().Add(tt.timeout)); err != nil {
				t.Fatalf("%v", err)
			}
			for j, xerr := range tt.xerrs {
				for {
					n, err := w.Write([]byte("WRITE TIMEOUT TEST"))
					if xerr != nil {
						if !os.IsTimeout(err) {
							t.Fatalf("%d: %v", j, err)
						}
					}
					if err == nil {
						time.Sleep(tt.timeout / 3)
						continue
					}
					if n != 0 {
						t.Fatalf("%d: wrote %d; want 0", j, n)
					}
					break
				}
			}
		})
	}
}

func TestWriteTimeoutMustNotReturn(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	max := time.NewTimer(100 * time.Millisecond)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := w.SetDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := w.SetReadDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := w.SetWriteDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		var b [1]byte
		for {
			if _, err := w.Write(b[:]); err != nil {
				ch <- err
				break
			}
		}
	}()

	select {
	case err := <-ch:
		t.Fatalf("expected Write to not return, but it returned with %v", err)
	case <-max.C:
		r.Close()
		err := <-ch // wait for tester goroutine to stop
		if os.IsTimeout(err) {
			t.Fatal(err)
		}
	}
}

func timeoutReader(r *os.File, d, min, max time.Duration, ch chan<- error) {
	var err error
	defer func() { ch <- err }()

	t0 := time.Now()
	if err = r.SetReadDeadline(time.Now().Add(d)); err != nil {
		return
	}
	b := make([]byte, 256)
	var n int
	n, err = r.Read(b)
	t1 := time.Now()
	if n != 0 || err == nil || !os.IsTimeout(err) {
		err = fmt.Errorf("Read did not return (0, timeout): (%d, %v)", n, err)
		return
	}
	if dt := t1.Sub(t0); min > dt || dt > max && !testing.Short() {
		err = fmt.Errorf("Read took %s; expected %s", dt, d)
		return
	}
}

func TestReadTimeoutFluctuation(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	max := time.NewTimer(time.Second)
	defer max.Stop()
	ch := make(chan error)
	go timeoutReader(r, 100*time.Millisecond, 50*time.Millisecond, 250*time.Millisecond, ch)

	select {
	case <-max.C:
		t.Fatal("Read took over 1s; expected 0.1s")
	case err := <-ch:
		if !os.IsTimeout(err) {
			t.Fatal(err)
		}
	}
}

func timeoutWriter(w *os.File, d, min, max time.Duration, ch chan<- error) {
	var err error
	defer func() { ch <- err }()

	t0 := time.Now()
	if err = w.SetWriteDeadline(time.Now().Add(d)); err != nil {
		return
	}
	var n int
	for {
		n, err = w.Write([]byte("TIMEOUT WRITER"))
		if err != nil {
			break
		}
	}
	t1 := time.Now()
	if err == nil || !os.IsTimeout(err) {
		err = fmt.Errorf("Write did not return (any, timeout): (%d, %v)", n, err)
		return
	}
	if dt := t1.Sub(t0); min > dt || dt > max && !testing.Short() {
		err = fmt.Errorf("Write took %s; expected %s", dt, d)
		return
	}
}

func TestWriteTimeoutFluctuation(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	d := time.Second
	max := time.NewTimer(d)
	defer max.Stop()
	ch := make(chan error)
	go timeoutWriter(w, 100*time.Millisecond, 50*time.Millisecond, 250*time.Millisecond, ch)

	select {
	case <-max.C:
		t.Fatalf("Write took over %v; expected 0.1s", d)
	case err := <-ch:
		if !os.IsTimeout(err) {
			t.Fatal(err)
		}
	}
}

func TestVariousDeadlines(t *testing.T) {
	t.Parallel()
	testVariousDeadlines(t)
}

func TestVariousDeadlines1Proc(t *testing.T) {
	// Cannot use t.Parallel - modifies global GOMAXPROCS.
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	testVariousDeadlines(t)
}

func TestVariousDeadlines4Proc(t *testing.T) {
	// Cannot use t.Parallel - modifies global GOMAXPROCS.
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	testVariousDeadlines(t)
}

type neverEnding byte

func (b neverEnding) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

func testVariousDeadlines(t *testing.T) {
	type result struct {
		n   int64
		err error
		d   time.Duration
	}

	handler := func(w *os.File, pasvch chan result) {
		// The writer, with no timeouts of its own,
		// sending bytes to clients as fast as it can.
		t0 := time.Now()
		n, err := io.Copy(w, neverEnding('a'))
		dt := time.Since(t0)
		pasvch <- result{n, err, dt}
	}

	for _, timeout := range []time.Duration{
		1 * time.Nanosecond,
		2 * time.Nanosecond,
		5 * time.Nanosecond,
		50 * time.Nanosecond,
		100 * time.Nanosecond,
		200 * time.Nanosecond,
		500 * time.Nanosecond,
		750 * time.Nanosecond,
		1 * time.Microsecond,
		5 * time.Microsecond,
		25 * time.Microsecond,
		250 * time.Microsecond,
		500 * time.Microsecond,
		1 * time.Millisecond,
		5 * time.Millisecond,
		100 * time.Millisecond,
		250 * time.Millisecond,
		500 * time.Millisecond,
		1 * time.Second,
	} {
		numRuns := 3
		if testing.Short() {
			numRuns = 1
			if timeout > 500*time.Microsecond {
				continue
			}
		}
		for run := 0; run < numRuns; run++ {
			t.Run(fmt.Sprintf("%v-%d", timeout, run+1), func(t *testing.T) {
				r, w, err := os.Pipe()
				if err != nil {
					t.Fatal(err)
				}
				defer r.Close()
				defer w.Close()

				pasvch := make(chan result)
				go handler(w, pasvch)

				tooLong := 5 * time.Second
				max := time.NewTimer(tooLong)
				defer max.Stop()
				actvch := make(chan result)
				go func() {
					t0 := time.Now()
					if err := r.SetDeadline(t0.Add(timeout)); err != nil {
						t.Error(err)
					}
					n, err := io.Copy(ioutil.Discard, r)
					dt := time.Since(t0)
					r.Close()
					actvch <- result{n, err, dt}
				}()

				select {
				case res := <-actvch:
					if os.IsTimeout(res.err) {
						t.Logf("good client timeout after %v, reading %d bytes", res.d, res.n)
					} else {
						t.Fatalf("client Copy = %d, %v; want timeout", res.n, res.err)
					}
				case <-max.C:
					t.Fatalf("timeout (%v) waiting for client to timeout (%v) reading", tooLong, timeout)
				}

				select {
				case res := <-pasvch:
					t.Logf("writer in %v wrote %d: %v", res.d, res.n, res.err)
				case <-max.C:
					t.Fatalf("timeout waiting for writer to finish writing")
				}
			})
		}
	}
}

func TestReadWriteDeadlineRace(t *testing.T) {
	t.Parallel()

	N := 1000
	if testing.Short() {
		N = 50
	}

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		tic := time.NewTicker(2 * time.Microsecond)
		defer tic.Stop()
		for i := 0; i < N; i++ {
			if err := r.SetReadDeadline(time.Now().Add(2 * time.Microsecond)); err != nil {
				break
			}
			if err := w.SetWriteDeadline(time.Now().Add(2 * time.Microsecond)); err != nil {
				break
			}
			<-tic.C
		}
	}()
	go func() {
		defer wg.Done()
		var b [1]byte
		for i := 0; i < N; i++ {
			_, err := r.Read(b[:])
			if err != nil && !os.IsTimeout(err) {
				t.Error("Read returned non-timeout error", err)
			}
		}
	}()
	go func() {
		defer wg.Done()
		var b [1]byte
		for i := 0; i < N; i++ {
			_, err := w.Write(b[:])
			if err != nil && !os.IsTimeout(err) {
				t.Error("Write returned non-timeout error", err)
			}
		}
	}()
	wg.Wait() // wait for tester goroutine to stop
}

// TestRacyRead tests that it is safe to mutate the input Read buffer
// immediately after cancelation has occurred.
func TestRacyRead(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	var wg sync.WaitGroup
	defer wg.Wait()

	go io.Copy(w, rand.New(rand.NewSource(0)))

	r.SetReadDeadline(time.Now().Add(time.Millisecond))
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			b1 := make([]byte, 1024)
			b2 := make([]byte, 1024)
			for j := 0; j < 100; j++ {
				_, err := r.Read(b1)
				copy(b1, b2) // Mutate b1 to trigger potential race
				if err != nil {
					if !os.IsTimeout(err) {
						t.Error(err)
					}
					r.SetReadDeadline(time.Now().Add(time.Millisecond))
				}
			}
		}()
	}
}

// TestRacyWrite tests that it is safe to mutate the input Write buffer
// immediately after cancelation has occurred.
func TestRacyWrite(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	var wg sync.WaitGroup
	defer wg.Wait()

	go io.Copy(ioutil.Discard, r)

	w.SetWriteDeadline(time.Now().Add(time.Millisecond))
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			b1 := make([]byte, 1024)
			b2 := make([]byte, 1024)
			for j := 0; j < 100; j++ {
				_, err := w.Write(b1)
				copy(b1, b2) // Mutate b1 to trigger potential race
				if err != nil {
					if !os.IsTimeout(err) {
						t.Error(err)
					}
					w.SetWriteDeadline(time.Now().Add(time.Millisecond))
				}
			}
		}()
	}
}

// Closing a TTY while reading from it should not hang.  Issue 23943.
func TestTTYClose(t *testing.T) {
	// Ignore SIGTTIN in case we are running in the background.
	signal.Ignore(syscall.SIGTTIN)
	defer signal.Reset(syscall.SIGTTIN)

	f, err := os.Open("/dev/tty")
	if err != nil {
		t.Skipf("skipping because opening /dev/tty failed: %v", err)
	}

	go func() {
		var buf [1]byte
		f.Read(buf[:])
	}()

	// Give the goroutine a chance to enter the read.
	// It doesn't matter much if it occasionally fails to do so,
	// we won't be testing what we want to test but the test will pass.
	time.Sleep(time.Millisecond)

	c := make(chan bool)
	go func() {
		defer close(c)
		f.Close()
	}()

	select {
	case <-c:
	case <-time.After(time.Second):
		t.Error("timed out waiting for close")
	}

	// On some systems the goroutines may now be hanging.
	// There's not much we can do about that.
}
