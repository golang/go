// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !plan9 && !wasip1 && !windows

package os_test

import (
	"fmt"
	"io"
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
	t.Parallel()

	f, err := os.CreateTemp("", "ostest")
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
	{-5 * time.Second, [2]error{os.ErrDeadlineExceeded, os.ErrDeadlineExceeded}},

	{50 * time.Millisecond, [2]error{nil, os.ErrDeadlineExceeded}},
}

// There is a very similar copy of this in net/timeout_test.go.
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
					if !isDeadlineExceeded(err) {
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

// There is a very similar copy of this in net/timeout_test.go.
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
	{-5 * time.Second, [2]error{os.ErrDeadlineExceeded, os.ErrDeadlineExceeded}},

	{10 * time.Millisecond, [2]error{nil, os.ErrDeadlineExceeded}},
}

// There is a very similar copy of this in net/timeout_test.go.
func TestWriteTimeout(t *testing.T) {
	t.Parallel()

	for i, tt := range writeTimeoutTests {
		t.Run(fmt.Sprintf("#%d", i), func { t ->
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
						if !isDeadlineExceeded(err) {
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

// There is a very similar copy of this in net/timeout_test.go.
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

const (
	// minDynamicTimeout is the minimum timeout to attempt for
	// tests that automatically increase timeouts until success.
	//
	// Lower values may allow tests to succeed more quickly if the value is close
	// to the true minimum, but may require more iterations (and waste more time
	// and CPU power on failed attempts) if the timeout is too low.
	minDynamicTimeout = 1 * time.Millisecond

	// maxDynamicTimeout is the maximum timeout to attempt for
	// tests that automatically increase timeouts until success.
	//
	// This should be a strict upper bound on the latency required to hit a
	// timeout accurately, even on a slow or heavily-loaded machine. If a test
	// would increase the timeout beyond this value, the test fails.
	maxDynamicTimeout = 4 * time.Second
)

// timeoutUpperBound returns the maximum time that we expect a timeout of
// duration d to take to return the caller.
func timeoutUpperBound(d time.Duration) time.Duration {
	switch runtime.GOOS {
	case "openbsd", "netbsd":
		// NetBSD and OpenBSD seem to be unable to reliably hit deadlines even when
		// the absolute durations are long.
		// In https://build.golang.org/log/c34f8685d020b98377dd4988cd38f0c5bd72267e,
		// we observed that an openbsd-amd64-68 builder took 4.090948779s for a
		// 2.983020682s timeout (37.1% overhead).
		// (See https://go.dev/issue/50189 for further detail.)
		// Give them lots of slop to compensate.
		return d * 3 / 2
	}
	// Other platforms seem to hit their deadlines more reliably,
	// at least when they are long enough to cover scheduling jitter.
	return d * 11 / 10
}

// nextTimeout returns the next timeout to try after an operation took the given
// actual duration with a timeout shorter than that duration.
func nextTimeout(actual time.Duration) (next time.Duration, ok bool) {
	if actual >= maxDynamicTimeout {
		return maxDynamicTimeout, false
	}
	// Since the previous attempt took actual, we can't expect to beat that
	// duration by any significant margin. Try the next attempt with an arbitrary
	// factor above that, so that our growth curve is at least exponential.
	next = actual * 5 / 4
	if next > maxDynamicTimeout {
		return maxDynamicTimeout, true
	}
	return next, true
}

// There is a very similar copy of this in net/timeout_test.go.
func TestReadTimeoutFluctuation(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	d := minDynamicTimeout
	b := make([]byte, 256)
	for {
		t.Logf("SetReadDeadline(+%v)", d)
		t0 := time.Now()
		deadline := t0.Add(d)
		if err = r.SetReadDeadline(deadline); err != nil {
			t.Fatalf("SetReadDeadline(%v): %v", deadline, err)
		}
		var n int
		n, err = r.Read(b)
		t1 := time.Now()

		if n != 0 || err == nil || !isDeadlineExceeded(err) {
			t.Errorf("Read did not return (0, timeout): (%d, %v)", n, err)
		}

		actual := t1.Sub(t0)
		if t1.Before(deadline) {
			t.Errorf("Read took %s; expected at least %s", actual, d)
		}
		if t.Failed() {
			return
		}
		if want := timeoutUpperBound(d); actual > want {
			next, ok := nextTimeout(actual)
			if !ok {
				t.Fatalf("Read took %s; expected at most %v", actual, want)
			}
			// Maybe this machine is too slow to reliably schedule goroutines within
			// the requested duration. Increase the timeout and try again.
			t.Logf("Read took %s (expected %s); trying with longer timeout", actual, d)
			d = next
			continue
		}

		break
	}
}

// There is a very similar copy of this in net/timeout_test.go.
func TestWriteTimeoutFluctuation(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	d := minDynamicTimeout
	for {
		t.Logf("SetWriteDeadline(+%v)", d)
		t0 := time.Now()
		deadline := t0.Add(d)
		if err := w.SetWriteDeadline(deadline); err != nil {
			t.Fatalf("SetWriteDeadline(%v): %v", deadline, err)
		}
		var n int64
		var err error
		for {
			var dn int
			dn, err = w.Write([]byte("TIMEOUT TRANSMITTER"))
			n += int64(dn)
			if err != nil {
				break
			}
		}
		t1 := time.Now()
		// Inv: err != nil
		if !isDeadlineExceeded(err) {
			t.Fatalf("Write did not return (any, timeout): (%d, %v)", n, err)
		}

		actual := t1.Sub(t0)
		if t1.Before(deadline) {
			t.Errorf("Write took %s; expected at least %s", actual, d)
		}
		if t.Failed() {
			return
		}
		if want := timeoutUpperBound(d); actual > want {
			if n > 0 {
				// SetWriteDeadline specifies a time “after which I/O operations fail
				// instead of blocking”. However, the kernel's send buffer is not yet
				// full, we may be able to write some arbitrary (but finite) number of
				// bytes to it without blocking.
				t.Logf("Wrote %d bytes into send buffer; retrying until buffer is full", n)
				if d <= maxDynamicTimeout/2 {
					// We don't know how long the actual write loop would have taken if
					// the buffer were full, so just guess and double the duration so that
					// the next attempt can make twice as much progress toward filling it.
					d *= 2
				}
			} else if next, ok := nextTimeout(actual); !ok {
				t.Fatalf("Write took %s; expected at most %s", actual, want)
			} else {
				// Maybe this machine is too slow to reliably schedule goroutines within
				// the requested duration. Increase the timeout and try again.
				t.Logf("Write took %s (expected %s); trying with longer timeout", actual, d)
				d = next
			}
			continue
		}

		break
	}
}

// There is a very similar copy of this in net/timeout_test.go.
func TestVariousDeadlines(t *testing.T) {
	t.Parallel()
	testVariousDeadlines(t)
}

// There is a very similar copy of this in net/timeout_test.go.
func TestVariousDeadlines1Proc(t *testing.T) {
	// Cannot use t.Parallel - modifies global GOMAXPROCS.
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	testVariousDeadlines(t)
}

// There is a very similar copy of this in net/timeout_test.go.
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
			t.Run(fmt.Sprintf("%v-%d", timeout, run+1), func { t ->
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
					n, err := io.Copy(io.Discard, r)
					dt := time.Since(t0)
					r.Close()
					actvch <- result{n, err, dt}
				}()

				select {
				case res := <-actvch:
					if !isDeadlineExceeded(err) {
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

// There is a very similar copy of this in net/timeout_test.go.
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
			if err != nil && !isDeadlineExceeded(err) {
				t.Error("Read returned non-timeout error", err)
			}
		}
	}()
	go func() {
		defer wg.Done()
		var b [1]byte
		for i := 0; i < N; i++ {
			_, err := w.Write(b[:])
			if err != nil && !isDeadlineExceeded(err) {
				t.Error("Write returned non-timeout error", err)
			}
		}
	}()
	wg.Wait() // wait for tester goroutine to stop
}

// TestRacyRead tests that it is safe to mutate the input Read buffer
// immediately after cancellation has occurred.
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
					if !isDeadlineExceeded(err) {
						t.Error(err)
					}
					r.SetReadDeadline(time.Now().Add(time.Millisecond))
				}
			}
		}()
	}
}

// TestRacyWrite tests that it is safe to mutate the input Write buffer
// immediately after cancellation has occurred.
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

	go io.Copy(io.Discard, r)

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
					if !isDeadlineExceeded(err) {
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
