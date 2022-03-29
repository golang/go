// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

package net

import (
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"net/internal/socktest"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"
)

var dialTimeoutTests = []struct {
	timeout time.Duration
	delta   time.Duration // for deadline

	guard time.Duration
	max   time.Duration
}{
	// Tests that dial timeouts, deadlines in the past work.
	{-5 * time.Second, 0, -5 * time.Second, 100 * time.Millisecond},
	{0, -5 * time.Second, -5 * time.Second, 100 * time.Millisecond},
	{-5 * time.Second, 5 * time.Second, -5 * time.Second, 100 * time.Millisecond}, // timeout over deadline
	{-1 << 63, 0, time.Second, 100 * time.Millisecond},
	{0, -1 << 63, time.Second, 100 * time.Millisecond},

	{50 * time.Millisecond, 0, 100 * time.Millisecond, time.Second},
	{0, 50 * time.Millisecond, 100 * time.Millisecond, time.Second},
	{50 * time.Millisecond, 5 * time.Second, 100 * time.Millisecond, time.Second}, // timeout over deadline
}

func TestDialTimeout(t *testing.T) {
	// Cannot use t.Parallel - modifies global hooks.
	origTestHookDialChannel := testHookDialChannel
	defer func() { testHookDialChannel = origTestHookDialChannel }()
	defer sw.Set(socktest.FilterConnect, nil)

	for i, tt := range dialTimeoutTests {
		switch runtime.GOOS {
		case "plan9", "windows":
			testHookDialChannel = func() { time.Sleep(tt.guard) }
			if runtime.GOOS == "plan9" {
				break
			}
			fallthrough
		default:
			sw.Set(socktest.FilterConnect, func(so *socktest.Status) (socktest.AfterFilter, error) {
				time.Sleep(tt.guard)
				return nil, errTimedout
			})
		}

		ch := make(chan error)
		d := Dialer{Timeout: tt.timeout}
		if tt.delta != 0 {
			d.Deadline = time.Now().Add(tt.delta)
		}
		max := time.NewTimer(tt.max)
		defer max.Stop()
		go func() {
			// This dial never starts to send any TCP SYN
			// segment because of above socket filter and
			// test hook.
			c, err := d.Dial("tcp", "127.0.0.1:0")
			if err == nil {
				err = fmt.Errorf("unexpectedly established: tcp:%s->%s", c.LocalAddr(), c.RemoteAddr())
				c.Close()
			}
			ch <- err
		}()

		select {
		case <-max.C:
			t.Fatalf("#%d: Dial didn't return in an expected time", i)
		case err := <-ch:
			if perr := parseDialError(err); perr != nil {
				t.Errorf("#%d: %v", i, perr)
			}
			if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
				t.Fatalf("#%d: %v", i, err)
			}
		}
	}
}

func TestDialTimeoutMaxDuration(t *testing.T) {
	ln := newLocalListener(t, "tcp")
	defer func() {
		if err := ln.Close(); err != nil {
			t.Error(err)
		}
	}()

	for _, tt := range []struct {
		timeout time.Duration
		delta   time.Duration // for deadline
	}{
		// Large timeouts that will overflow an int64 unix nanos.
		{1<<63 - 1, 0},
		{0, 1<<63 - 1},
	} {
		t.Run(fmt.Sprintf("timeout=%s/delta=%s", tt.timeout, tt.delta), func(t *testing.T) {
			d := Dialer{Timeout: tt.timeout}
			if tt.delta != 0 {
				d.Deadline = time.Now().Add(tt.delta)
			}
			c, err := d.Dial(ln.Addr().Network(), ln.Addr().String())
			if err != nil {
				t.Fatal(err)
			}
			if err := c.Close(); err != nil {
				t.Error(err)
			}
		})
	}
}

var acceptTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that accept deadlines in the past work, even if
	// there's incoming connections available.
	{-5 * time.Second, [2]error{os.ErrDeadlineExceeded, os.ErrDeadlineExceeded}},

	{50 * time.Millisecond, [2]error{nil, os.ErrDeadlineExceeded}},
}

func TestAcceptTimeout(t *testing.T) {
	testenv.SkipFlaky(t, 17948)
	t.Parallel()

	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	var wg sync.WaitGroup
	for i, tt := range acceptTimeoutTests {
		if tt.timeout < 0 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				d := Dialer{Timeout: 100 * time.Millisecond}
				c, err := d.Dial(ln.Addr().Network(), ln.Addr().String())
				if err != nil {
					t.Error(err)
					return
				}
				c.Close()
			}()
		}

		if err := ln.(*TCPListener).SetDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("$%d: %v", i, err)
		}
		for j, xerr := range tt.xerrs {
			for {
				c, err := ln.Accept()
				if xerr != nil {
					if perr := parseAcceptError(err); perr != nil {
						t.Errorf("#%d/%d: %v", i, j, perr)
					}
					if !isDeadlineExceeded(err) {
						t.Fatalf("#%d/%d: %v", i, j, err)
					}
				}
				if err == nil {
					c.Close()
					time.Sleep(10 * time.Millisecond)
					continue
				}
				break
			}
		}
	}
	wg.Wait()
}

func TestAcceptTimeoutMustReturn(t *testing.T) {
	t.Parallel()

	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	max := time.NewTimer(time.Second)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := ln.(*TCPListener).SetDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		if err := ln.(*TCPListener).SetDeadline(time.Now().Add(10 * time.Millisecond)); err != nil {
			t.Error(err)
		}
		c, err := ln.Accept()
		if err == nil {
			c.Close()
		}
		ch <- err
	}()

	select {
	case <-max.C:
		ln.Close()
		<-ch // wait for tester goroutine to stop
		t.Fatal("Accept didn't return in an expected time")
	case err := <-ch:
		if perr := parseAcceptError(err); perr != nil {
			t.Error(perr)
		}
		if !isDeadlineExceeded(err) {
			t.Fatal(err)
		}
	}
}

func TestAcceptTimeoutMustNotReturn(t *testing.T) {
	t.Parallel()

	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	max := time.NewTimer(100 * time.Millisecond)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := ln.(*TCPListener).SetDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := ln.(*TCPListener).SetDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		_, err := ln.Accept()
		ch <- err
	}()

	select {
	case err := <-ch:
		if perr := parseAcceptError(err); perr != nil {
			t.Error(perr)
		}
		t.Fatalf("expected Accept to not return, but it returned with %v", err)
	case <-max.C:
		ln.Close()
		<-ch // wait for tester goroutine to stop
	}
}

var readTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that read deadlines work, even if there's data ready
	// to be read.
	{-5 * time.Second, [2]error{os.ErrDeadlineExceeded, os.ErrDeadlineExceeded}},

	{50 * time.Millisecond, [2]error{nil, os.ErrDeadlineExceeded}},
}

func TestReadTimeout(t *testing.T) {
	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		c.Write([]byte("READ TIMEOUT TEST"))
		defer c.Close()
	}
	ls := newLocalServer(t, "tcp")
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	for i, tt := range readTimeoutTests {
		if err := c.SetReadDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		var b [1]byte
		for j, xerr := range tt.xerrs {
			for {
				n, err := c.Read(b[:])
				if xerr != nil {
					if perr := parseReadError(err); perr != nil {
						t.Errorf("#%d/%d: %v", i, j, perr)
					}
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

func TestReadTimeoutMustNotReturn(t *testing.T) {
	t.Parallel()

	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	max := time.NewTimer(100 * time.Millisecond)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := c.SetDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := c.SetWriteDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := c.SetReadDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		var b [1]byte
		_, err := c.Read(b[:])
		ch <- err
	}()

	select {
	case err := <-ch:
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
		t.Fatalf("expected Read to not return, but it returned with %v", err)
	case <-max.C:
		c.Close()
		err := <-ch // wait for tester goroutine to stop
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
		if nerr, ok := err.(Error); !ok || nerr.Timeout() || nerr.Temporary() {
			t.Fatal(err)
		}
	}
}

var readFromTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that read deadlines work, even if there's data ready
	// to be read.
	{-5 * time.Second, [2]error{os.ErrDeadlineExceeded, os.ErrDeadlineExceeded}},

	{50 * time.Millisecond, [2]error{nil, os.ErrDeadlineExceeded}},
}

func TestReadFromTimeout(t *testing.T) {
	ch := make(chan Addr)
	defer close(ch)
	handler := func(ls *localPacketServer, c PacketConn) {
		if dst, ok := <-ch; ok {
			c.WriteTo([]byte("READFROM TIMEOUT TEST"), dst)
		}
	}
	ls := newLocalPacketServer(t, "udp")
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	host, _, err := SplitHostPort(ls.PacketConn.LocalAddr().String())
	if err != nil {
		t.Fatal(err)
	}
	c, err := ListenPacket(ls.PacketConn.LocalAddr().Network(), JoinHostPort(host, "0"))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	ch <- c.LocalAddr()

	for i, tt := range readFromTimeoutTests {
		if err := c.SetReadDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		var b [1]byte
		for j, xerr := range tt.xerrs {
			for {
				n, _, err := c.ReadFrom(b[:])
				if xerr != nil {
					if perr := parseReadError(err); perr != nil {
						t.Errorf("#%d/%d: %v", i, j, perr)
					}
					if !isDeadlineExceeded(err) {
						t.Fatalf("#%d/%d: %v", i, j, err)
					}
				}
				if err == nil {
					time.Sleep(tt.timeout / 3)
					continue
				}
				if nerr, ok := err.(Error); ok && nerr.Timeout() && n != 0 {
					t.Fatalf("#%d/%d: read %d; want 0", i, j, n)
				}
				break
			}
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

func TestWriteTimeout(t *testing.T) {
	t.Parallel()

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	for i, tt := range writeTimeoutTests {
		c, err := Dial(ln.Addr().Network(), ln.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		if err := c.SetWriteDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		for j, xerr := range tt.xerrs {
			for {
				n, err := c.Write([]byte("WRITE TIMEOUT TEST"))
				if xerr != nil {
					if perr := parseWriteError(err); perr != nil {
						t.Errorf("#%d/%d: %v", i, j, perr)
					}
					if !isDeadlineExceeded(err) {
						t.Fatalf("#%d/%d: %v", i, j, err)
					}
				}
				if err == nil {
					time.Sleep(tt.timeout / 3)
					continue
				}
				if n != 0 {
					t.Fatalf("#%d/%d: wrote %d; want 0", i, j, n)
				}
				break
			}
		}
	}
}

func TestWriteTimeoutMustNotReturn(t *testing.T) {
	t.Parallel()

	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	max := time.NewTimer(100 * time.Millisecond)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := c.SetDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := c.SetReadDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := c.SetWriteDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		var b [1]byte
		for {
			if _, err := c.Write(b[:]); err != nil {
				ch <- err
				break
			}
		}
	}()

	select {
	case err := <-ch:
		if perr := parseWriteError(err); perr != nil {
			t.Error(perr)
		}
		t.Fatalf("expected Write to not return, but it returned with %v", err)
	case <-max.C:
		c.Close()
		err := <-ch // wait for tester goroutine to stop
		if perr := parseWriteError(err); perr != nil {
			t.Error(perr)
		}
		if nerr, ok := err.(Error); !ok || nerr.Timeout() || nerr.Temporary() {
			t.Fatal(err)
		}
	}
}

func TestWriteToTimeout(t *testing.T) {
	t.Parallel()

	c1 := newLocalPacketListener(t, "udp")
	defer c1.Close()

	host, _, err := SplitHostPort(c1.LocalAddr().String())
	if err != nil {
		t.Fatal(err)
	}

	timeouts := []time.Duration{
		-5 * time.Second,
		10 * time.Millisecond,
	}

	for _, timeout := range timeouts {
		t.Run(fmt.Sprint(timeout), func(t *testing.T) {
			c2, err := ListenPacket(c1.LocalAddr().Network(), JoinHostPort(host, "0"))
			if err != nil {
				t.Fatal(err)
			}
			defer c2.Close()

			if err := c2.SetWriteDeadline(time.Now().Add(timeout)); err != nil {
				t.Fatalf("SetWriteDeadline: %v", err)
			}
			backoff := 1 * time.Millisecond
			nDeadlineExceeded := 0
			for j := 0; nDeadlineExceeded < 2; j++ {
				n, err := c2.WriteTo([]byte("WRITETO TIMEOUT TEST"), c1.LocalAddr())
				t.Logf("#%d: WriteTo: %d, %v", j, n, err)
				if err == nil && timeout >= 0 && nDeadlineExceeded == 0 {
					// If the timeout is nonnegative, some number of WriteTo calls may
					// succeed before the timeout takes effect.
					t.Logf("WriteTo succeeded; sleeping %v", timeout/3)
					time.Sleep(timeout / 3)
					continue
				}
				if isENOBUFS(err) {
					t.Logf("WriteTo: %v", err)
					// We're looking for a deadline exceeded error, but if the kernel's
					// network buffers are saturated we may see ENOBUFS instead (see
					// https://go.dev/issue/49930). Give it some time to unsaturate.
					time.Sleep(backoff)
					backoff *= 2
					continue
				}
				if perr := parseWriteError(err); perr != nil {
					t.Errorf("failed to parse error: %v", perr)
				}
				if !isDeadlineExceeded(err) {
					t.Errorf("error is not 'deadline exceeded'")
				}
				if n != 0 {
					t.Errorf("unexpectedly wrote %d bytes", n)
				}
				if !t.Failed() {
					t.Logf("WriteTo timed out as expected")
				}
				nDeadlineExceeded++
			}
		})
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
	// tests that automatically increase timeouts until succeess.
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

func TestReadTimeoutFluctuation(t *testing.T) {
	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	d := minDynamicTimeout
	b := make([]byte, 256)
	for {
		t.Logf("SetReadDeadline(+%v)", d)
		t0 := time.Now()
		deadline := t0.Add(d)
		if err = c.SetReadDeadline(deadline); err != nil {
			t.Fatalf("SetReadDeadline(%v): %v", deadline, err)
		}
		var n int
		n, err = c.Read(b)
		t1 := time.Now()

		if n != 0 || err == nil || !err.(Error).Timeout() {
			t.Errorf("Read did not return (0, timeout): (%d, %v)", n, err)
		}
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
		if !isDeadlineExceeded(err) {
			t.Errorf("Read error is not DeadlineExceeded: %v", err)
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

func TestReadFromTimeoutFluctuation(t *testing.T) {
	c1 := newLocalPacketListener(t, "udp")
	defer c1.Close()

	c2, err := Dial(c1.LocalAddr().Network(), c1.LocalAddr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c2.Close()

	d := minDynamicTimeout
	b := make([]byte, 256)
	for {
		t.Logf("SetReadDeadline(+%v)", d)
		t0 := time.Now()
		deadline := t0.Add(d)
		if err = c2.SetReadDeadline(deadline); err != nil {
			t.Fatalf("SetReadDeadline(%v): %v", deadline, err)
		}
		var n int
		n, _, err = c2.(PacketConn).ReadFrom(b)
		t1 := time.Now()

		if n != 0 || err == nil || !err.(Error).Timeout() {
			t.Errorf("ReadFrom did not return (0, timeout): (%d, %v)", n, err)
		}
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
		if !isDeadlineExceeded(err) {
			t.Errorf("ReadFrom error is not DeadlineExceeded: %v", err)
		}

		actual := t1.Sub(t0)
		if t1.Before(deadline) {
			t.Errorf("ReadFrom took %s; expected at least %s", actual, d)
		}
		if t.Failed() {
			return
		}
		if want := timeoutUpperBound(d); actual > want {
			next, ok := nextTimeout(actual)
			if !ok {
				t.Fatalf("ReadFrom took %s; expected at most %s", actual, want)
			}
			// Maybe this machine is too slow to reliably schedule goroutines within
			// the requested duration. Increase the timeout and try again.
			t.Logf("ReadFrom took %s (expected %s); trying with longer timeout", actual, d)
			d = next
			continue
		}

		break
	}
}

func TestWriteTimeoutFluctuation(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	d := minDynamicTimeout
	for {
		t.Logf("SetWriteDeadline(+%v)", d)
		t0 := time.Now()
		deadline := t0.Add(d)
		if err = c.SetWriteDeadline(deadline); err != nil {
			t.Fatalf("SetWriteDeadline(%v): %v", deadline, err)
		}
		var n int64
		for {
			var dn int
			dn, err = c.Write([]byte("TIMEOUT TRANSMITTER"))
			n += int64(dn)
			if err != nil {
				break
			}
		}
		t1 := time.Now()

		if err == nil || !err.(Error).Timeout() {
			t.Fatalf("Write did not return (any, timeout): (%d, %v)", n, err)
		}
		if perr := parseWriteError(err); perr != nil {
			t.Error(perr)
		}
		if !isDeadlineExceeded(err) {
			t.Errorf("Write error is not DeadlineExceeded: %v", err)
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
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test on plan9; see golang.org/issue/26945")
	}
	type result struct {
		n   int64
		err error
		d   time.Duration
	}

	handler := func(ls *localServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				break
			}
			c.Read(make([]byte, 1)) // wait for client to close connection
			c.Close()
		}
	}
	ls := newLocalServer(t, "tcp")
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
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
			name := fmt.Sprintf("%v %d/%d", timeout, run, numRuns)
			t.Log(name)

			c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
			if err != nil {
				t.Fatal(err)
			}

			t0 := time.Now()
			if err := c.SetDeadline(t0.Add(timeout)); err != nil {
				t.Error(err)
			}
			n, err := io.Copy(io.Discard, c)
			dt := time.Since(t0)
			c.Close()

			if nerr, ok := err.(Error); ok && nerr.Timeout() {
				t.Logf("%v: good timeout after %v; %d bytes", name, dt, n)
			} else {
				t.Fatalf("%v: Copy = %d, %v; want timeout", name, n, err)
			}
		}
	}
}

// TestReadWriteProlongedTimeout tests concurrent deadline
// modification. Known to cause data races in the past.
func TestReadWriteProlongedTimeout(t *testing.T) {
	t.Parallel()

	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		defer c.Close()

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			var b [1]byte
			for {
				if err := c.SetReadDeadline(time.Now().Add(time.Hour)); err != nil {
					if perr := parseCommonError(err); perr != nil {
						t.Error(perr)
					}
					t.Error(err)
					return
				}
				if _, err := c.Read(b[:]); err != nil {
					if perr := parseReadError(err); perr != nil {
						t.Error(perr)
					}
					return
				}
			}
		}()
		go func() {
			defer wg.Done()
			var b [1]byte
			for {
				if err := c.SetWriteDeadline(time.Now().Add(time.Hour)); err != nil {
					if perr := parseCommonError(err); perr != nil {
						t.Error(perr)
					}
					t.Error(err)
					return
				}
				if _, err := c.Write(b[:]); err != nil {
					if perr := parseWriteError(err); perr != nil {
						t.Error(perr)
					}
					return
				}
			}
		}()
		wg.Wait()
	}
	ls := newLocalServer(t, "tcp")
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	var b [1]byte
	for i := 0; i < 1000; i++ {
		c.Write(b[:])
		c.Read(b[:])
	}
}

func TestReadWriteDeadlineRace(t *testing.T) {
	t.Parallel()

	N := 1000
	if testing.Short() {
		N = 50
	}

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		tic := time.NewTicker(2 * time.Microsecond)
		defer tic.Stop()
		for i := 0; i < N; i++ {
			if err := c.SetReadDeadline(time.Now().Add(2 * time.Microsecond)); err != nil {
				if perr := parseCommonError(err); perr != nil {
					t.Error(perr)
				}
				break
			}
			if err := c.SetWriteDeadline(time.Now().Add(2 * time.Microsecond)); err != nil {
				if perr := parseCommonError(err); perr != nil {
					t.Error(perr)
				}
				break
			}
			<-tic.C
		}
	}()
	go func() {
		defer wg.Done()
		var b [1]byte
		for i := 0; i < N; i++ {
			c.Read(b[:]) // ignore possible timeout errors
		}
	}()
	go func() {
		defer wg.Done()
		var b [1]byte
		for i := 0; i < N; i++ {
			c.Write(b[:]) // ignore possible timeout errors
		}
	}()
	wg.Wait() // wait for tester goroutine to stop
}

// Issue 35367.
func TestConcurrentSetDeadline(t *testing.T) {
	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	const goroutines = 8
	const conns = 10
	const tries = 100

	var c [conns]Conn
	for i := 0; i < conns; i++ {
		var err error
		c[i], err = Dial(ln.Addr().Network(), ln.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer c[i].Close()
	}

	var wg sync.WaitGroup
	wg.Add(goroutines)
	now := time.Now()
	for i := 0; i < goroutines; i++ {
		go func(i int) {
			defer wg.Done()
			// Make the deadlines steadily earlier,
			// to trigger runtime adjusttimers calls.
			for j := tries; j > 0; j-- {
				for k := 0; k < conns; k++ {
					c[k].SetReadDeadline(now.Add(2*time.Hour + time.Duration(i*j*k)*time.Second))
					c[k].SetWriteDeadline(now.Add(1*time.Hour + time.Duration(i*j*k)*time.Second))
				}
			}
		}(i)
	}
	wg.Wait()
}

// isDeadlineExceeded reports whether err is or wraps os.ErrDeadlineExceeded.
// We also check that the error implements net.Error, and that the
// Timeout method returns true.
func isDeadlineExceeded(err error) bool {
	nerr, ok := err.(Error)
	if !ok {
		return false
	}
	if !nerr.Timeout() {
		return false
	}
	if !errors.Is(err, os.ErrDeadlineExceeded) {
		return false
	}
	return true
}
