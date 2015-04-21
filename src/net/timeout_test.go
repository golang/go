// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/internal/socktest"
	"runtime"
	"testing"
	"time"
)

func TestDialTimeout(t *testing.T) {
	const T = 100 * time.Millisecond

	switch runtime.GOOS {
	case "plan9", "windows":
		origTestHookDialChannel := testHookDialChannel
		testHookDialChannel = func() { time.Sleep(2 * T) }
		defer func() { testHookDialChannel = origTestHookDialChannel }()
		if runtime.GOOS == "plan9" {
			break
		}
		fallthrough
	default:
		sw.Set(socktest.FilterConnect, func(so *socktest.Status) (socktest.AfterFilter, error) {
			time.Sleep(2 * T)
			return nil, errTimeout
		})
		defer sw.Set(socktest.FilterConnect, nil)
	}

	ch := make(chan error)
	go func() {
		// This dial never starts to send any SYN segment
		// because of above socket filter and test hook.
		c, err := DialTimeout("tcp", "127.0.0.1:0", T)
		if err == nil {
			err = fmt.Errorf("unexpectedly established: tcp:%s->%s", c.LocalAddr(), c.RemoteAddr())
			c.Close()
		}
		ch <- err
	}()
	tmo := time.NewTimer(3 * T)
	defer tmo.Stop()
	select {
	case <-tmo.C:
		t.Fatal("dial has not returned")
	case err := <-ch:
		if perr := parseDialError(err); perr != nil {
			t.Error(perr)
		}
		if !isTimeoutError(err) {
			t.Fatalf("got %v; want timeout", err)
		}
	}
}

type copyRes struct {
	n   int64
	err error
	d   time.Duration
}

func TestAcceptTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	ln.(*TCPListener).SetDeadline(time.Now().Add(-1 * time.Second))
	if _, err := ln.Accept(); !isTimeoutError(err) {
		t.Fatalf("Accept: expected err %v, got %v", errTimeout, err)
	}
	if perr := parseAcceptError(err); perr != nil {
		t.Error(perr)
	}
	if _, err := ln.Accept(); !isTimeoutError(err) {
		t.Fatalf("Accept: expected err %v, got %v", errTimeout, err)
	}
	if perr := parseAcceptError(err); perr != nil {
		t.Error(perr)
	}
	ln.(*TCPListener).SetDeadline(time.Now().Add(100 * time.Millisecond))
	if _, err := ln.Accept(); !isTimeoutError(err) {
		t.Fatalf("Accept: expected err %v, got %v", errTimeout, err)
	}
	if perr := parseAcceptError(err); perr != nil {
		t.Error(perr)
	}
	if _, err := ln.Accept(); !isTimeoutError(err) {
		t.Fatalf("Accept: expected err %v, got %v", errTimeout, err)
	}
	if perr := parseAcceptError(err); perr != nil {
		t.Error(perr)
	}
	ln.(*TCPListener).SetDeadline(noDeadline)
	errc := make(chan error)
	go func() {
		_, err := ln.Accept()
		errc <- err
	}()
	time.Sleep(100 * time.Millisecond)
	select {
	case err := <-errc:
		t.Fatalf("Expected Accept() to not return, but it returned with %v\n", err)
	default:
	}
	ln.Close()
	err = <-errc
	if perr := parseAcceptError(err); perr != nil {
		t.Error(perr)
	}
}

func TestReadTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	c, err := DialTCP("tcp", nil, ln.Addr().(*TCPAddr))
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer c.Close()
	c.SetDeadline(time.Now().Add(time.Hour))
	c.SetReadDeadline(time.Now().Add(-1 * time.Second))
	buf := make([]byte, 1)
	if _, err = c.Read(buf); !isTimeoutError(err) {
		t.Fatalf("Read: expected err %v, got %v", errTimeout, err)
	}
	if perr := parseReadError(err); perr != nil {
		t.Error(perr)
	}
	if _, err = c.Read(buf); !isTimeoutError(err) {
		t.Fatalf("Read: expected err %v, got %v", errTimeout, err)
	}
	if perr := parseReadError(err); perr != nil {
		t.Error(perr)
	}
	c.SetDeadline(time.Now().Add(100 * time.Millisecond))
	if _, err = c.Read(buf); !isTimeoutError(err) {
		t.Fatalf("Read: expected err %v, got %v", errTimeout, err)
	}
	if perr := parseReadError(err); perr != nil {
		t.Error(perr)
	}
	if _, err = c.Read(buf); !isTimeoutError(err) {
		t.Fatalf("Read: expected err %v, got %v", errTimeout, err)
	}
	if perr := parseReadError(err); perr != nil {
		t.Error(perr)
	}
	c.SetReadDeadline(noDeadline)
	c.SetWriteDeadline(time.Now().Add(-1 * time.Second))
	errc := make(chan error)
	go func() {
		_, err := c.Read(buf)
		errc <- err
	}()
	time.Sleep(100 * time.Millisecond)
	select {
	case err := <-errc:
		t.Fatalf("Expected Read() to not return, but it returned with %v\n", err)
	default:
	}
	c.Close()
	switch nerr := <-errc; err := nerr.(type) {
	case *OpError:
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
	default:
		if err == io.EOF && runtime.GOOS == "nacl" { // close enough; golang.org/issue/8044
			break
		}
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
	}
}

func TestWriteTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	c, err := DialTCP("tcp", nil, ln.Addr().(*TCPAddr))
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer c.Close()
	c.SetDeadline(time.Now().Add(time.Hour))
	c.SetWriteDeadline(time.Now().Add(-1 * time.Second))
	buf := make([]byte, 4096)
	writeUntilTimeout := func() {
		for {
			_, err := c.Write(buf)
			if perr := parseWriteError(err); perr != nil {
				t.Error(perr)
			}
			if err != nil {
				if isTimeoutError(err) {
					return
				}
				t.Fatalf("Write: expected err %v, got %v", errTimeout, err)
			}
		}
	}
	writeUntilTimeout()
	c.SetDeadline(time.Now().Add(10 * time.Millisecond))
	writeUntilTimeout()
	writeUntilTimeout()
	c.SetWriteDeadline(noDeadline)
	c.SetReadDeadline(time.Now().Add(-1 * time.Second))
	errc := make(chan error)
	go func() {
		for {
			_, err := c.Write(buf)
			if err != nil {
				errc <- err
			}
		}
	}()
	time.Sleep(100 * time.Millisecond)
	select {
	case err := <-errc:
		t.Fatalf("Expected Write() to not return, but it returned with %v\n", err)
	default:
	}
	c.Close()
	err = <-errc
	if perr := parseWriteError(err); perr != nil {
		t.Error(perr)
	}
}

func testTimeout(t *testing.T, net, addr string, readFrom bool) {
	c, err := Dial(net, addr)
	if err != nil {
		t.Errorf("Dial(%q, %q) failed: %v", net, addr, err)
		return
	}
	defer c.Close()
	what := "Read"
	if readFrom {
		what = "ReadFrom"
	}

	errc := make(chan error, 1)
	go func() {
		t0 := time.Now()
		c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		var b [100]byte
		var n int
		var err error
		if readFrom {
			n, _, err = c.(PacketConn).ReadFrom(b[0:])
		} else {
			n, err = c.Read(b[0:])
		}
		t1 := time.Now()
		if n != 0 || err == nil || !err.(Error).Timeout() {
			errc <- fmt.Errorf("%s(%q, %q) did not return 0, timeout: %v, %v", what, net, addr, n, err)
			return
		}
		if dt := t1.Sub(t0); dt < 50*time.Millisecond || !testing.Short() && dt > 250*time.Millisecond {
			errc <- fmt.Errorf("%s(%q, %q) took %s, expected 0.1s", what, net, addr, dt)
			return
		}
		errc <- nil
	}()
	select {
	case err := <-errc:
		if err != nil {
			t.Error(err)
		}
	case <-time.After(1 * time.Second):
		t.Errorf("%s(%q, %q) took over 1 second, expected 0.1s", what, net, addr)
	}
}

func TestTimeoutUDP(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	c, err := newLocalPacketListener("udp") // a listener that won't talk back
	if err != nil {
		t.Fatal(err)
	}

	testTimeout(t, "udp", c.LocalAddr().String(), false)
	testTimeout(t, "udp", c.LocalAddr().String(), true)
	c.Close()
}

func TestTimeoutTCP(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	handler := func(ls *localServer, ln Listener) { // a listener that won't talk back
		for {
			c, err := ln.Accept()
			if err != nil {
				break
			}
			defer c.Close()
		}
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	testTimeout(t, "tcp", ls.Listener.Addr().String(), false)
}

func TestDeadlineReset(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	tl := ln.(*TCPListener)
	tl.SetDeadline(time.Now().Add(1 * time.Minute))
	tl.SetDeadline(noDeadline) // reset it
	errc := make(chan error, 1)
	go func() {
		_, err := ln.Accept()
		errc <- err
	}()
	select {
	case <-time.After(50 * time.Millisecond):
		// Pass.
	case err := <-errc:
		// Accept should never return; we never
		// connected to it.
		t.Errorf("unexpected return from Accept; err=%v", err)
	}
}

func TestConcurrentAcceptTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	ln.(*TCPListener).SetDeadline(time.Now().Add(100 * time.Millisecond))
	errc := make(chan error, 1)
	go func() {
		_, err := ln.Accept()
		errc <- err
	}()
	select {
	case <-time.After(1 * time.Second):
		// Accept shouldn't block indefinitely
		t.Error("Accept didn't return in an expected time")
	case err := <-errc:
		if perr := parseAcceptError(err); perr != nil {
			t.Error(perr)
		}
	}
}

func TestReadWriteDeadline(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	const (
		readTimeout  = 50 * time.Millisecond
		writeTimeout = 250 * time.Millisecond
	)
	checkTimeout := func(command string, start time.Time, should time.Duration) {
		is := time.Now().Sub(start)
		d := is - should
		if d < -30*time.Millisecond || !testing.Short() && 150*time.Millisecond < d {
			t.Errorf("%s timeout test failed: is=%v should=%v\n", command, is, should)
		}
	}

	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("ListenTCP on :0: %v", err)
	}
	defer ln.Close()

	lnquit := make(chan bool)

	go func() {
		c, err := ln.Accept()
		if err != nil {
			t.Errorf("Accept: %v", err)
			return
		}
		defer c.Close()
		lnquit <- true
	}()

	c, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()

	start := time.Now()
	err = c.SetReadDeadline(start.Add(readTimeout))
	if err != nil {
		t.Fatalf("SetReadDeadline: %v", err)
	}
	err = c.SetWriteDeadline(start.Add(writeTimeout))
	if err != nil {
		t.Fatalf("SetWriteDeadline: %v", err)
	}

	quit := make(chan bool)

	go func() {
		var buf [10]byte
		_, err := c.Read(buf[:])
		if err == nil {
			t.Errorf("Read should not succeed")
		}
		checkTimeout("Read", start, readTimeout)
		quit <- true
	}()

	go func() {
		var buf [10000]byte
		for {
			_, err := c.Write(buf[:])
			if err != nil {
				break
			}
		}
		checkTimeout("Write", start, writeTimeout)
		quit <- true
	}()

	<-quit
	<-quit
	<-lnquit
}

type neverEnding byte

func (b neverEnding) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

func TestVariousDeadlines1Proc(t *testing.T) {
	testVariousDeadlines(t, 1)
}

func TestVariousDeadlines4Proc(t *testing.T) {
	testVariousDeadlines(t, 4)
}

func testVariousDeadlines(t *testing.T, maxProcs int) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(maxProcs))

	acceptc := make(chan error, 1)
	// The server, with no timeouts of its own, sending bytes to clients
	// as fast as it can.
	servec := make(chan copyRes)
	handler := func(ls *localServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				acceptc <- err
				return
			}
			go func() {
				t0 := time.Now()
				n, err := io.Copy(c, neverEnding('a'))
				d := time.Since(t0)
				c.Close()
				servec <- copyRes{n, err, d}
			}()
		}
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
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
			name := fmt.Sprintf("%v run %d/%d", timeout, run+1, numRuns)
			t.Log(name)

			c, err := Dial("tcp", ls.Listener.Addr().String())
			if err != nil {
				t.Fatalf("Dial: %v", err)
			}
			clientc := make(chan copyRes)
			go func() {
				t0 := time.Now()
				c.SetDeadline(t0.Add(timeout))
				n, err := io.Copy(ioutil.Discard, c)
				d := time.Since(t0)
				c.Close()
				clientc <- copyRes{n, err, d}
			}()

			tooLong := 5 * time.Second
			select {
			case res := <-clientc:
				if isTimeoutError(res.err) {
					t.Logf("for %v, good client timeout after %v, reading %d bytes", name, res.d, res.n)
				} else {
					t.Fatalf("for %v: client Copy = %d, %v (want timeout)", name, res.n, res.err)
				}
			case <-time.After(tooLong):
				t.Fatalf("for %v: timeout (%v) waiting for client to timeout (%v) reading", name, tooLong, timeout)
			}

			select {
			case res := <-servec:
				t.Logf("for %v: server in %v wrote %d, %v", name, res.d, res.n, res.err)
			case err := <-acceptc:
				t.Fatalf("for %v: server Accept = %v", name, err)
			case <-time.After(tooLong):
				t.Fatalf("for %v, timeout waiting for server to finish writing", name)
			}
		}
	}
}

// TestReadDeadlineDataAvailable tests that read deadlines work, even
// if there's data ready to be read.
func TestReadDeadlineDataAvailable(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	servec := make(chan copyRes)
	const msg = "data client shouldn't read, even though it'll be waiting"
	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			t.Errorf("Accept: %v", err)
			return
		}
		defer c.Close()
		n, err := c.Write([]byte(msg))
		servec <- copyRes{n: int64(n), err: err}
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	c, err := Dial("tcp", ls.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()
	if res := <-servec; res.err != nil || res.n != int64(len(msg)) {
		t.Fatalf("unexpected server Write: n=%d, err=%v; want n=%d, err=nil", res.n, res.err, len(msg))
	}
	c.SetReadDeadline(time.Now().Add(-5 * time.Second)) // in the psat.
	buf := make([]byte, len(msg)/2)
	n, err := c.Read(buf)
	if perr := parseReadError(err); perr != nil {
		t.Error(perr)
	}
	if n > 0 || !isTimeoutError(err) {
		t.Fatalf("client read = %d (%q) err=%v; want 0, timeout", n, buf[:n], err)
	}
}

// TestWriteDeadlineBufferAvailable tests that write deadlines work, even
// if there's buffer space available to write.
func TestWriteDeadlineBufferAvailable(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	servec := make(chan copyRes)
	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			t.Errorf("Accept: %v", err)
			return
		}
		defer c.Close()
		c.SetWriteDeadline(time.Now().Add(-5 * time.Second)) // in the past
		n, err := c.Write([]byte{'x'})
		servec <- copyRes{n: int64(n), err: err}
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	c, err := Dial("tcp", ls.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()
	res := <-servec
	if res.n != 0 {
		t.Errorf("Write = %d; want 0", res.n)
	}
	if perr := parseWriteError(res.err); perr != nil {
		t.Error(perr)
	}
	if !isTimeoutError(res.err) {
		t.Errorf("Write error = %v; want timeout", res.err)
	}
}

// TestAcceptDeadlineConnectionAvailable tests that accept deadlines work, even
// if there's incoming connections available.
func TestAcceptDeadlineConnectionAvailable(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	go func() {
		c, err := Dial("tcp", ln.Addr().String())
		if err != nil {
			t.Errorf("Dial: %v", err)
			return
		}
		defer c.Close()
		var buf [1]byte
		c.Read(buf[:]) // block until the connection or listener is closed
	}()
	time.Sleep(10 * time.Millisecond)
	ln.(*TCPListener).SetDeadline(time.Now().Add(-5 * time.Second)) // in the past
	c, err := ln.Accept()
	if err == nil {
		defer c.Close()
	}
	if !isTimeoutError(err) {
		t.Fatalf("Accept: got %v; want timeout", err)
	}
}

// TestConnectDeadlineInThePast tests that connect deadlines work, even
// if the connection can be established w/o blocking.
func TestConnectDeadlineInThePast(t *testing.T) {
	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	go func() {
		c, err := ln.Accept()
		if err == nil {
			defer c.Close()
		}
	}()
	time.Sleep(10 * time.Millisecond)
	c, err := DialTimeout("tcp", ln.Addr().String(), -5*time.Second) // in the past
	if err == nil {
		defer c.Close()
	}
	if perr := parseDialError(err); perr != nil {
		t.Error(perr)
	}
	if !isTimeoutError(err) {
		t.Fatalf("got %v; want timeout", err)
	}
}

// TestProlongTimeout tests concurrent deadline modification.
// Known to cause data races in the past.
func TestProlongTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	connected := make(chan bool)
	handler := func(ls *localServer, ln Listener) {
		s, err := ln.Accept()
		connected <- true
		if err != nil {
			t.Errorf("ln.Accept: %v", err)
			return
		}
		defer s.Close()
		s.SetDeadline(time.Now().Add(time.Hour))
		go func() {
			var buf [4096]byte
			for {
				_, err := s.Write(buf[:])
				if err != nil {
					break
				}
				s.SetDeadline(time.Now().Add(time.Hour))
			}
		}()
		buf := make([]byte, 1)
		for {
			_, err := s.Read(buf)
			if err != nil {
				break
			}
			s.SetDeadline(time.Now().Add(time.Hour))
		}
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	c, err := Dial("tcp", ls.Listener.Addr().String())
	if err != nil {
		t.Fatalf("DialTCP: %v", err)
	}
	defer c.Close()
	<-connected
	for i := 0; i < 1024; i++ {
		var buf [1]byte
		c.Write(buf[:])
	}
}

func TestDeadlineRace(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	N := 1000
	if testing.Short() {
		N = 50
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	c, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()
	done := make(chan bool)
	go func() {
		t := time.NewTicker(2 * time.Microsecond).C
		for i := 0; i < N; i++ {
			if err := c.SetDeadline(time.Now().Add(2 * time.Microsecond)); err != nil {
				break
			}
			<-t
		}
		done <- true
	}()
	var buf [1]byte
	for i := 0; i < N; i++ {
		c.Read(buf[:]) // ignore possible timeout errors
	}
	c.Close()
	<-done
}
