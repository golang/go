// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"runtime"
	"testing"
	"time"
)

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
		t.Logf("skipping test on %q", runtime.GOOS)
		return
	}

	// set up a listener that won't talk back
	listening := make(chan string)
	done := make(chan int)
	go runDatagramPacketConnServer(t, "udp", "127.0.0.1:0", listening, done)
	addr := <-listening

	testTimeout(t, "udp", addr, false)
	testTimeout(t, "udp", addr, true)
	<-done
}

func TestTimeoutTCP(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Logf("skipping test on %q", runtime.GOOS)
		return
	}

	// set up a listener that won't talk back
	listening := make(chan string)
	done := make(chan int)
	go runStreamConnServer(t, "tcp", "127.0.0.1:0", listening, done)
	addr := <-listening

	testTimeout(t, "tcp", addr, false)
	<-done
}

func TestDeadlineReset(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Logf("skipping test on %q", runtime.GOOS)
		return
	}
	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	tl := ln.(*TCPListener)
	tl.SetDeadline(time.Now().Add(1 * time.Minute))
	tl.SetDeadline(time.Time{}) // reset it
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

func TestTimeoutAccept(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Logf("skipping test on %q", runtime.GOOS)
		return
	}
	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	tl := ln.(*TCPListener)
	tl.SetDeadline(time.Now().Add(100 * time.Millisecond))
	errc := make(chan error, 1)
	go func() {
		_, err := ln.Accept()
		errc <- err
	}()
	select {
	case <-time.After(1 * time.Second):
		// Accept shouldn't block indefinitely
		t.Errorf("Accept didn't return in an expected time")
	case <-errc:
		// Pass.
	}
}

func TestReadWriteDeadline(t *testing.T) {
	if !canCancelIO {
		t.Logf("skipping test on this system")
		return
	}
	const (
		readTimeout  = 100 * time.Millisecond
		writeTimeout = 200 * time.Millisecond
		delta        = 40 * time.Millisecond
	)
	checkTimeout := func(command string, start time.Time, should time.Duration) {
		is := time.Now().Sub(start)
		d := should - is
		if d < -delta || delta < d {
			t.Errorf("%s timeout test failed: is=%v should=%v\n", command, is, should)
		}
	}

	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("ListenTCP on :0: %v", err)
	}

	lnquit := make(chan bool)

	go func() {
		c, err := ln.Accept()
		if err != nil {
			t.Fatalf("Accept: %v", err)
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
		_, err = c.Read(buf[:])
		if err == nil {
			t.Errorf("Read should not succeed")
		}
		checkTimeout("Read", start, readTimeout)
		quit <- true
	}()

	go func() {
		var buf [10000]byte
		for {
			_, err = c.Write(buf[:])
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
