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
