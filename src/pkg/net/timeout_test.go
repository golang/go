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

func testTimeout(t *testing.T, network, addr string, readFrom bool) {
	fd, err := Dial(network, addr)
	if err != nil {
		t.Errorf("dial %s %s failed: %v", network, addr, err)
		return
	}
	defer fd.Close()
	what := "Read"
	if readFrom {
		what = "ReadFrom"
	}

	errc := make(chan error, 1)
	go func() {
		t0 := time.Now()
		fd.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		var b [100]byte
		var n int
		var err1 error
		if readFrom {
			n, _, err1 = fd.(PacketConn).ReadFrom(b[0:])
		} else {
			n, err1 = fd.Read(b[0:])
		}
		t1 := time.Now()
		if n != 0 || err1 == nil || !err1.(Error).Timeout() {
			errc <- fmt.Errorf("fd.%s on %s %s did not return 0, timeout: %v, %v", what, network, addr, n, err1)
			return
		}
		if dt := t1.Sub(t0); dt < 50*time.Millisecond || !testing.Short() && dt > 250*time.Millisecond {
			errc <- fmt.Errorf("fd.%s on %s %s took %s, expected 0.1s", what, network, addr, dt)
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
		t.Errorf("%s on %s %s took over 1 second, expected 0.1s", what, network, addr)
	}
}

func TestTimeoutUDP(t *testing.T) {
	if runtime.GOOS == "plan9" {
		return
	}
	testTimeout(t, "udp", "127.0.0.1:53", false)
	testTimeout(t, "udp", "127.0.0.1:53", true)
}

func TestTimeoutTCP(t *testing.T) {
	if runtime.GOOS == "plan9" {
		return
	}
	// set up a listener that won't talk back
	listening := make(chan string)
	done := make(chan int)
	go runServe(t, "tcp", "127.0.0.1:0", listening, done)
	addr := <-listening

	testTimeout(t, "tcp", addr, false)
	<-done
}

func TestDeadlineReset(t *testing.T) {
	if runtime.GOOS == "plan9" {
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
