// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
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
	t0 := time.Now()
	fd.SetReadTimeout(1e8) // 100ms
	var b [100]byte
	var n int
	var err1 error
	if readFrom {
		n, _, err1 = fd.(PacketConn).ReadFrom(b[0:])
	} else {
		n, err1 = fd.Read(b[0:])
	}
	t1 := time.Now()
	what := "Read"
	if readFrom {
		what = "ReadFrom"
	}
	if n != 0 || err1 == nil || !err1.(Error).Timeout() {
		t.Errorf("fd.%s on %s %s did not return 0, timeout: %v, %v", what, network, addr, n, err1)
	}
	if dt := t1.Sub(t0); dt < 50*time.Millisecond || dt > 150*time.Millisecond {
		t.Errorf("fd.%s on %s %s took %s, expected 0.1s", what, network, addr, dt)
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
