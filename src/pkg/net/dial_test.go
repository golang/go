// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"runtime"
	"testing"
	"time"
)

func newLocalListener(t *testing.T) Listener {
	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		ln, err = Listen("tcp6", "[::1]:0")
	}
	if err != nil {
		t.Fatal(err)
	}
	return ln
}

func TestDialTimeout(t *testing.T) {
	ln := newLocalListener(t)
	defer ln.Close()

	errc := make(chan error)

	const SOMAXCONN = 0x80 // copied from syscall, but not always available
	const numConns = SOMAXCONN + 10

	// TODO(bradfitz): It's hard to test this in a portable
	// way. This is unforunate, but works for now.
	switch runtime.GOOS {
	case "linux":
		// The kernel will start accepting TCP connections before userspace
		// gets a chance to not accept them, so fire off a bunch to fill up
		// the kernel's backlog.  Then we test we get a failure after that.
		for i := 0; i < numConns; i++ {
			go func() {
				_, err := DialTimeout("tcp", ln.Addr().String(), 200*time.Millisecond)
				errc <- err
			}()
		}
	case "darwin", "windows":
		// At least OS X 10.7 seems to accept any number of
		// connections, ignoring listen's backlog, so resort
		// to connecting to a hopefully-dead 127/8 address.
		// Same for windows.
		go func() {
			_, err := DialTimeout("tcp", "127.0.71.111:80", 200*time.Millisecond)
			errc <- err
		}()
	default:
		// TODO(bradfitz):
		// OpenBSD may have a reject route to 10/8.
		// FreeBSD likely works, but is untested.
		t.Logf("skipping test on %q; untested.", runtime.GOOS)
		return
	}

	connected := 0
	for {
		select {
		case <-time.After(15 * time.Second):
			t.Fatal("too slow")
		case err := <-errc:
			if err == nil {
				connected++
				if connected == numConns {
					t.Fatal("all connections connected; expected some to time out")
				}
			} else {
				terr, ok := err.(timeout)
				if !ok {
					t.Fatalf("got error %q; want error with timeout interface", err)
				}
				if !terr.Timeout() {
					t.Fatalf("got error %q; not a timeout", err)
				}
				// Pass. We saw a timeout error.
				return
			}
		}
	}
}
