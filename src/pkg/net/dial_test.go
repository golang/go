// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag"
	"regexp"
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

	numConns := listenerBacklog + 10

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
	case "darwin":
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
		// OpenBSD may have a reject route to 127/8 except 127.0.0.1/32
		// by default. FreeBSD likely works, but is untested.
		// TODO(rsc):
		// The timeout never happens on Windows.  Why?  Issue 3016.
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

func TestSelfConnect(t *testing.T) {
	if runtime.GOOS == "windows" {
		// TODO(brainman): do not know why it hangs.
		t.Logf("skipping known-broken test on windows")
		return
	}
	// Test that Dial does not honor self-connects.
	// See the comment in DialTCP.

	// Find a port that would be used as a local address.
	l, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	c, err := Dial("tcp", l.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	addr := c.LocalAddr().String()
	c.Close()
	l.Close()

	// Try to connect to that address repeatedly.
	n := 100000
	if testing.Short() {
		n = 1000
	}
	switch runtime.GOOS {
	case "darwin", "freebsd", "openbsd", "windows":
		// Non-Linux systems take a long time to figure
		// out that there is nothing listening on localhost.
		n = 100
	}
	for i := 0; i < n; i++ {
		c, err := Dial("tcp", addr)
		if err == nil {
			c.Close()
			t.Errorf("#%d: Dial %q succeeded", i, addr)
		}
	}
}

var runErrorTest = flag.Bool("run_error_test", false, "let TestDialError check for dns errors")

type DialErrorTest struct {
	Net     string
	Raddr   string
	Pattern string
}

var dialErrorTests = []DialErrorTest{
	{
		"datakit", "mh/astro/r70",
		"dial datakit mh/astro/r70: unknown network datakit",
	},
	{
		"tcp", "127.0.0.1:☺",
		"dial tcp 127.0.0.1:☺: unknown port tcp/☺",
	},
	{
		"tcp", "no-such-name.google.com.:80",
		"dial tcp no-such-name.google.com.:80: lookup no-such-name.google.com.( on .*)?: no (.*)",
	},
	{
		"tcp", "no-such-name.no-such-top-level-domain.:80",
		"dial tcp no-such-name.no-such-top-level-domain.:80: lookup no-such-name.no-such-top-level-domain.( on .*)?: no (.*)",
	},
	{
		"tcp", "no-such-name:80",
		`dial tcp no-such-name:80: lookup no-such-name\.(.*\.)?( on .*)?: no (.*)`,
	},
	{
		"tcp", "mh/astro/r70:http",
		"dial tcp mh/astro/r70:http: lookup mh/astro/r70: invalid domain name",
	},
	{
		"unix", "/etc/file-not-found",
		"dial unix /etc/file-not-found: no such file or directory",
	},
	{
		"unix", "/etc/",
		"dial unix /etc/: (permission denied|socket operation on non-socket|connection refused)",
	},
	{
		"unixpacket", "/etc/file-not-found",
		"dial unixpacket /etc/file-not-found: no such file or directory",
	},
	{
		"unixpacket", "/etc/",
		"dial unixpacket /etc/: (permission denied|socket operation on non-socket|connection refused)",
	},
}

var duplicateErrorPattern = `dial (.*) dial (.*)`

func TestDialError(t *testing.T) {
	if !*runErrorTest {
		t.Logf("test disabled; use -run_error_test to enable")
		return
	}
	for i, tt := range dialErrorTests {
		c, err := Dial(tt.Net, tt.Raddr)
		if c != nil {
			c.Close()
		}
		if err == nil {
			t.Errorf("#%d: nil error, want match for %#q", i, tt.Pattern)
			continue
		}
		s := err.Error()
		match, _ := regexp.MatchString(tt.Pattern, s)
		if !match {
			t.Errorf("#%d: %q, want match for %#q", i, s, tt.Pattern)
		}
		match, _ = regexp.MatchString(duplicateErrorPattern, s)
		if match {
			t.Errorf("#%d: %q, duplicate error return from Dial", i, s)
		}
	}
}
