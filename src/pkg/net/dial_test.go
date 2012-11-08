// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag"
	"fmt"
	"io"
	"os"
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
	// way. This is unfortunate, but works for now.
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
		//
		// Use an IANA reserved port (49151) instead of 80, because
		// on our 386 builder, this Dial succeeds, connecting
		// to an IIS web server somewhere.  The data center
		// or VM or firewall must be stealing the TCP connection.
		//
		// IANA Service Name and Transport Protocol Port Number Registry
		// <http://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xml>
		go func() {
			c, err := DialTimeout("tcp", "127.0.71.111:49151", 200*time.Millisecond)
			if err == nil {
				err = fmt.Errorf("unexpected: connected to %s!", c.RemoteAddr())
				c.Close()
			}
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
	case "darwin", "freebsd", "netbsd", "openbsd", "plan9", "windows":
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

func TestDialTimeoutFDLeak(t *testing.T) {
	if runtime.GOOS != "linux" {
		// TODO(bradfitz): test on other platforms
		t.Logf("skipping test on %s", runtime.GOOS)
		return
	}

	ln := newLocalListener(t)
	defer ln.Close()

	type connErr struct {
		conn Conn
		err  error
	}
	dials := listenerBacklog + 100
	maxGoodConnect := listenerBacklog + 5 // empirically 131 good ones (of 128). who knows?
	resc := make(chan connErr)
	for i := 0; i < dials; i++ {
		go func() {
			conn, err := DialTimeout("tcp", ln.Addr().String(), 500*time.Millisecond)
			resc <- connErr{conn, err}
		}()
	}

	var firstErr string
	var ngood int
	var toClose []io.Closer
	for i := 0; i < dials; i++ {
		ce := <-resc
		if ce.err == nil {
			ngood++
			if ngood > maxGoodConnect {
				t.Errorf("%d good connects; expected at most %d", ngood, maxGoodConnect)
			}
			toClose = append(toClose, ce.conn)
			continue
		}
		err := ce.err
		if firstErr == "" {
			firstErr = err.Error()
		} else if err.Error() != firstErr {
			t.Fatalf("inconsistent error messages: first was %q, then later %q", firstErr, err)
		}
	}
	for _, c := range toClose {
		c.Close()
	}
	for i := 0; i < 100; i++ {
		if got := numFD(); got < dials {
			// Test passes.
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got := numFD(); got >= dials {
		t.Errorf("num fds after %d timeouts = %d; want <%d", dials, got, dials)
	}
}

func numFD() int {
	if runtime.GOOS == "linux" {
		f, err := os.Open("/proc/self/fd")
		if err != nil {
			panic(err)
		}
		defer f.Close()
		names, err := f.Readdirnames(0)
		if err != nil {
			panic(err)
		}
		return len(names)
	}
	// All tests using this should be skipped anyway, but:
	panic("numFDs not implemented on " + runtime.GOOS)
}
