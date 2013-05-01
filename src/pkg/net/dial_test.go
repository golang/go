// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag"
	"fmt"
	"io"
	"os"
	"reflect"
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
	origBacklog := listenerBacklog
	defer func() {
		listenerBacklog = origBacklog
	}()
	listenerBacklog = 1

	ln := newLocalListener(t)
	defer ln.Close()

	errc := make(chan error)

	numConns := listenerBacklog + 100

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
		t.Skipf("skipping test on %q; untested.", runtime.GOOS)
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
		t.Skip("skipping known-broken test on windows")
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

var invalidDialAndListenArgTests = []struct {
	net  string
	addr string
	err  error
}{
	{"foo", "bar", &OpError{Op: "dial", Net: "foo", Addr: nil, Err: UnknownNetworkError("foo")}},
	{"baz", "", &OpError{Op: "listen", Net: "baz", Addr: nil, Err: UnknownNetworkError("baz")}},
	{"tcp", "", &OpError{Op: "dial", Net: "tcp", Addr: nil, Err: errMissingAddress}},
}

func TestInvalidDialAndListenArgs(t *testing.T) {
	for _, tt := range invalidDialAndListenArgTests {
		var err error
		switch tt.err.(*OpError).Op {
		case "dial":
			_, err = Dial(tt.net, tt.addr)
		case "listen":
			_, err = Listen(tt.net, tt.addr)
		}
		if !reflect.DeepEqual(tt.err, err) {
			t.Fatalf("got %#v; expected %#v", err, tt.err)
		}
	}
}

func TestDialTimeoutFDLeak(t *testing.T) {
	if runtime.GOOS != "linux" {
		// TODO(bradfitz): test on other platforms
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	ln := newLocalListener(t)
	defer ln.Close()

	type connErr struct {
		conn Conn
		err  error
	}
	dials := listenerBacklog + 100
	// used to be listenerBacklog + 5, but was found to be unreliable, issue 4384.
	maxGoodConnect := listenerBacklog + runtime.NumCPU()*10
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

var testPoller = flag.Bool("poller", false, "platform supports runtime-integrated poller")

// Assert that a failed Dial attempt does not leak
// runtime.PollDesc structures
func TestDialFailPDLeak(t *testing.T) {
	if !*testPoller {
		t.Skip("test disabled; use -poller to enable")
	}

	const loops = 10
	const count = 20000
	var old runtime.MemStats // used by sysdelta
	runtime.ReadMemStats(&old)
	sysdelta := func() uint64 {
		var new runtime.MemStats
		runtime.ReadMemStats(&new)
		delta := old.Sys - new.Sys
		old = new
		return delta
	}
	d := &Dialer{Timeout: time.Nanosecond} // don't bother TCP with handshaking
	failcount := 0
	for i := 0; i < loops; i++ {
		for i := 0; i < count; i++ {
			conn, err := d.Dial("tcp", "127.0.0.1:1")
			if err == nil {
				t.Error("dial should not succeed")
				conn.Close()
				t.FailNow()
			}
		}
		if delta := sysdelta(); delta > 0 {
			failcount++
		}
		// there are always some allocations on the first loop
		if failcount > 3 {
			t.Error("detected possible memory leak in runtime")
			t.FailNow()
		}
	}
}

func TestDialer(t *testing.T) {
	ln, err := Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer ln.Close()
	ch := make(chan error, 1)
	go func() {
		var err error
		c, err := ln.Accept()
		if err != nil {
			ch <- fmt.Errorf("Accept failed: %v", err)
			return
		}
		defer c.Close()
		ch <- nil
	}()

	laddr, err := ResolveTCPAddr("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("ResolveTCPAddr failed: %v", err)
	}
	d := &Dialer{LocalAddr: laddr}
	c, err := d.Dial("tcp4", ln.Addr().String())
	if err != nil {
		t.Fatalf("Dial failed: %v", err)
	}
	defer c.Close()
	c.Read(make([]byte, 1))
	err = <-ch
	if err != nil {
		t.Error(err)
	}
}
