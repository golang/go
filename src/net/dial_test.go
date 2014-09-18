// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"sync"
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
	case "darwin", "plan9", "windows":
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
	case "darwin", "dragonfly", "freebsd", "netbsd", "openbsd", "plan9", "solaris", "windows":
		// Non-Linux systems take a long time to figure
		// out that there is nothing listening on localhost.
		n = 100
	}
	for i := 0; i < n; i++ {
		c, err := DialTimeout("tcp", addr, time.Millisecond)
		if err == nil {
			if c.LocalAddr().String() == addr {
				t.Errorf("#%d: Dial %q self-connect", i, addr)
			} else {
				t.Logf("#%d: Dial %q succeeded - possibly racing with other listener", i, addr)
			}
			c.Close()
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

func numTCP() (ntcp, nopen, nclose int, err error) {
	lsof, err := exec.Command("lsof", "-n", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		return 0, 0, 0, err
	}
	ntcp += bytes.Count(lsof, []byte("TCP"))
	for _, state := range []string{"LISTEN", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED"} {
		nopen += bytes.Count(lsof, []byte(state))
	}
	for _, state := range []string{"CLOSED", "CLOSE_WAIT", "LAST_ACK", "FIN_WAIT_1", "FIN_WAIT_2", "CLOSING", "TIME_WAIT"} {
		nclose += bytes.Count(lsof, []byte(state))
	}
	return ntcp, nopen, nclose, nil
}

func TestDialMultiFDLeak(t *testing.T) {
	t.Skip("flaky test - golang.org/issue/8764")

	if !supportsIPv4 || !supportsIPv6 {
		t.Skip("neither ipv4 nor ipv6 is supported")
	}

	halfDeadServer := func(dss *dualStackServer, ln Listener) {
		for {
			if c, err := ln.Accept(); err != nil {
				return
			} else {
				// It just keeps established
				// connections like a half-dead server
				// does.
				dss.putConn(c)
			}
		}
	}
	dss, err := newDualStackServer([]streamListener{
		{net: "tcp4", addr: "127.0.0.1"},
		{net: "tcp6", addr: "[::1]"},
	})
	if err != nil {
		t.Fatalf("newDualStackServer failed: %v", err)
	}
	defer dss.teardown()
	if err := dss.buildup(halfDeadServer); err != nil {
		t.Fatalf("dualStackServer.buildup failed: %v", err)
	}

	_, before, _, err := numTCP()
	if err != nil {
		t.Skipf("skipping test; error finding or running lsof: %v", err)
	}

	var wg sync.WaitGroup
	portnum, _, _ := dtoi(dss.port, 0)
	ras := addrList{
		// Losers that will fail to connect, see RFC 6890.
		&TCPAddr{IP: IPv4(198, 18, 0, 254), Port: portnum},
		&TCPAddr{IP: ParseIP("2001:2::254"), Port: portnum},

		// Winner candidates of this race.
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: portnum},
		&TCPAddr{IP: IPv6loopback, Port: portnum},

		// Losers that will have established connections.
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: portnum},
		&TCPAddr{IP: IPv6loopback, Port: portnum},
	}
	const T1 = 10 * time.Millisecond
	const T2 = 2 * T1
	const N = 10
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if c, err := dialMulti("tcp", "fast failover test", nil, ras, time.Now().Add(T1)); err == nil {
				c.Close()
			}
		}()
	}
	wg.Wait()
	time.Sleep(T2)

	ntcp, after, nclose, err := numTCP()
	if err != nil {
		t.Skipf("skipping test; error finding or running lsof: %v", err)
	}
	t.Logf("tcp sessions: %v, open sessions: %v, closing sessions: %v", ntcp, after, nclose)

	if after != before {
		t.Fatalf("got %v open sessions; expected %v", after, before)
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

func TestDialer(t *testing.T) {
	ln, err := Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer ln.Close()
	ch := make(chan error, 1)
	go func() {
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

func TestDialDualStackLocalhost(t *testing.T) {
	switch runtime.GOOS {
	case "nacl":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	if ips, err := LookupIP("localhost"); err != nil {
		t.Fatalf("LookupIP failed: %v", err)
	} else if len(ips) < 2 || !supportsIPv4 || !supportsIPv6 {
		t.Skip("localhost doesn't have a pair of different address family IP addresses")
	}

	touchAndByeServer := func(dss *dualStackServer, ln Listener) {
		for {
			if c, err := ln.Accept(); err != nil {
				return
			} else {
				c.Close()
			}
		}
	}
	dss, err := newDualStackServer([]streamListener{
		{net: "tcp4", addr: "127.0.0.1"},
		{net: "tcp6", addr: "[::1]"},
	})
	if err != nil {
		t.Fatalf("newDualStackServer failed: %v", err)
	}
	defer dss.teardown()
	if err := dss.buildup(touchAndByeServer); err != nil {
		t.Fatalf("dualStackServer.buildup failed: %v", err)
	}

	d := &Dialer{DualStack: true}
	for range dss.lns {
		if c, err := d.Dial("tcp", "localhost:"+dss.port); err != nil {
			t.Errorf("Dial failed: %v", err)
		} else {
			if addr := c.LocalAddr().(*TCPAddr); addr.IP.To4() != nil {
				dss.teardownNetwork("tcp4")
			} else if addr.IP.To16() != nil && addr.IP.To4() == nil {
				dss.teardownNetwork("tcp6")
			}
			c.Close()
		}
	}
}

func TestDialerKeepAlive(t *testing.T) {
	ln := newLocalListener(t)
	defer ln.Close()
	defer func() {
		testHookSetKeepAlive = func() {}
	}()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}()
	for _, keepAlive := range []bool{false, true} {
		got := false
		testHookSetKeepAlive = func() { got = true }
		var d Dialer
		if keepAlive {
			d.KeepAlive = 30 * time.Second
		}
		c, err := d.Dial("tcp", ln.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		c.Close()
		if got != keepAlive {
			t.Errorf("Dialer.KeepAlive = %v: SetKeepAlive called = %v, want %v", d.KeepAlive, got, !got)
		}
	}
}
