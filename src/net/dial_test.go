// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"fmt"
	"net/internal/socktest"
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
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("%s does not have full support of socktest", runtime.GOOS)
	}

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

	before := sw.Sockets()
	const N = 100
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			// This dial never starts to send any SYN
			// segment because of above socket filter and
			// test hook.
			c, err := DialTimeout("tcp", "127.0.0.1:0", T)
			if err == nil {
				t.Errorf("unexpectedly established: tcp:%s->%s", c.LocalAddr(), c.RemoteAddr())
				c.Close()
			}
		}()
	}
	wg.Wait()
	after := sw.Sockets()
	if len(after) != len(before) {
		t.Errorf("got %d; want %d", len(after), len(before))
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

	handler := func(dss *dualStackServer, ln Listener) {
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
		{network: "tcp4", address: "127.0.0.1"},
		{network: "tcp6", address: "::1"},
	})
	if err != nil {
		t.Fatalf("newDualStackServer failed: %v", err)
	}
	defer dss.teardown()
	if err := dss.buildup(handler); err != nil {
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

func TestDialerLocalAddr(t *testing.T) {
	ch := make(chan error, 1)
	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			ch <- fmt.Errorf("Accept failed: %v", err)
			return
		}
		defer c.Close()
		ch <- nil
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	laddr, err := ResolveTCPAddr(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
	if err != nil {
		t.Fatalf("ResolveTCPAddr failed: %v", err)
	}
	laddr.Port = 0
	d := &Dialer{LocalAddr: laddr}
	c, err := d.Dial(ls.Listener.Addr().Network(), ls.Addr().String())
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

func TestDialerDualStack(t *testing.T) {
	switch runtime.GOOS {
	case "nacl":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	if ips, err := LookupIP("localhost"); err != nil {
		t.Fatalf("LookupIP failed: %v", err)
	} else if len(ips) < 2 || !supportsIPv4 || !supportsIPv6 {
		t.Skip("localhost doesn't have a pair of different address family IP addresses")
	}

	handler := func(dss *dualStackServer, ln Listener) {
		for {
			if c, err := ln.Accept(); err != nil {
				return
			} else {
				c.Close()
			}
		}
	}
	dss, err := newDualStackServer([]streamListener{
		{network: "tcp4", address: "127.0.0.1"},
		{network: "tcp6", address: "::1"},
	})
	if err != nil {
		t.Fatalf("newDualStackServer failed: %v", err)
	}
	defer dss.teardown()
	if err := dss.buildup(handler); err != nil {
		t.Fatalf("dualStackServer.buildup failed: %v", err)
	}

	d := &Dialer{DualStack: true}
	for range dss.lns {
		if c, err := d.Dial("tcp", JoinHostPort("localhost", dss.port)); err != nil {
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
	handler := func(ls *localServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
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
	defer func() {
		testHookSetKeepAlive = func() {}
	}()

	for _, keepAlive := range []bool{false, true} {
		got := false
		testHookSetKeepAlive = func() { got = true }
		var d Dialer
		if keepAlive {
			d.KeepAlive = 30 * time.Second
		}
		c, err := d.Dial("tcp", ls.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		c.Close()
		if got != keepAlive {
			t.Errorf("Dialer.KeepAlive = %v: SetKeepAlive called = %v, want %v", d.KeepAlive, got, !got)
		}
	}
}
