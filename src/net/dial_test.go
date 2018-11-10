// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bufio"
	"context"
	"internal/poll"
	"internal/testenv"
	"io"
	"runtime"
	"sync"
	"testing"
	"time"
)

var prohibitionaryDialArgTests = []struct {
	network string
	address string
}{
	{"tcp6", "127.0.0.1"},
	{"tcp6", "::ffff:127.0.0.1"},
}

func TestProhibitionaryDialArg(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv4map() {
		t.Skip("mapping ipv4 address inside ipv6 address not supported")
	}

	ln, err := Listen("tcp", "[::]:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	_, port, err := SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	for i, tt := range prohibitionaryDialArgTests {
		c, err := Dial(tt.network, JoinHostPort(tt.address, port))
		if err == nil {
			c.Close()
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestDialLocal(t *testing.T) {
	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	_, port, err := SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	c, err := Dial("tcp", JoinHostPort("", port))
	if err != nil {
		t.Fatal(err)
	}
	c.Close()
}

func TestDialerDualStackFDLeak(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("%s does not have full support of socktest", runtime.GOOS)
	case "windows":
		t.Skipf("not implemented a way to cancel dial racers in TCP SYN-SENT state on %s", runtime.GOOS)
	case "openbsd":
		testenv.SkipFlaky(t, 15157)
	}
	if !supportsIPv4() || !supportsIPv6() {
		t.Skip("both IPv4 and IPv6 are required")
	}

	closedPortDelay, expectClosedPortDelay := dialClosedPort()
	if closedPortDelay > expectClosedPortDelay {
		t.Errorf("got %v; want <= %v", closedPortDelay, expectClosedPortDelay)
	}

	before := sw.Sockets()
	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = lookupLocalhost
	handler := func(dss *dualStackServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}
	dss, err := newDualStackServer()
	if err != nil {
		t.Fatal(err)
	}
	if err := dss.buildup(handler); err != nil {
		dss.teardown()
		t.Fatal(err)
	}

	const N = 10
	var wg sync.WaitGroup
	wg.Add(N)
	d := &Dialer{DualStack: true, Timeout: 100*time.Millisecond + closedPortDelay}
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			c, err := d.Dial("tcp", JoinHostPort("localhost", dss.port))
			if err != nil {
				t.Error(err)
				return
			}
			c.Close()
		}()
	}
	wg.Wait()
	dss.teardown()
	after := sw.Sockets()
	if len(after) != len(before) {
		t.Errorf("got %d; want %d", len(after), len(before))
	}
}

// Define a pair of blackholed (IPv4, IPv6) addresses, for which dialTCP is
// expected to hang until the timeout elapses. These addresses are reserved
// for benchmarking by RFC 6890.
const (
	slowDst4 = "198.18.0.254"
	slowDst6 = "2001:2::254"
)

// In some environments, the slow IPs may be explicitly unreachable, and fail
// more quickly than expected. This test hook prevents dialTCP from returning
// before the deadline.
func slowDialTCP(ctx context.Context, net string, laddr, raddr *TCPAddr) (*TCPConn, error) {
	c, err := doDialTCP(ctx, net, laddr, raddr)
	if ParseIP(slowDst4).Equal(raddr.IP) || ParseIP(slowDst6).Equal(raddr.IP) {
		// Wait for the deadline, or indefinitely if none exists.
		<-ctx.Done()
	}
	return c, err
}

func dialClosedPort() (actual, expected time.Duration) {
	// Estimate the expected time for this platform.
	// On Windows, dialing a closed port takes roughly 1 second,
	// but other platforms should be instantaneous.
	if runtime.GOOS == "windows" {
		expected = 1500 * time.Millisecond
	} else if runtime.GOOS == "darwin" {
		expected = 150 * time.Millisecond
	} else {
		expected = 95 * time.Millisecond
	}

	l, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 999 * time.Hour, expected
	}
	addr := l.Addr().String()
	l.Close()
	// On OpenBSD, interference from TestSelfConnect is mysteriously
	// causing the first attempt to hang for a few seconds, so we throw
	// away the first result and keep the second.
	for i := 1; ; i++ {
		startTime := time.Now()
		c, err := Dial("tcp", addr)
		if err == nil {
			c.Close()
		}
		elapsed := time.Now().Sub(startTime)
		if i == 2 {
			return elapsed, expected
		}
	}
}

func TestDialParallel(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	if !supportsIPv4() || !supportsIPv6() {
		t.Skip("both IPv4 and IPv6 are required")
	}

	closedPortDelay, expectClosedPortDelay := dialClosedPort()
	if closedPortDelay > expectClosedPortDelay {
		t.Errorf("got %v; want <= %v", closedPortDelay, expectClosedPortDelay)
	}

	const instant time.Duration = 0
	const fallbackDelay = 200 * time.Millisecond

	// Some cases will run quickly when "connection refused" is fast,
	// or trigger the fallbackDelay on Windows. This value holds the
	// lesser of the two delays.
	var closedPortOrFallbackDelay time.Duration
	if closedPortDelay < fallbackDelay {
		closedPortOrFallbackDelay = closedPortDelay
	} else {
		closedPortOrFallbackDelay = fallbackDelay
	}

	origTestHookDialTCP := testHookDialTCP
	defer func() { testHookDialTCP = origTestHookDialTCP }()
	testHookDialTCP = slowDialTCP

	nCopies := func(s string, n int) []string {
		out := make([]string, n)
		for i := 0; i < n; i++ {
			out[i] = s
		}
		return out
	}

	var testCases = []struct {
		primaries       []string
		fallbacks       []string
		teardownNetwork string
		expectOk        bool
		expectElapsed   time.Duration
	}{
		// These should just work on the first try.
		{[]string{"127.0.0.1"}, []string{}, "", true, instant},
		{[]string{"::1"}, []string{}, "", true, instant},
		{[]string{"127.0.0.1", "::1"}, []string{slowDst6}, "tcp6", true, instant},
		{[]string{"::1", "127.0.0.1"}, []string{slowDst4}, "tcp4", true, instant},
		// Primary is slow; fallback should kick in.
		{[]string{slowDst4}, []string{"::1"}, "", true, fallbackDelay},
		// Skip a "connection refused" in the primary thread.
		{[]string{"127.0.0.1", "::1"}, []string{}, "tcp4", true, closedPortDelay},
		{[]string{"::1", "127.0.0.1"}, []string{}, "tcp6", true, closedPortDelay},
		// Skip a "connection refused" in the fallback thread.
		{[]string{slowDst4, slowDst6}, []string{"::1", "127.0.0.1"}, "tcp6", true, fallbackDelay + closedPortDelay},
		// Primary refused, fallback without delay.
		{[]string{"127.0.0.1"}, []string{"::1"}, "tcp4", true, closedPortOrFallbackDelay},
		{[]string{"::1"}, []string{"127.0.0.1"}, "tcp6", true, closedPortOrFallbackDelay},
		// Everything is refused.
		{[]string{"127.0.0.1"}, []string{}, "tcp4", false, closedPortDelay},
		// Nothing to do; fail instantly.
		{[]string{}, []string{}, "", false, instant},
		// Connecting to tons of addresses should not trip the deadline.
		{nCopies("::1", 1000), []string{}, "", true, instant},
	}

	handler := func(dss *dualStackServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}

	// Convert a list of IP strings into TCPAddrs.
	makeAddrs := func(ips []string, port string) addrList {
		var out addrList
		for _, ip := range ips {
			addr, err := ResolveTCPAddr("tcp", JoinHostPort(ip, port))
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, addr)
		}
		return out
	}

	for i, tt := range testCases {
		dss, err := newDualStackServer()
		if err != nil {
			t.Fatal(err)
		}
		defer dss.teardown()
		if err := dss.buildup(handler); err != nil {
			t.Fatal(err)
		}
		if tt.teardownNetwork != "" {
			// Destroy one of the listening sockets, creating an unreachable port.
			dss.teardownNetwork(tt.teardownNetwork)
		}

		primaries := makeAddrs(tt.primaries, dss.port)
		fallbacks := makeAddrs(tt.fallbacks, dss.port)
		d := Dialer{
			FallbackDelay: fallbackDelay,
		}
		startTime := time.Now()
		dp := &dialParam{
			Dialer:  d,
			network: "tcp",
			address: "?",
		}
		c, err := dialParallel(context.Background(), dp, primaries, fallbacks)
		elapsed := time.Since(startTime)

		if c != nil {
			c.Close()
		}

		if tt.expectOk && err != nil {
			t.Errorf("#%d: got %v; want nil", i, err)
		} else if !tt.expectOk && err == nil {
			t.Errorf("#%d: got nil; want non-nil", i)
		}

		expectElapsedMin := tt.expectElapsed - 95*time.Millisecond
		expectElapsedMax := tt.expectElapsed + 95*time.Millisecond
		if !(elapsed >= expectElapsedMin) {
			t.Errorf("#%d: got %v; want >= %v", i, elapsed, expectElapsedMin)
		} else if !(elapsed <= expectElapsedMax) {
			t.Errorf("#%d: got %v; want <= %v", i, elapsed, expectElapsedMax)
		}

		// Repeat each case, ensuring that it can be canceled quickly.
		ctx, cancel := context.WithCancel(context.Background())
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			time.Sleep(5 * time.Millisecond)
			cancel()
			wg.Done()
		}()
		startTime = time.Now()
		c, err = dialParallel(ctx, dp, primaries, fallbacks)
		if c != nil {
			c.Close()
		}
		elapsed = time.Now().Sub(startTime)
		if elapsed > 100*time.Millisecond {
			t.Errorf("#%d (cancel): got %v; want <= 100ms", i, elapsed)
		}
		wg.Wait()
	}
}

func lookupSlowFast(ctx context.Context, fn func(context.Context, string) ([]IPAddr, error), host string) ([]IPAddr, error) {
	switch host {
	case "slow6loopback4":
		// Returns a slow IPv6 address, and a local IPv4 address.
		return []IPAddr{
			{IP: ParseIP(slowDst6)},
			{IP: ParseIP("127.0.0.1")},
		}, nil
	default:
		return fn(ctx, host)
	}
}

func TestDialerFallbackDelay(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	if !supportsIPv4() || !supportsIPv6() {
		t.Skip("both IPv4 and IPv6 are required")
	}

	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = lookupSlowFast

	origTestHookDialTCP := testHookDialTCP
	defer func() { testHookDialTCP = origTestHookDialTCP }()
	testHookDialTCP = slowDialTCP

	var testCases = []struct {
		dualstack     bool
		delay         time.Duration
		expectElapsed time.Duration
	}{
		// Use a very brief delay, which should fallback immediately.
		{true, 1 * time.Nanosecond, 0},
		// Use a 200ms explicit timeout.
		{true, 200 * time.Millisecond, 200 * time.Millisecond},
		// The default is 300ms.
		{true, 0, 300 * time.Millisecond},
	}

	handler := func(dss *dualStackServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}
	dss, err := newDualStackServer()
	if err != nil {
		t.Fatal(err)
	}
	defer dss.teardown()
	if err := dss.buildup(handler); err != nil {
		t.Fatal(err)
	}

	for i, tt := range testCases {
		d := &Dialer{DualStack: tt.dualstack, FallbackDelay: tt.delay}

		startTime := time.Now()
		c, err := d.Dial("tcp", JoinHostPort("slow6loopback4", dss.port))
		elapsed := time.Now().Sub(startTime)
		if err == nil {
			c.Close()
		} else if tt.dualstack {
			t.Error(err)
		}
		expectMin := tt.expectElapsed - 1*time.Millisecond
		expectMax := tt.expectElapsed + 95*time.Millisecond
		if !(elapsed >= expectMin) {
			t.Errorf("#%d: got %v; want >= %v", i, elapsed, expectMin)
		}
		if !(elapsed <= expectMax) {
			t.Errorf("#%d: got %v; want <= %v", i, elapsed, expectMax)
		}
	}
}

func TestDialParallelSpuriousConnection(t *testing.T) {
	if !supportsIPv4() || !supportsIPv6() {
		t.Skip("both IPv4 and IPv6 are required")
	}

	var wg sync.WaitGroup
	wg.Add(2)
	handler := func(dss *dualStackServer, ln Listener) {
		// Accept one connection per address.
		c, err := ln.Accept()
		if err != nil {
			t.Fatal(err)
		}
		// The client should close itself, without sending data.
		c.SetReadDeadline(time.Now().Add(1 * time.Second))
		var b [1]byte
		if _, err := c.Read(b[:]); err != io.EOF {
			t.Errorf("got %v; want %v", err, io.EOF)
		}
		c.Close()
		wg.Done()
	}
	dss, err := newDualStackServer()
	if err != nil {
		t.Fatal(err)
	}
	defer dss.teardown()
	if err := dss.buildup(handler); err != nil {
		t.Fatal(err)
	}

	const fallbackDelay = 100 * time.Millisecond

	origTestHookDialTCP := testHookDialTCP
	defer func() { testHookDialTCP = origTestHookDialTCP }()
	testHookDialTCP = func(ctx context.Context, net string, laddr, raddr *TCPAddr) (*TCPConn, error) {
		// Sleep long enough for Happy Eyeballs to kick in, and inhibit cancelation.
		// This forces dialParallel to juggle two successful connections.
		time.Sleep(fallbackDelay * 2)

		// Now ignore the provided context (which will be canceled) and use a
		// different one to make sure this completes with a valid connection,
		// which we hope to be closed below:
		return doDialTCP(context.Background(), net, laddr, raddr)
	}

	d := Dialer{
		FallbackDelay: fallbackDelay,
	}
	dp := &dialParam{
		Dialer:  d,
		network: "tcp",
		address: "?",
	}

	makeAddr := func(ip string) addrList {
		addr, err := ResolveTCPAddr("tcp", JoinHostPort(ip, dss.port))
		if err != nil {
			t.Fatal(err)
		}
		return addrList{addr}
	}

	// dialParallel returns one connection (and closes the other.)
	c, err := dialParallel(context.Background(), dp, makeAddr("127.0.0.1"), makeAddr("::1"))
	if err != nil {
		t.Fatal(err)
	}
	c.Close()

	// The server should've seen both connections.
	wg.Wait()
}

func TestDialerPartialDeadline(t *testing.T) {
	now := time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)
	var testCases = []struct {
		now            time.Time
		deadline       time.Time
		addrs          int
		expectDeadline time.Time
		expectErr      error
	}{
		// Regular division.
		{now, now.Add(12 * time.Second), 1, now.Add(12 * time.Second), nil},
		{now, now.Add(12 * time.Second), 2, now.Add(6 * time.Second), nil},
		{now, now.Add(12 * time.Second), 3, now.Add(4 * time.Second), nil},
		// Bump against the 2-second sane minimum.
		{now, now.Add(12 * time.Second), 999, now.Add(2 * time.Second), nil},
		// Total available is now below the sane minimum.
		{now, now.Add(1900 * time.Millisecond), 999, now.Add(1900 * time.Millisecond), nil},
		// Null deadline.
		{now, noDeadline, 1, noDeadline, nil},
		// Step the clock forward and cross the deadline.
		{now.Add(-1 * time.Millisecond), now, 1, now, nil},
		{now.Add(0 * time.Millisecond), now, 1, noDeadline, poll.ErrTimeout},
		{now.Add(1 * time.Millisecond), now, 1, noDeadline, poll.ErrTimeout},
	}
	for i, tt := range testCases {
		deadline, err := partialDeadline(tt.now, tt.deadline, tt.addrs)
		if err != tt.expectErr {
			t.Errorf("#%d: got %v; want %v", i, err, tt.expectErr)
		}
		if !deadline.Equal(tt.expectDeadline) {
			t.Errorf("#%d: got %v; want %v", i, deadline, tt.expectDeadline)
		}
	}
}

func TestDialerLocalAddr(t *testing.T) {
	if !supportsIPv4() || !supportsIPv6() {
		t.Skip("both IPv4 and IPv6 are required")
	}

	type test struct {
		network, raddr string
		laddr          Addr
		error
	}
	var tests = []test{
		{"tcp4", "127.0.0.1", nil, nil},
		{"tcp4", "127.0.0.1", &TCPAddr{}, nil},
		{"tcp4", "127.0.0.1", &TCPAddr{IP: ParseIP("0.0.0.0")}, nil},
		{"tcp4", "127.0.0.1", &TCPAddr{IP: ParseIP("0.0.0.0").To4()}, nil},
		{"tcp4", "127.0.0.1", &TCPAddr{IP: ParseIP("::")}, &AddrError{Err: "some error"}},
		{"tcp4", "127.0.0.1", &TCPAddr{IP: ParseIP("127.0.0.1").To4()}, nil},
		{"tcp4", "127.0.0.1", &TCPAddr{IP: ParseIP("127.0.0.1").To16()}, nil},
		{"tcp4", "127.0.0.1", &TCPAddr{IP: IPv6loopback}, errNoSuitableAddress},
		{"tcp4", "127.0.0.1", &UDPAddr{}, &AddrError{Err: "some error"}},
		{"tcp4", "127.0.0.1", &UnixAddr{}, &AddrError{Err: "some error"}},

		{"tcp6", "::1", nil, nil},
		{"tcp6", "::1", &TCPAddr{}, nil},
		{"tcp6", "::1", &TCPAddr{IP: ParseIP("0.0.0.0")}, nil},
		{"tcp6", "::1", &TCPAddr{IP: ParseIP("0.0.0.0").To4()}, nil},
		{"tcp6", "::1", &TCPAddr{IP: ParseIP("::")}, nil},
		{"tcp6", "::1", &TCPAddr{IP: ParseIP("127.0.0.1").To4()}, errNoSuitableAddress},
		{"tcp6", "::1", &TCPAddr{IP: ParseIP("127.0.0.1").To16()}, errNoSuitableAddress},
		{"tcp6", "::1", &TCPAddr{IP: IPv6loopback}, nil},
		{"tcp6", "::1", &UDPAddr{}, &AddrError{Err: "some error"}},
		{"tcp6", "::1", &UnixAddr{}, &AddrError{Err: "some error"}},

		{"tcp", "127.0.0.1", nil, nil},
		{"tcp", "127.0.0.1", &TCPAddr{}, nil},
		{"tcp", "127.0.0.1", &TCPAddr{IP: ParseIP("0.0.0.0")}, nil},
		{"tcp", "127.0.0.1", &TCPAddr{IP: ParseIP("0.0.0.0").To4()}, nil},
		{"tcp", "127.0.0.1", &TCPAddr{IP: ParseIP("127.0.0.1").To4()}, nil},
		{"tcp", "127.0.0.1", &TCPAddr{IP: ParseIP("127.0.0.1").To16()}, nil},
		{"tcp", "127.0.0.1", &TCPAddr{IP: IPv6loopback}, errNoSuitableAddress},
		{"tcp", "127.0.0.1", &UDPAddr{}, &AddrError{Err: "some error"}},
		{"tcp", "127.0.0.1", &UnixAddr{}, &AddrError{Err: "some error"}},

		{"tcp", "::1", nil, nil},
		{"tcp", "::1", &TCPAddr{}, nil},
		{"tcp", "::1", &TCPAddr{IP: ParseIP("0.0.0.0")}, nil},
		{"tcp", "::1", &TCPAddr{IP: ParseIP("0.0.0.0").To4()}, nil},
		{"tcp", "::1", &TCPAddr{IP: ParseIP("::")}, nil},
		{"tcp", "::1", &TCPAddr{IP: ParseIP("127.0.0.1").To4()}, errNoSuitableAddress},
		{"tcp", "::1", &TCPAddr{IP: ParseIP("127.0.0.1").To16()}, errNoSuitableAddress},
		{"tcp", "::1", &TCPAddr{IP: IPv6loopback}, nil},
		{"tcp", "::1", &UDPAddr{}, &AddrError{Err: "some error"}},
		{"tcp", "::1", &UnixAddr{}, &AddrError{Err: "some error"}},
	}

	if supportsIPv4map() {
		tests = append(tests, test{
			"tcp", "127.0.0.1", &TCPAddr{IP: ParseIP("::")}, nil,
		})
	} else {
		tests = append(tests, test{
			"tcp", "127.0.0.1", &TCPAddr{IP: ParseIP("::")}, &AddrError{Err: "some error"},
		})
	}

	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = lookupLocalhost
	handler := func(ls *localServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}
	var err error
	var lss [2]*localServer
	for i, network := range []string{"tcp4", "tcp6"} {
		lss[i], err = newLocalServer(network)
		if err != nil {
			t.Fatal(err)
		}
		defer lss[i].teardown()
		if err := lss[i].buildup(handler); err != nil {
			t.Fatal(err)
		}
	}

	for _, tt := range tests {
		d := &Dialer{LocalAddr: tt.laddr}
		var addr string
		ip := ParseIP(tt.raddr)
		if ip.To4() != nil {
			addr = lss[0].Listener.Addr().String()
		}
		if ip.To16() != nil && ip.To4() == nil {
			addr = lss[1].Listener.Addr().String()
		}
		c, err := d.Dial(tt.network, addr)
		if err == nil && tt.error != nil || err != nil && tt.error == nil {
			t.Errorf("%s %v->%s: got %v; want %v", tt.network, tt.laddr, tt.raddr, err, tt.error)
		}
		if err != nil {
			if perr := parseDialError(err); perr != nil {
				t.Error(perr)
			}
			continue
		}
		c.Close()
	}
}

func TestDialerDualStack(t *testing.T) {
	testenv.SkipFlaky(t, 13324)

	if !supportsIPv4() || !supportsIPv6() {
		t.Skip("both IPv4 and IPv6 are required")
	}

	closedPortDelay, expectClosedPortDelay := dialClosedPort()
	if closedPortDelay > expectClosedPortDelay {
		t.Errorf("got %v; want <= %v", closedPortDelay, expectClosedPortDelay)
	}

	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = lookupLocalhost
	handler := func(dss *dualStackServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}

	var timeout = 150*time.Millisecond + closedPortDelay
	for _, dualstack := range []bool{false, true} {
		dss, err := newDualStackServer()
		if err != nil {
			t.Fatal(err)
		}
		defer dss.teardown()
		if err := dss.buildup(handler); err != nil {
			t.Fatal(err)
		}

		d := &Dialer{DualStack: dualstack, Timeout: timeout}
		for range dss.lns {
			c, err := d.Dial("tcp", JoinHostPort("localhost", dss.port))
			if err != nil {
				t.Error(err)
				continue
			}
			switch addr := c.LocalAddr().(*TCPAddr); {
			case addr.IP.To4() != nil:
				dss.teardownNetwork("tcp4")
			case addr.IP.To16() != nil && addr.IP.To4() == nil:
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
	defer func() { testHookSetKeepAlive = func() {} }()

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

func TestDialCancel(t *testing.T) {
	switch testenv.Builder() {
	case "linux-arm64-buildlet":
		t.Skip("skipping on linux-arm64-buildlet; incompatible network config? issue 15191")
	case "":
		testenv.MustHaveExternalNetwork(t)
	}

	if runtime.GOOS == "nacl" {
		// nacl doesn't have external network access.
		t.Skipf("skipping on %s", runtime.GOOS)
	}

	blackholeIPPort := JoinHostPort(slowDst4, "1234")
	if !supportsIPv4() {
		blackholeIPPort = JoinHostPort(slowDst6, "1234")
	}

	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()

	const cancelTick = 5 // the timer tick we cancel the dial at
	const timeoutTick = 100

	var d Dialer
	cancel := make(chan struct{})
	d.Cancel = cancel
	errc := make(chan error, 1)
	connc := make(chan Conn, 1)
	go func() {
		if c, err := d.Dial("tcp", blackholeIPPort); err != nil {
			errc <- err
		} else {
			connc <- c
		}
	}()
	ticks := 0
	for {
		select {
		case <-ticker.C:
			ticks++
			if ticks == cancelTick {
				close(cancel)
			}
			if ticks == timeoutTick {
				t.Fatal("timeout waiting for dial to fail")
			}
		case c := <-connc:
			c.Close()
			t.Fatal("unexpected successful connection")
		case err := <-errc:
			if perr := parseDialError(err); perr != nil {
				t.Error(perr)
			}
			if ticks < cancelTick {
				t.Fatalf("dial error after %d ticks (%d before cancel sent): %v",
					ticks, cancelTick-ticks, err)
			}
			if oe, ok := err.(*OpError); !ok || oe.Err != errCanceled {
				t.Fatalf("dial error = %v (%T); want OpError with Err == errCanceled", err, err)
			}
			return // success.
		}
	}
}

func TestCancelAfterDial(t *testing.T) {
	if testing.Short() {
		t.Skip("avoiding time.Sleep")
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	defer func() {
		ln.Close()
		wg.Wait()
	}()

	// Echo back the first line of each incoming connection.
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				break
			}
			rb := bufio.NewReader(c)
			line, err := rb.ReadString('\n')
			if err != nil {
				t.Error(err)
				c.Close()
				continue
			}
			if _, err := c.Write([]byte(line)); err != nil {
				t.Error(err)
			}
			c.Close()
		}
		wg.Done()
	}()

	try := func() {
		cancel := make(chan struct{})
		d := &Dialer{Cancel: cancel}
		c, err := d.Dial("tcp", ln.Addr().String())

		// Immediately after dialing, request cancelation and sleep.
		// Before Issue 15078 was fixed, this would cause subsequent operations
		// to fail with an i/o timeout roughly 50% of the time.
		close(cancel)
		time.Sleep(10 * time.Millisecond)

		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		// Send some data to confirm that the connection is still alive.
		const message = "echo!\n"
		if _, err := c.Write([]byte(message)); err != nil {
			t.Fatal(err)
		}

		// The server should echo the line, and close the connection.
		rb := bufio.NewReader(c)
		line, err := rb.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		if line != message {
			t.Errorf("got %q; want %q", line, message)
		}
		if _, err := rb.ReadByte(); err != io.EOF {
			t.Errorf("got %v; want %v", err, io.EOF)
		}
	}

	// This bug manifested about 50% of the time, so try it a few times.
	for i := 0; i < 10; i++ {
		try()
	}
}

// Issue 18806: it should always be possible to net.Dial a
// net.Listener().Addr().String when the listen address was ":n", even
// if the machine has halfway configured IPv6 such that it can bind on
// "::" not connect back to that same address.
func TestDialListenerAddr(t *testing.T) {
	if testenv.Builder() == "" {
		testenv.MustHaveExternalNetwork(t)
	}
	ln, err := Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	addr := ln.Addr().String()
	c, err := Dial("tcp", addr)
	if err != nil {
		t.Fatalf("for addr %q, dial error: %v", addr, err)
	}
	c.Close()
}
