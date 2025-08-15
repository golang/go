// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"net/netip"
	"os"
	"runtime"
	"strings"
	"sync"
	"syscall"
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
	ln := newLocalListener(t, "tcp")
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
	d := &Dialer{DualStack: true, Timeout: 5 * time.Second}
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
func slowDialTCP(ctx context.Context, network string, laddr, raddr *TCPAddr) (*TCPConn, error) {
	sd := &sysDialer{network: network, address: raddr.String()}
	c, err := sd.doDialTCP(ctx, laddr, raddr)
	if ParseIP(slowDst4).Equal(raddr.IP) || ParseIP(slowDst6).Equal(raddr.IP) {
		// Wait for the deadline, or indefinitely if none exists.
		<-ctx.Done()
	}
	return c, err
}

func dialClosedPort(t *testing.T) (dialLatency time.Duration) {
	// On most platforms, dialing a closed port should be nearly instantaneous —
	// less than a few hundred milliseconds. However, on some platforms it may be
	// much slower: on Windows and OpenBSD, it has been observed to take up to a
	// few seconds.

	l, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("dialClosedPort: Listen failed: %v", err)
	}
	addr := l.Addr().String()
	l.Close()

	startTime := time.Now()
	c, err := Dial("tcp", addr)
	if err == nil {
		c.Close()
	}
	elapsed := time.Since(startTime)
	t.Logf("dialClosedPort: measured delay %v", elapsed)
	return elapsed
}

func TestDialParallel(t *testing.T) {
	const instant time.Duration = 0
	const fallbackDelay = 200 * time.Millisecond

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
		{[]string{"127.0.0.1", "::1"}, []string{}, "tcp4", true, instant},
		{[]string{"::1", "127.0.0.1"}, []string{}, "tcp6", true, instant},
		// Skip a "connection refused" in the fallback thread.
		{[]string{slowDst4, slowDst6}, []string{"::1", "127.0.0.1"}, "tcp6", true, fallbackDelay},
		// Primary refused, fallback without delay.
		{[]string{"127.0.0.1"}, []string{"::1"}, "tcp4", true, instant},
		{[]string{"::1"}, []string{"127.0.0.1"}, "tcp6", true, instant},
		// Everything is refused.
		{[]string{"127.0.0.1"}, []string{}, "tcp4", false, instant},
		// Nothing to do; fail instantly.
		{[]string{}, []string{}, "", false, instant},
		// Connecting to tons of addresses should not trip the deadline.
		{nCopies("::1", 1000), []string{}, "", true, instant},
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
		i, tt := i, tt
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			dialTCP := func(ctx context.Context, network string, laddr, raddr *TCPAddr) (*TCPConn, error) {
				n := "tcp6"
				if raddr.IP.To4() != nil {
					n = "tcp4"
				}
				if n == tt.teardownNetwork {
					return nil, errors.New("unreachable")
				}
				if r := raddr.IP.String(); r == slowDst4 || r == slowDst6 {
					<-ctx.Done()
					return nil, ctx.Err()
				}
				return &TCPConn{}, nil
			}

			primaries := makeAddrs(tt.primaries, "80")
			fallbacks := makeAddrs(tt.fallbacks, "80")
			d := Dialer{
				FallbackDelay: fallbackDelay,
			}
			const forever = 60 * time.Minute
			if tt.expectElapsed == instant {
				d.FallbackDelay = forever
			}
			startTime := time.Now()
			sd := &sysDialer{
				Dialer:          d,
				network:         "tcp",
				address:         "?",
				testHookDialTCP: dialTCP,
			}
			c, err := sd.dialParallel(context.Background(), primaries, fallbacks)
			elapsed := time.Since(startTime)

			if c != nil {
				c.Close()
			}

			if tt.expectOk && err != nil {
				t.Errorf("#%d: got %v; want nil", i, err)
			} else if !tt.expectOk && err == nil {
				t.Errorf("#%d: got nil; want non-nil", i)
			}

			if elapsed < tt.expectElapsed || elapsed >= forever {
				t.Errorf("#%d: got %v; want >= %v, < forever", i, elapsed, tt.expectElapsed)
			}

			// Repeat each case, ensuring that it can be canceled.
			ctx, cancel := context.WithCancel(context.Background())
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				time.Sleep(5 * time.Millisecond)
				cancel()
				wg.Done()
			}()
			// Ignore errors, since all we care about is that the
			// call can be canceled.
			c, _ = sd.dialParallel(ctx, primaries, fallbacks)
			if c != nil {
				c.Close()
			}
			wg.Wait()
		})
	}
}

func lookupSlowFast(ctx context.Context, fn func(context.Context, string, string) ([]IPAddr, error), network, host string) ([]IPAddr, error) {
	switch host {
	case "slow6loopback4":
		// Returns a slow IPv6 address, and a local IPv4 address.
		return []IPAddr{
			{IP: ParseIP(slowDst6)},
			{IP: ParseIP("127.0.0.1")},
		}, nil
	default:
		return fn(ctx, network, host)
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
		elapsed := time.Since(startTime)
		if err == nil {
			c.Close()
		} else if tt.dualstack {
			t.Error(err)
		}
		expectMin := tt.expectElapsed - 1*time.Millisecond
		expectMax := tt.expectElapsed + 95*time.Millisecond
		if elapsed < expectMin {
			t.Errorf("#%d: got %v; want >= %v", i, elapsed, expectMin)
		}
		if elapsed > expectMax {
			t.Errorf("#%d: got %v; want <= %v", i, elapsed, expectMax)
		}
	}
}

func TestDialParallelSpuriousConnection(t *testing.T) {
	if !supportsIPv4() || !supportsIPv6() {
		t.Skip("both IPv4 and IPv6 are required")
	}

	var readDeadline time.Time
	if td, ok := t.Deadline(); ok {
		const arbitraryCleanupMargin = 1 * time.Second
		readDeadline = td.Add(-arbitraryCleanupMargin)
	} else {
		readDeadline = time.Now().Add(5 * time.Second)
	}

	var closed sync.WaitGroup
	closed.Add(2)
	handler := func(dss *dualStackServer, ln Listener) {
		// Accept one connection per address.
		c, err := ln.Accept()
		if err != nil {
			t.Fatal(err)
		}

		// Workaround for https://go.dev/issue/37795.
		// On arm64 macOS (current as of macOS 12.4),
		// reading from a socket at the same time as the client
		// is closing it occasionally hangs for 60 seconds before
		// returning ECONNRESET. Sleep for a bit to give the
		// socket time to close before trying to read from it.
		if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
			time.Sleep(10 * time.Millisecond)
		}

		// The client should close itself, without sending data.
		c.SetReadDeadline(readDeadline)
		var b [1]byte
		if _, err := c.Read(b[:]); err != io.EOF {
			t.Errorf("got %v; want %v", err, io.EOF)
		}
		c.Close()
		closed.Done()
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

	var dialing sync.WaitGroup
	dialing.Add(2)
	origTestHookDialTCP := testHookDialTCP
	defer func() { testHookDialTCP = origTestHookDialTCP }()
	testHookDialTCP = func(ctx context.Context, net string, laddr, raddr *TCPAddr) (*TCPConn, error) {
		// Wait until Happy Eyeballs kicks in and both connections are dialing,
		// and inhibit cancellation.
		// This forces dialParallel to juggle two successful connections.
		dialing.Done()
		dialing.Wait()

		// Now ignore the provided context (which will be canceled) and use a
		// different one to make sure this completes with a valid connection,
		// which we hope to be closed below:
		sd := &sysDialer{network: net, address: raddr.String()}
		return sd.doDialTCP(context.Background(), laddr, raddr)
	}

	d := Dialer{
		FallbackDelay: fallbackDelay,
	}
	sd := &sysDialer{
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
	c, err := sd.dialParallel(context.Background(), makeAddr("127.0.0.1"), makeAddr("::1"))
	if err != nil {
		t.Fatal(err)
	}
	c.Close()

	// The server should've seen both connections.
	closed.Wait()
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
		{now.Add(0 * time.Millisecond), now, 1, noDeadline, errTimeout},
		{now.Add(1 * time.Millisecond), now, 1, noDeadline, errTimeout},
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

// isEADDRINUSE reports whether err is syscall.EADDRINUSE.
var isEADDRINUSE = func(err error) bool { return false }

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

	issue34264Index := -1
	if supportsIPv4map() {
		issue34264Index = len(tests)
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
	var lss [2]*localServer
	for i, network := range []string{"tcp4", "tcp6"} {
		lss[i] = newLocalServer(t, network)
		defer lss[i].teardown()
		if err := lss[i].buildup(handler); err != nil {
			t.Fatal(err)
		}
	}

	for i, tt := range tests {
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
			if i == issue34264Index && runtime.GOOS == "freebsd" && isEADDRINUSE(err) {
				// https://golang.org/issue/34264: FreeBSD through at least version 12.2
				// has been observed to fail with EADDRINUSE when dialing from an IPv6
				// local address to an IPv4 remote address.
				t.Logf("%s %v->%s: got %v; want %v", tt.network, tt.laddr, tt.raddr, err, tt.error)
				t.Logf("(spurious EADDRINUSE ignored on freebsd: see https://golang.org/issue/34264)")
			} else {
				t.Errorf("%s %v->%s: got %v; want %v", tt.network, tt.laddr, tt.raddr, err, tt.error)
			}
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

	closedPortDelay := dialClosedPort(t)

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
	t.Cleanup(func() {
		testHookSetKeepAlive = func(KeepAliveConfig) {}
	})

	handler := func(ls *localServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}
	ln := newLocalListener(t, "tcp", &ListenConfig{
		KeepAlive: -1, // prevent calling hook from accepting
	})
	ls := (&streamListener{Listener: ln}).newLocalServer()
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		ka       time.Duration
		expected time.Duration
	}{
		{-1, -1},
		{0, 0},
		{5 * time.Second, 5 * time.Second},
		{30 * time.Second, 30 * time.Second},
	}

	var got time.Duration = -1
	testHookSetKeepAlive = func(cfg KeepAliveConfig) { got = cfg.Idle }

	for _, test := range tests {
		got = -1
		d := Dialer{KeepAlive: test.ka}
		c, err := d.Dial("tcp", ls.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		c.Close()
		if got != test.expected {
			t.Errorf("Dialer.KeepAlive = %v: SetKeepAlive set to %v, want %v", d.KeepAlive, got, test.expected)
		}
	}
}

func TestDialCancel(t *testing.T) {
	mustHaveExternalNetwork(t)

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
				// Using strings.Contains is ugly but
				// may work on plan9 and windows.
				ignorable := []string{
					"connection refused",
					"unreachable",
					"no route to host",
					"invalid argument",
				}
				e := err.Error()
				for _, ignore := range ignorable {
					if strings.Contains(e, ignore) {
						t.Skipf("connection to %v failed fast with %v", blackholeIPPort, err)
					}
				}

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

	ln := newLocalListener(t, "tcp")

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

		// Immediately after dialing, request cancellation and sleep.
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

func TestDialClosedPortFailFast(t *testing.T) {
	if runtime.GOOS != "windows" {
		// Reported by go.dev/issues/23366.
		t.Skip("skipping windows only test")
	}
	for _, network := range []string{"tcp", "tcp4", "tcp6"} {
		t.Run(network, func(t *testing.T) {
			if !testableNetwork(network) {
				t.Skipf("skipping: can't listen on %s", network)
			}
			// Reserve a local port till the end of the
			// test by opening a listener and connecting to
			// it using Dial.
			ln := newLocalListener(t, network)
			addr := ln.Addr().String()
			conn1, err := Dial(network, addr)
			if err != nil {
				ln.Close()
				t.Fatal(err)
			}
			defer conn1.Close()
			// Now close the listener so the next Dial fails
			// keeping conn1 alive so the port is not made
			// available.
			ln.Close()

			maxElapsed := time.Second
			// The host can be heavy-loaded and take
			// longer than configured. Retry until
			// Dial takes less than maxElapsed or
			// the test times out.
			for {
				startTime := time.Now()
				conn2, err := Dial(network, addr)
				if err == nil {
					conn2.Close()
					t.Fatal("error expected")
				}
				elapsed := time.Since(startTime)
				if elapsed < maxElapsed {
					break
				}
				t.Logf("got %v; want < %v", elapsed, maxElapsed)
			}
		})
	}
}

// Issue 18806: it should always be possible to net.Dial a
// net.Listener().Addr().String when the listen address was ":n", even
// if the machine has halfway configured IPv6 such that it can bind on
// "::" not connect back to that same address.
func TestDialListenerAddr(t *testing.T) {
	if !testableNetwork("tcp4") {
		t.Skipf("skipping: can't listen on tcp4")
	}

	// The original issue report was for listening on just ":0" on a system that
	// supports both tcp4 and tcp6 for external traffic but only tcp4 for loopback
	// traffic. However, the port opened by ":0" is externally-accessible, and may
	// trigger firewall alerts or otherwise be mistaken for malicious activity
	// (see https://go.dev/issue/59497). Moreover, it often does not reproduce
	// the scenario in the issue, in which the port *cannot* be dialed as tcp6.
	//
	// To address both of those problems, we open a tcp4-only localhost port, but
	// then dial the address string that the listener would have reported for a
	// dual-stack port.
	ln, err := Listen("tcp4", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	t.Logf("listening on %q", ln.Addr())
	_, port, err := SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	// If we had opened a dual-stack port without an explicit "localhost" address,
	// the Listener would arbitrarily report an empty tcp6 address in its Addr
	// string.
	//
	// The documentation for Dial says ‘if the host is empty or a literal
	// unspecified IP address, as in ":80", "0.0.0.0:80" or "[::]:80" for TCP and
	// UDP, "", "0.0.0.0" or "::" for IP, the local system is assumed.’
	// In #18806, it was decided that that should include the local tcp4 host
	// even if the string is in the tcp6 format.
	dialAddr := "[::]:" + port
	c, err := Dial("tcp4", dialAddr)
	if err != nil {
		t.Fatalf(`Dial("tcp4", %q): %v`, dialAddr, err)
	}
	c.Close()
	t.Logf(`Dial("tcp4", %q) succeeded`, dialAddr)
}

func TestDialerControl(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	case "js", "wasip1":
		t.Skipf("skipping: fake net does not support Dialer.Control")
	}

	t.Run("StreamDial", func(t *testing.T) {
		for _, network := range []string{"tcp", "tcp4", "tcp6", "unix", "unixpacket"} {
			if !testableNetwork(network) {
				continue
			}
			ln := newLocalListener(t, network)
			defer ln.Close()
			d := Dialer{Control: controlOnConnSetup}
			c, err := d.Dial(network, ln.Addr().String())
			if err != nil {
				t.Error(err)
				continue
			}
			c.Close()
		}
	})
	t.Run("PacketDial", func(t *testing.T) {
		for _, network := range []string{"udp", "udp4", "udp6", "unixgram"} {
			if !testableNetwork(network) {
				continue
			}
			c1 := newLocalPacketListener(t, network)
			if network == "unixgram" {
				defer os.Remove(c1.LocalAddr().String())
			}
			defer c1.Close()
			d := Dialer{Control: controlOnConnSetup}
			c2, err := d.Dial(network, c1.LocalAddr().String())
			if err != nil {
				t.Error(err)
				continue
			}
			c2.Close()
		}
	})
}

func TestDialerControlContext(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("%s does not have full support of socktest", runtime.GOOS)
	case "js", "wasip1":
		t.Skipf("skipping: fake net does not support Dialer.ControlContext")
	}
	t.Run("StreamDial", func(t *testing.T) {
		for i, network := range []string{"tcp", "tcp4", "tcp6", "unix", "unixpacket"} {
			t.Run(network, func(t *testing.T) {
				if !testableNetwork(network) {
					t.Skipf("skipping: %s not available", network)
				}

				ln := newLocalListener(t, network)
				defer ln.Close()
				var id int
				d := Dialer{ControlContext: func(ctx context.Context, network string, address string, c syscall.RawConn) error {
					id = ctx.Value("id").(int)
					return controlOnConnSetup(network, address, c)
				}}
				c, err := d.DialContext(context.WithValue(context.Background(), "id", i+1), network, ln.Addr().String())
				if err != nil {
					t.Fatal(err)
				}
				if id != i+1 {
					t.Errorf("got id %d, want %d", id, i+1)
				}
				c.Close()
			})
		}
	})
}

func TestDialContext(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	case "js", "wasip1":
		t.Skipf("skipping: fake net does not support Dialer.ControlContext")
	}

	t.Run("StreamDial", func(t *testing.T) {
		var err error
		for i, network := range []string{"tcp", "tcp4", "tcp6", "unix", "unixpacket"} {
			if !testableNetwork(network) {
				continue
			}
			ln := newLocalListener(t, network)
			defer ln.Close()
			var id int
			d := Dialer{ControlContext: func(ctx context.Context, network string, address string, c syscall.RawConn) error {
				id = ctx.Value("id").(int)
				return controlOnConnSetup(network, address, c)
			}}
			var c Conn
			switch network {
			case "tcp", "tcp4", "tcp6":
				raddr, err := netip.ParseAddrPort(ln.Addr().String())
				if err != nil {
					t.Error(err)
					continue
				}
				c, err = d.DialTCP(context.WithValue(context.Background(), "id", i+1), network, (*TCPAddr)(nil).AddrPort(), raddr)
			case "unix", "unixpacket":
				raddr, err := ResolveUnixAddr(network, ln.Addr().String())
				if err != nil {
					t.Error(err)
					continue
				}
				c, err = d.DialUnix(context.WithValue(context.Background(), "id", i+1), network, nil, raddr)
			}
			if err != nil {
				t.Error(err)
				continue
			}
			if id != i+1 {
				t.Errorf("%s: got id %d, want %d", network, id, i+1)
			}
			c.Close()
		}
	})
	t.Run("PacketDial", func(t *testing.T) {
		var err error
		for i, network := range []string{"udp", "udp4", "udp6", "unixgram"} {
			if !testableNetwork(network) {
				continue
			}
			c1 := newLocalPacketListener(t, network)
			if network == "unixgram" {
				defer os.Remove(c1.LocalAddr().String())
			}
			defer c1.Close()
			var id int
			d := Dialer{ControlContext: func(ctx context.Context, network string, address string, c syscall.RawConn) error {
				id = ctx.Value("id").(int)
				return controlOnConnSetup(network, address, c)
			}}
			var c2 Conn
			switch network {
			case "udp", "udp4", "udp6":
				raddr, err := netip.ParseAddrPort(c1.LocalAddr().String())
				if err != nil {
					t.Error(err)
					continue
				}
				c2, err = d.DialUDP(context.WithValue(context.Background(), "id", i+1), network, (*UDPAddr)(nil).AddrPort(), raddr)
			case "unixgram":
				raddr, err := ResolveUnixAddr(network, c1.LocalAddr().String())
				if err != nil {
					t.Error(err)
					continue
				}
				c2, err = d.DialUnix(context.WithValue(context.Background(), "id", i+1), network, nil, raddr)
			}
			if err != nil {
				t.Error(err)
				continue
			}
			if id != i+1 {
				t.Errorf("%s: got id %d, want %d", network, id, i+1)
			}
			c2.Close()
		}
	})
}

// mustHaveExternalNetwork is like testenv.MustHaveExternalNetwork
// except on non-Linux, non-mobile builders it permits the test to
// run in -short mode.
func mustHaveExternalNetwork(t *testing.T) {
	t.Helper()
	definitelyHasLongtestBuilder := runtime.GOOS == "linux"
	mobile := runtime.GOOS == "android" || runtime.GOOS == "ios"
	fake := runtime.GOOS == "js" || runtime.GOOS == "wasip1"
	if testenv.Builder() != "" && !definitelyHasLongtestBuilder && !mobile && !fake {
		// On a non-Linux, non-mobile builder (e.g., freebsd-amd64-13_0).
		//
		// Don't skip testing because otherwise the test may never run on
		// any builder if this port doesn't also have a -longtest builder.
		return
	}
	testenv.MustHaveExternalNetwork(t)
}

type contextWithNonZeroDeadline struct {
	context.Context
}

func (contextWithNonZeroDeadline) Deadline() (time.Time, bool) {
	// Return non-zero time.Time value with false indicating that no deadline is set.
	return time.Unix(0, 0), false
}

func TestDialWithNonZeroDeadline(t *testing.T) {
	ln := newLocalListener(t, "tcp")
	defer ln.Close()
	_, port, err := SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	ctx := contextWithNonZeroDeadline{Context: context.Background()}
	var dialer Dialer
	c, err := dialer.DialContext(ctx, "tcp", JoinHostPort("", port))
	if err != nil {
		t.Fatal(err)
	}
	c.Close()
}
