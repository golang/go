// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !js

package net

import (
	"flag"
	"fmt"
	"net/internal/socktest"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
)

var (
	sw socktest.Switch

	// uninstallTestHooks runs just before a run of benchmarks.
	testHookUninstaller sync.Once
)

var (
	testTCPBig = flag.Bool("tcpbig", false, "whether to test massive size of data per read or write call on TCP connection")

	testDNSFlood = flag.Bool("dnsflood", false, "whether to test DNS query flooding")

	// If external IPv4 connectivity exists, we can try dialing
	// non-node/interface local scope IPv4 addresses.
	// On Windows, Lookup APIs may not return IPv4-related
	// resource records when a node has no external IPv4
	// connectivity.
	testIPv4 = flag.Bool("ipv4", true, "assume external IPv4 connectivity exists")

	// If external IPv6 connectivity exists, we can try dialing
	// non-node/interface local scope IPv6 addresses.
	// On Windows, Lookup APIs may not return IPv6-related
	// resource records when a node has no external IPv6
	// connectivity.
	testIPv6 = flag.Bool("ipv6", false, "assume external IPv6 connectivity exists")
)

func TestMain(m *testing.M) {
	setupTestData()
	installTestHooks()

	st := m.Run()

	testHookUninstaller.Do(uninstallTestHooks)
	if testing.Verbose() {
		printRunningGoroutines()
		printInflightSockets()
		printSocketStats()
	}
	forceCloseSockets()
	os.Exit(st)
}

type ipv6LinkLocalUnicastTest struct {
	network, address string
	nameLookup       bool
}

var (
	ipv6LinkLocalUnicastTCPTests []ipv6LinkLocalUnicastTest
	ipv6LinkLocalUnicastUDPTests []ipv6LinkLocalUnicastTest
)

func setupTestData() {
	if supportsIPv4() {
		resolveTCPAddrTests = append(resolveTCPAddrTests, []resolveTCPAddrTest{
			{"tcp", "localhost:1", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 1}, nil},
			{"tcp4", "localhost:2", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 2}, nil},
		}...)
		resolveUDPAddrTests = append(resolveUDPAddrTests, []resolveUDPAddrTest{
			{"udp", "localhost:1", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 1}, nil},
			{"udp4", "localhost:2", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 2}, nil},
		}...)
		resolveIPAddrTests = append(resolveIPAddrTests, []resolveIPAddrTest{
			{"ip", "localhost", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
			{"ip4", "localhost", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
		}...)
	}

	if supportsIPv6() {
		resolveTCPAddrTests = append(resolveTCPAddrTests, resolveTCPAddrTest{"tcp6", "localhost:3", &TCPAddr{IP: IPv6loopback, Port: 3}, nil})
		resolveUDPAddrTests = append(resolveUDPAddrTests, resolveUDPAddrTest{"udp6", "localhost:3", &UDPAddr{IP: IPv6loopback, Port: 3}, nil})
		resolveIPAddrTests = append(resolveIPAddrTests, resolveIPAddrTest{"ip6", "localhost", &IPAddr{IP: IPv6loopback}, nil})

		// Issue 20911: don't return IPv4 addresses for
		// Resolve*Addr calls of the IPv6 unspecified address.
		resolveTCPAddrTests = append(resolveTCPAddrTests, resolveTCPAddrTest{"tcp", "[::]:4", &TCPAddr{IP: IPv6unspecified, Port: 4}, nil})
		resolveUDPAddrTests = append(resolveUDPAddrTests, resolveUDPAddrTest{"udp", "[::]:4", &UDPAddr{IP: IPv6unspecified, Port: 4}, nil})
		resolveIPAddrTests = append(resolveIPAddrTests, resolveIPAddrTest{"ip", "::", &IPAddr{IP: IPv6unspecified}, nil})
	}

	ifi := loopbackInterface()
	if ifi != nil {
		index := fmt.Sprintf("%v", ifi.Index)
		resolveTCPAddrTests = append(resolveTCPAddrTests, []resolveTCPAddrTest{
			{"tcp6", "[fe80::1%" + ifi.Name + "]:1", &TCPAddr{IP: ParseIP("fe80::1"), Port: 1, Zone: zoneCache.name(ifi.Index)}, nil},
			{"tcp6", "[fe80::1%" + index + "]:2", &TCPAddr{IP: ParseIP("fe80::1"), Port: 2, Zone: index}, nil},
		}...)
		resolveUDPAddrTests = append(resolveUDPAddrTests, []resolveUDPAddrTest{
			{"udp6", "[fe80::1%" + ifi.Name + "]:1", &UDPAddr{IP: ParseIP("fe80::1"), Port: 1, Zone: zoneCache.name(ifi.Index)}, nil},
			{"udp6", "[fe80::1%" + index + "]:2", &UDPAddr{IP: ParseIP("fe80::1"), Port: 2, Zone: index}, nil},
		}...)
		resolveIPAddrTests = append(resolveIPAddrTests, []resolveIPAddrTest{
			{"ip6", "fe80::1%" + ifi.Name, &IPAddr{IP: ParseIP("fe80::1"), Zone: zoneCache.name(ifi.Index)}, nil},
			{"ip6", "fe80::1%" + index, &IPAddr{IP: ParseIP("fe80::1"), Zone: index}, nil},
		}...)
	}

	addr := ipv6LinkLocalUnicastAddr(ifi)
	if addr != "" {
		if runtime.GOOS != "dragonfly" {
			ipv6LinkLocalUnicastTCPTests = append(ipv6LinkLocalUnicastTCPTests, []ipv6LinkLocalUnicastTest{
				{"tcp", "[" + addr + "%" + ifi.Name + "]:0", false},
			}...)
			ipv6LinkLocalUnicastUDPTests = append(ipv6LinkLocalUnicastUDPTests, []ipv6LinkLocalUnicastTest{
				{"udp", "[" + addr + "%" + ifi.Name + "]:0", false},
			}...)
		}
		ipv6LinkLocalUnicastTCPTests = append(ipv6LinkLocalUnicastTCPTests, []ipv6LinkLocalUnicastTest{
			{"tcp6", "[" + addr + "%" + ifi.Name + "]:0", false},
		}...)
		ipv6LinkLocalUnicastUDPTests = append(ipv6LinkLocalUnicastUDPTests, []ipv6LinkLocalUnicastTest{
			{"udp6", "[" + addr + "%" + ifi.Name + "]:0", false},
		}...)
		switch runtime.GOOS {
		case "darwin", "ios", "dragonfly", "freebsd", "openbsd", "netbsd":
			ipv6LinkLocalUnicastTCPTests = append(ipv6LinkLocalUnicastTCPTests, []ipv6LinkLocalUnicastTest{
				{"tcp", "[localhost%" + ifi.Name + "]:0", true},
				{"tcp6", "[localhost%" + ifi.Name + "]:0", true},
			}...)
			ipv6LinkLocalUnicastUDPTests = append(ipv6LinkLocalUnicastUDPTests, []ipv6LinkLocalUnicastTest{
				{"udp", "[localhost%" + ifi.Name + "]:0", true},
				{"udp6", "[localhost%" + ifi.Name + "]:0", true},
			}...)
		case "linux":
			ipv6LinkLocalUnicastTCPTests = append(ipv6LinkLocalUnicastTCPTests, []ipv6LinkLocalUnicastTest{
				{"tcp", "[ip6-localhost%" + ifi.Name + "]:0", true},
				{"tcp6", "[ip6-localhost%" + ifi.Name + "]:0", true},
			}...)
			ipv6LinkLocalUnicastUDPTests = append(ipv6LinkLocalUnicastUDPTests, []ipv6LinkLocalUnicastTest{
				{"udp", "[ip6-localhost%" + ifi.Name + "]:0", true},
				{"udp6", "[ip6-localhost%" + ifi.Name + "]:0", true},
			}...)
		}
	}
}

func printRunningGoroutines() {
	gss := runningGoroutines()
	if len(gss) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "Running goroutines:\n")
	for _, gs := range gss {
		fmt.Fprintf(os.Stderr, "%v\n", gs)
	}
	fmt.Fprintf(os.Stderr, "\n")
}

// runningGoroutines returns a list of remaining goroutines.
func runningGoroutines() []string {
	var gss []string
	b := make([]byte, 2<<20)
	b = b[:runtime.Stack(b, true)]
	for _, s := range strings.Split(string(b), "\n\n") {
		ss := strings.SplitN(s, "\n", 2)
		if len(ss) != 2 {
			continue
		}
		stack := strings.TrimSpace(ss[1])
		if !strings.Contains(stack, "created by net") {
			continue
		}
		gss = append(gss, stack)
	}
	sort.Strings(gss)
	return gss
}

func printInflightSockets() {
	sos := sw.Sockets()
	if len(sos) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "Inflight sockets:\n")
	for s, so := range sos {
		fmt.Fprintf(os.Stderr, "%v: %v\n", s, so)
	}
	fmt.Fprintf(os.Stderr, "\n")
}

func printSocketStats() {
	sts := sw.Stats()
	if len(sts) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "Socket statistical information:\n")
	for _, st := range sts {
		fmt.Fprintf(os.Stderr, "%v\n", st)
	}
	fmt.Fprintf(os.Stderr, "\n")
}
