// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/testenv"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

var unixEnabledOnAIX bool

func init() {
	if runtime.GOOS == "aix" {
		// Unix network isn't properly working on AIX 7.2 with
		// Technical Level < 2.
		// The information is retrieved only once in this init()
		// instead of everytime testableNetwork is called.
		out, _ := exec.Command("oslevel", "-s").Output()
		if len(out) >= len("7200-XX-ZZ-YYMM") { // AIX 7.2, Tech Level XX, Service Pack ZZ, date YYMM
			aixVer := string(out[:4])
			tl, _ := strconv.Atoi(string(out[5:7]))
			unixEnabledOnAIX = aixVer > "7200" || (aixVer == "7200" && tl >= 2)
		}
	}
}

// testableNetwork reports whether network is testable on the current
// platform configuration.
func testableNetwork(network string) bool {
	net, _, _ := strings.Cut(network, ":")
	switch net {
	case "ip+nopriv":
	case "ip", "ip4", "ip6":
		switch runtime.GOOS {
		case "plan9":
			return false
		default:
			if os.Getuid() != 0 {
				return false
			}
		}
	case "unix", "unixgram":
		switch runtime.GOOS {
		case "android", "plan9", "windows":
			return false
		case "aix":
			return unixEnabledOnAIX
		}
		// iOS does not support unix, unixgram.
		if iOS() {
			return false
		}
	case "unixpacket":
		switch runtime.GOOS {
		case "aix", "android", "darwin", "ios", "plan9", "windows":
			return false
		case "netbsd":
			// It passes on amd64 at least. 386 fails (Issue 22927). arm is unknown.
			if runtime.GOARCH == "386" {
				return false
			}
		}
	}
	switch net {
	case "tcp4", "udp4", "ip4":
		if !supportsIPv4() {
			return false
		}
	case "tcp6", "udp6", "ip6":
		if !supportsIPv6() {
			return false
		}
	}
	return true
}

func iOS() bool {
	return runtime.GOOS == "ios"
}

// testableAddress reports whether address of network is testable on
// the current platform configuration.
func testableAddress(network, address string) bool {
	switch net, _, _ := strings.Cut(network, ":"); net {
	case "unix", "unixgram", "unixpacket":
		// Abstract unix domain sockets, a Linux-ism.
		if address[0] == '@' && runtime.GOOS != "linux" {
			return false
		}
	}
	return true
}

// testableListenArgs reports whether arguments are testable on the
// current platform configuration.
func testableListenArgs(network, address, client string) bool {
	if !testableNetwork(network) || !testableAddress(network, address) {
		return false
	}

	var err error
	var addr Addr
	switch net, _, _ := strings.Cut(network, ":"); net {
	case "tcp", "tcp4", "tcp6":
		addr, err = ResolveTCPAddr("tcp", address)
	case "udp", "udp4", "udp6":
		addr, err = ResolveUDPAddr("udp", address)
	case "ip", "ip4", "ip6":
		addr, err = ResolveIPAddr("ip", address)
	default:
		return true
	}
	if err != nil {
		return false
	}
	var ip IP
	var wildcard bool
	switch addr := addr.(type) {
	case *TCPAddr:
		ip = addr.IP
		wildcard = addr.isWildcard()
	case *UDPAddr:
		ip = addr.IP
		wildcard = addr.isWildcard()
	case *IPAddr:
		ip = addr.IP
		wildcard = addr.isWildcard()
	}

	// Test wildcard IP addresses.
	if wildcard && !testenv.HasExternalNetwork() {
		return false
	}

	// Test functionality of IPv4 communication using AF_INET and
	// IPv6 communication using AF_INET6 sockets.
	if !supportsIPv4() && ip.To4() != nil {
		return false
	}
	if !supportsIPv6() && ip.To16() != nil && ip.To4() == nil {
		return false
	}
	cip := ParseIP(client)
	if cip != nil {
		if !supportsIPv4() && cip.To4() != nil {
			return false
		}
		if !supportsIPv6() && cip.To16() != nil && cip.To4() == nil {
			return false
		}
	}

	// Test functionality of IPv4 communication using AF_INET6
	// sockets.
	if !supportsIPv4map() && supportsIPv4() && (network == "tcp" || network == "udp" || network == "ip") && wildcard {
		// At this point, we prefer IPv4 when ip is nil.
		// See favoriteAddrFamily for further information.
		if ip.To16() != nil && ip.To4() == nil && cip.To4() != nil { // a pair of IPv6 server and IPv4 client
			return false
		}
		if (ip.To4() != nil || ip == nil) && cip.To16() != nil && cip.To4() == nil { // a pair of IPv4 server and IPv6 client
			return false
		}
	}

	return true
}

func condFatalf(t *testing.T, network string, format string, args ...interface{}) {
	t.Helper()
	// A few APIs like File and Read/WriteMsg{UDP,IP} are not
	// fully implemented yet on Plan 9 and Windows.
	switch runtime.GOOS {
	case "windows":
		if network == "file+net" {
			t.Logf(format, args...)
			return
		}
	case "plan9":
		t.Logf(format, args...)
		return
	}
	t.Fatalf(format, args...)
}
