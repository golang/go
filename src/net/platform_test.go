// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"runtime"
	"strings"
	"testing"
)

// testableNetwork reports whether network is testable on the current
// platform configuration.
func testableNetwork(network string) bool {
	ss := strings.Split(network, ":")
	switch ss[0] {
	case "ip+nopriv":
		switch runtime.GOOS {
		case "nacl":
			return false
		}
	case "ip", "ip4", "ip6":
		switch runtime.GOOS {
		case "nacl", "plan9":
			return false
		default:
			if os.Getuid() != 0 {
				return false
			}
		}
	case "unix", "unixgram":
		switch runtime.GOOS {
		case "nacl", "plan9", "windows":
			return false
		}
		// iOS does not support unix, unixgram.
		if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
			return false
		}
	case "unixpacket":
		switch runtime.GOOS {
		case "android", "darwin", "nacl", "plan9", "windows":
			fallthrough
		case "freebsd": // FreeBSD 8 and below don't support unixpacket
			return false
		}
	}
	switch ss[0] {
	case "tcp4", "udp4", "ip4":
		if !supportsIPv4 {
			return false
		}
	case "tcp6", "udp6", "ip6":
		if !supportsIPv6 {
			return false
		}
	}
	return true
}

// testableAddress reports whether address of network is testable on
// the current platform configuration.
func testableAddress(network, address string) bool {
	switch ss := strings.Split(network, ":"); ss[0] {
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
	switch ss := strings.Split(network, ":"); ss[0] {
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
	if wildcard && (testing.Short() || !*testExternal) {
		return false
	}

	// Test functionality of IPv4 communication using AF_INET and
	// IPv6 communication using AF_INET6 sockets.
	if !supportsIPv4 && ip.To4() != nil {
		return false
	}
	if !supportsIPv6 && ip.To16() != nil && ip.To4() == nil {
		return false
	}
	cip := ParseIP(client)
	if cip != nil {
		if !supportsIPv4 && cip.To4() != nil {
			return false
		}
		if !supportsIPv6 && cip.To16() != nil && cip.To4() == nil {
			return false
		}
	}

	// Test functionality of IPv4 communication using AF_INET6
	// sockets.
	if !supportsIPv4map && (network == "tcp" || network == "udp" || network == "ip") && wildcard {
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

var condFatalf = func() func(*testing.T, string, ...interface{}) {
	// A few APIs, File, Read/WriteMsg{UDP,IP}, are not
	// implemented yet on both Plan 9 and Windows.
	switch runtime.GOOS {
	case "plan9", "windows":
		return (*testing.T).Logf
	}
	return (*testing.T).Fatalf
}()
