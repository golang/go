// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package nettest provides utilities for network testing.
package nettest

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	stackOnce          sync.Once
	ipv4Enabled        bool
	ipv6Enabled        bool
	unStrmDgramEnabled bool
	rawSocketSess      bool

	aLongTimeAgo = time.Unix(233431200, 0)
	neverTimeout = time.Time{}

	errNoAvailableInterface = errors.New("no available interface")
	errNoAvailableAddress   = errors.New("no available address")
)

func probeStack() {
	if ln, err := net.Listen("tcp4", "127.0.0.1:0"); err == nil {
		ln.Close()
		ipv4Enabled = true
	}
	if ln, err := net.Listen("tcp6", "[::1]:0"); err == nil {
		ln.Close()
		ipv6Enabled = true
	}
	rawSocketSess = supportsRawSocket()
	switch runtime.GOOS {
	case "aix":
		// Unix network isn't properly working on AIX 7.2 with
		// Technical Level < 2.
		out, _ := exec.Command("oslevel", "-s").Output()
		if len(out) >= len("7200-XX-ZZ-YYMM") { // AIX 7.2, Tech Level XX, Service Pack ZZ, date YYMM
			ver := string(out[:4])
			tl, _ := strconv.Atoi(string(out[5:7]))
			unStrmDgramEnabled = ver > "7200" || (ver == "7200" && tl >= 2)
		}
	default:
		unStrmDgramEnabled = true
	}
}

func unixStrmDgramEnabled() bool {
	stackOnce.Do(probeStack)
	return unStrmDgramEnabled
}

// SupportsIPv4 reports whether the platform supports IPv4 networking
// functionality.
func SupportsIPv4() bool {
	stackOnce.Do(probeStack)
	return ipv4Enabled
}

// SupportsIPv6 reports whether the platform supports IPv6 networking
// functionality.
func SupportsIPv6() bool {
	stackOnce.Do(probeStack)
	return ipv6Enabled
}

// SupportsRawSocket reports whether the current session is available
// to use raw sockets.
func SupportsRawSocket() bool {
	stackOnce.Do(probeStack)
	return rawSocketSess
}

// TestableNetwork reports whether network is testable on the current
// platform configuration.
//
// See func Dial of the standard library for the supported networks.
func TestableNetwork(network string) bool {
	ss := strings.Split(network, ":")
	switch ss[0] {
	case "ip+nopriv":
		// This is an internal network name for testing on the
		// package net of the standard library.
		switch runtime.GOOS {
		case "android", "fuchsia", "hurd", "ios", "js", "nacl", "plan9", "windows":
			return false
		}
	case "ip", "ip4", "ip6":
		switch runtime.GOOS {
		case "fuchsia", "hurd", "js", "nacl", "plan9":
			return false
		default:
			if os.Getuid() != 0 {
				return false
			}
		}
	case "unix", "unixgram":
		switch runtime.GOOS {
		case "android", "fuchsia", "hurd", "ios", "js", "nacl", "plan9", "windows":
			return false
		case "aix":
			return unixStrmDgramEnabled()
		}
	case "unixpacket":
		switch runtime.GOOS {
		case "aix", "android", "fuchsia", "hurd", "darwin", "ios", "js", "nacl", "plan9", "windows", "zos":
			return false
		case "netbsd":
			// It passes on amd64 at least. 386 fails
			// (Issue 22927). arm is unknown.
			if runtime.GOARCH == "386" {
				return false
			}
		}
	}
	switch ss[0] {
	case "tcp4", "udp4", "ip4":
		return SupportsIPv4()
	case "tcp6", "udp6", "ip6":
		return SupportsIPv6()
	}
	return true
}

// TestableAddress reports whether address of network is testable on
// the current platform configuration.
func TestableAddress(network, address string) bool {
	switch ss := strings.Split(network, ":"); ss[0] {
	case "unix", "unixgram", "unixpacket":
		// Abstract unix domain sockets, a Linux-ism.
		if address[0] == '@' && runtime.GOOS != "linux" {
			return false
		}
	}
	return true
}

// NewLocalListener returns a listener which listens to a loopback IP
// address or local file system path.
//
// The provided network must be "tcp", "tcp4", "tcp6", "unix" or
// "unixpacket".
func NewLocalListener(network string) (net.Listener, error) {
	switch network {
	case "tcp":
		if SupportsIPv4() {
			if ln, err := net.Listen("tcp4", "127.0.0.1:0"); err == nil {
				return ln, nil
			}
		}
		if SupportsIPv6() {
			return net.Listen("tcp6", "[::1]:0")
		}
	case "tcp4":
		if SupportsIPv4() {
			return net.Listen("tcp4", "127.0.0.1:0")
		}
	case "tcp6":
		if SupportsIPv6() {
			return net.Listen("tcp6", "[::1]:0")
		}
	case "unix", "unixpacket":
		path, err := LocalPath()
		if err != nil {
			return nil, err
		}
		return net.Listen(network, path)
	}
	return nil, fmt.Errorf("%s is not supported on %s/%s", network, runtime.GOOS, runtime.GOARCH)
}

// NewLocalPacketListener returns a packet listener which listens to a
// loopback IP address or local file system path.
//
// The provided network must be "udp", "udp4", "udp6" or "unixgram".
func NewLocalPacketListener(network string) (net.PacketConn, error) {
	switch network {
	case "udp":
		if SupportsIPv4() {
			if c, err := net.ListenPacket("udp4", "127.0.0.1:0"); err == nil {
				return c, nil
			}
		}
		if SupportsIPv6() {
			return net.ListenPacket("udp6", "[::1]:0")
		}
	case "udp4":
		if SupportsIPv4() {
			return net.ListenPacket("udp4", "127.0.0.1:0")
		}
	case "udp6":
		if SupportsIPv6() {
			return net.ListenPacket("udp6", "[::1]:0")
		}
	case "unixgram":
		path, err := LocalPath()
		if err != nil {
			return nil, err
		}
		return net.ListenPacket(network, path)
	}
	return nil, fmt.Errorf("%s is not supported on %s/%s", network, runtime.GOOS, runtime.GOARCH)
}

// LocalPath returns a local path that can be used for Unix-domain
// protocol testing.
func LocalPath() (string, error) {
	dir := ""
	if runtime.GOOS == "darwin" {
		dir = "/tmp"
	}
	f, err := ioutil.TempFile(dir, "go-nettest")
	if err != nil {
		return "", err
	}
	path := f.Name()
	f.Close()
	os.Remove(path)
	return path, nil
}

// MulticastSource returns a unicast IP address on ifi when ifi is an
// IP multicast-capable network interface.
//
// The provided network must be "ip", "ip4" or "ip6".
func MulticastSource(network string, ifi *net.Interface) (net.IP, error) {
	switch network {
	case "ip", "ip4", "ip6":
	default:
		return nil, errNoAvailableAddress
	}
	if ifi == nil || ifi.Flags&net.FlagUp == 0 || ifi.Flags&net.FlagMulticast == 0 {
		return nil, errNoAvailableAddress
	}
	ip, ok := hasRoutableIP(network, ifi)
	if !ok {
		return nil, errNoAvailableAddress
	}
	return ip, nil
}

// LoopbackInterface returns an available logical network interface
// for loopback test.
func LoopbackInterface() (*net.Interface, error) {
	ift, err := net.Interfaces()
	if err != nil {
		return nil, errNoAvailableInterface
	}
	for _, ifi := range ift {
		if ifi.Flags&net.FlagLoopback != 0 && ifi.Flags&net.FlagUp != 0 {
			return &ifi, nil
		}
	}
	return nil, errNoAvailableInterface
}

// RoutedInterface returns a network interface that can route IP
// traffic and satisfies flags.
//
// The provided network must be "ip", "ip4" or "ip6".
func RoutedInterface(network string, flags net.Flags) (*net.Interface, error) {
	switch network {
	case "ip", "ip4", "ip6":
	default:
		return nil, errNoAvailableInterface
	}
	ift, err := net.Interfaces()
	if err != nil {
		return nil, errNoAvailableInterface
	}
	for _, ifi := range ift {
		if ifi.Flags&flags != flags {
			continue
		}
		if _, ok := hasRoutableIP(network, &ifi); !ok {
			continue
		}
		return &ifi, nil
	}
	return nil, errNoAvailableInterface
}

func hasRoutableIP(network string, ifi *net.Interface) (net.IP, bool) {
	ifat, err := ifi.Addrs()
	if err != nil {
		return nil, false
	}
	for _, ifa := range ifat {
		switch ifa := ifa.(type) {
		case *net.IPAddr:
			if ip, ok := routableIP(network, ifa.IP); ok {
				return ip, true
			}
		case *net.IPNet:
			if ip, ok := routableIP(network, ifa.IP); ok {
				return ip, true
			}
		}
	}
	return nil, false
}

func routableIP(network string, ip net.IP) (net.IP, bool) {
	if !ip.IsLoopback() && !ip.IsLinkLocalUnicast() && !ip.IsGlobalUnicast() {
		return nil, false
	}
	switch network {
	case "ip4":
		if ip := ip.To4(); ip != nil {
			return ip, true
		}
	case "ip6":
		if ip.IsLoopback() { // addressing scope of the loopback address depends on each implementation
			return nil, false
		}
		if ip := ip.To16(); ip != nil && ip.To4() == nil {
			return ip, true
		}
	default:
		if ip := ip.To4(); ip != nil {
			return ip, true
		}
		if ip := ip.To16(); ip != nil {
			return ip, true
		}
	}
	return nil, false
}
