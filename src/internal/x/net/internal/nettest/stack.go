// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package nettest provides utilities for network testing.
package nettest

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"runtime"
)

var (
	supportsIPv4 bool
	supportsIPv6 bool
)

func init() {
	if ln, err := net.Listen("tcp4", "127.0.0.1:0"); err == nil {
		ln.Close()
		supportsIPv4 = true
	}
	if ln, err := net.Listen("tcp6", "[::1]:0"); err == nil {
		ln.Close()
		supportsIPv6 = true
	}
}

// SupportsIPv4 reports whether the platform supports IPv4 networking
// functionality.
func SupportsIPv4() bool { return supportsIPv4 }

// SupportsIPv6 reports whether the platform supports IPv6 networking
// functionality.
func SupportsIPv6() bool { return supportsIPv6 }

// SupportsRawIPSocket reports whether the platform supports raw IP
// sockets.
func SupportsRawIPSocket() (string, bool) {
	return supportsRawIPSocket()
}

// SupportsIPv6MulticastDeliveryOnLoopback reports whether the
// platform supports IPv6 multicast packet delivery on software
// loopback interface.
func SupportsIPv6MulticastDeliveryOnLoopback() bool {
	return supportsIPv6MulticastDeliveryOnLoopback()
}

// ProtocolNotSupported reports whether err is a protocol not
// supported error.
func ProtocolNotSupported(err error) bool {
	return protocolNotSupported(err)
}

// TestableNetwork reports whether network is testable on the current
// platform configuration.
func TestableNetwork(network string) bool {
	// This is based on logic from standard library's
	// net/platform_test.go.
	switch network {
	case "unix", "unixgram":
		switch runtime.GOOS {
		case "android", "js", "nacl", "plan9", "windows":
			return false
		}
		if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
			return false
		}
	case "unixpacket":
		switch runtime.GOOS {
		case "aix", "android", "darwin", "freebsd", "js", "nacl", "plan9", "windows":
			return false
		case "netbsd":
			// It passes on amd64 at least. 386 fails (Issue 22927). arm is unknown.
			if runtime.GOARCH == "386" {
				return false
			}
		}
	}
	return true
}

// NewLocalListener returns a listener which listens to a loopback IP
// address or local file system path.
// Network must be "tcp", "tcp4", "tcp6", "unix" or "unixpacket".
func NewLocalListener(network string) (net.Listener, error) {
	switch network {
	case "tcp":
		if supportsIPv4 {
			if ln, err := net.Listen("tcp4", "127.0.0.1:0"); err == nil {
				return ln, nil
			}
		}
		if supportsIPv6 {
			return net.Listen("tcp6", "[::1]:0")
		}
	case "tcp4":
		if supportsIPv4 {
			return net.Listen("tcp4", "127.0.0.1:0")
		}
	case "tcp6":
		if supportsIPv6 {
			return net.Listen("tcp6", "[::1]:0")
		}
	case "unix", "unixpacket":
		return net.Listen(network, localPath())
	}
	return nil, fmt.Errorf("%s is not supported", network)
}

// NewLocalPacketListener returns a packet listener which listens to a
// loopback IP address or local file system path.
// Network must be "udp", "udp4", "udp6" or "unixgram".
func NewLocalPacketListener(network string) (net.PacketConn, error) {
	switch network {
	case "udp":
		if supportsIPv4 {
			if c, err := net.ListenPacket("udp4", "127.0.0.1:0"); err == nil {
				return c, nil
			}
		}
		if supportsIPv6 {
			return net.ListenPacket("udp6", "[::1]:0")
		}
	case "udp4":
		if supportsIPv4 {
			return net.ListenPacket("udp4", "127.0.0.1:0")
		}
	case "udp6":
		if supportsIPv6 {
			return net.ListenPacket("udp6", "[::1]:0")
		}
	case "unixgram":
		return net.ListenPacket(network, localPath())
	}
	return nil, fmt.Errorf("%s is not supported", network)
}

func localPath() string {
	f, err := ioutil.TempFile("", "nettest")
	if err != nil {
		panic(err)
	}
	path := f.Name()
	f.Close()
	os.Remove(path)
	return path
}
