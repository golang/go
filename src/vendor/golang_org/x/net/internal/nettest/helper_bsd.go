// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package nettest

import (
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

var darwinVersion int

func init() {
	if runtime.GOOS == "darwin" {
		// See http://support.apple.com/kb/HT1633.
		s, err := syscall.Sysctl("kern.osrelease")
		if err != nil {
			return
		}
		ss := strings.Split(s, ".")
		if len(ss) == 0 {
			return
		}
		darwinVersion, _ = strconv.Atoi(ss[0])
	}
}

func supportsIPv6MulticastDeliveryOnLoopback() bool {
	switch runtime.GOOS {
	case "freebsd":
		// See http://www.freebsd.org/cgi/query-pr.cgi?pr=180065.
		// Even after the fix, it looks like the latest
		// kernels don't deliver link-local scoped multicast
		// packets correctly.
		return false
	case "darwin":
		return !causesIPv6Crash()
	default:
		return true
	}
}

func causesIPv6Crash() bool {
	// We see some kernel crash when running IPv6 with IP-level
	// options on Darwin kernel version 12 or below.
	// See golang.org/issues/17015.
	return darwinVersion < 13
}
