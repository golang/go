// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || freebsd || linux || netbsd || openbsd || solaris

package syscall

import (
	"runtime"
)

// Round the length of a raw sockaddr up to align it properly.
func cmsgAlignOf(salen int) int {
	salign := sizeofPtr

	// dragonfly needs to check ABI version at runtime, see cmsgAlignOf in
	// sockcmsg_dragonfly.go
	switch runtime.GOOS {
	case "aix":
		// There is no alignment on AIX.
		salign = 1
	case "darwin", "ios", "illumos", "solaris":
		// NOTE: It seems like 64-bit Darwin, Illumos and Solaris
		// kernels still require 32-bit aligned access to network
		// subsystem.
		if sizeofPtr == 8 {
			salign = 4
		}
	case "netbsd", "openbsd":
		// NetBSD and OpenBSD armv7 require 64-bit alignment.
		if runtime.GOARCH == "arm" {
			salign = 8
		}
		// NetBSD aarch64 requires 128-bit alignment.
		if runtime.GOOS == "netbsd" && runtime.GOARCH == "arm64" {
			salign = 16
		}
	}

	return (salen + salign - 1) & ^(salign - 1)
}
