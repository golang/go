// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

//go:linkname procUname libc_uname

var procUname uintptr

// utsname represents the fields of a struct utsname defined in <sys/utsname.h>.
type utsname struct {
	Sysname  [257]byte
	Nodename [257]byte
	Release  [257]byte
	Version  [257]byte
	Machine  [257]byte
}

// KernelVersion returns major and minor kernel version numbers
// parsed from the syscall.Uname's Version field, or (0, 0) if the
// version can't be obtained or parsed.
func KernelVersion() (major int, minor int) {
	var un utsname
	_, _, errno := rawSyscall6(uintptr(unsafe.Pointer(&procUname)), 1, uintptr(unsafe.Pointer(&un)), 0, 0, 0, 0, 0)
	if errno != 0 {
		return 0, 0
	}

	// The version string is in the form "<version>.<update>.<sru>.<build>.<reserved>"
	// on Solaris: https://blogs.oracle.com/solaris/post/whats-in-a-uname-
	// Therefore, we use the Version field on Solaris when available.
	ver := un.Version[:]
	if runtime.GOOS == "illumos" {
		// Illumos distributions use different formats without a parsable
		// and unified pattern for the Version field while Release level
		// string is guaranteed to be in x.y or x.y.z format regardless of
		// whether the kernel is Solaris or illumos.
		ver = un.Release[:]
	}

	parseNext := func() (n int) {
		for i, c := range ver {
			if c == '.' {
				ver = ver[i+1:]
				return
			}
			if '0' <= c && c <= '9' {
				n = n*10 + int(c-'0')
			}
		}
		ver = nil
		return
	}

	major = parseNext()
	minor = parseNext()

	return
}

// SupportSockNonblockCloexec tests if SOCK_NONBLOCK and SOCK_CLOEXEC are supported
// for socket() system call, returns true if affirmative.
var SupportSockNonblockCloexec = sync.OnceValue(func() bool {
	// First test if socket() supports SOCK_NONBLOCK and SOCK_CLOEXEC directly.
	s, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM|syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC, 0)
	if err == nil {
		syscall.Close(s)
		return true
	}
	if err != syscall.EPROTONOSUPPORT && err != syscall.EINVAL {
		// Something wrong with socket(), fall back to checking the kernel version.
		if runtime.GOOS == "illumos" {
			return KernelVersionGE(5, 11) // Minimal requirement is SunOS 5.11.
		}
		return KernelVersionGE(11, 4)
	}
	return false
})

// SupportAccept4 tests whether accept4 system call is available.
var SupportAccept4 = sync.OnceValue(func() bool {
	for {
		// Test if the accept4() is available.
		_, _, err := syscall.Accept4(0, syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC)
		if err == syscall.EINTR {
			continue
		}
		return err != syscall.ENOSYS
	}
})

// SupportTCPKeepAliveIdleIntvlCNT determines whether the TCP_KEEPIDLE, TCP_KEEPINTVL and TCP_KEEPCNT
// are available by checking the kernel version for Solaris 11.4.
var SupportTCPKeepAliveIdleIntvlCNT = sync.OnceValue(func() bool {
	return KernelVersionGE(11, 4)
})
