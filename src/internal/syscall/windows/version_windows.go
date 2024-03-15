// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"sync"
	"syscall"
	"unsafe"
)

// version retrieves the major, minor, and build version numbers
// of the current Windows OS from the RtlGetNtVersionNumbers API
// and parse the results properly.
func version() (major, minor, build uint32) {
	rtlGetNtVersionNumbers(&major, &minor, &build)
	build &= 0x7fff
	return
}

//go:linkname rtlGetNtVersionNumbers syscall.rtlGetNtVersionNumbers
//go:noescape
func rtlGetNtVersionNumbers(majorVersion *uint32, minorVersion *uint32, buildNumber *uint32)

// SupportFullTCPKeepAlive indicates whether the current Windows version
// supports the full TCP keep-alive configurations.
// The minimal requirement is Windows 10.0.16299.
var SupportFullTCPKeepAlive = sync.OnceValue(func() bool {
	major, _, build := version()
	return major >= 10 && build >= 16299
})

// SupportTCPInitialRTONoSYNRetransmissions indicates whether the current
// Windows version supports the TCP_INITIAL_RTO_NO_SYN_RETRANSMISSIONS.
// The minimal requirement is Windows 10.0.16299.
var SupportTCPInitialRTONoSYNRetransmissions = sync.OnceValue(func() bool {
	major, _, build := version()
	return major >= 10 && build >= 16299
})

// SupportUnixSocket indicates whether the current Windows version supports
// Unix Domain Sockets.
// The minimal requirement is Windows 10.0.17063.
var SupportUnixSocket = sync.OnceValue(func() bool {
	var size uint32
	// First call to get the required buffer size in bytes.
	// Ignore the error, it will always fail.
	_, _ = syscall.WSAEnumProtocols(nil, nil, &size)
	n := int32(size) / int32(unsafe.Sizeof(syscall.WSAProtocolInfo{}))
	// Second call to get the actual protocols.
	buf := make([]syscall.WSAProtocolInfo, n)
	n, err := syscall.WSAEnumProtocols(nil, &buf[0], &size)
	if err != nil {
		return false
	}
	for i := int32(0); i < n; i++ {
		if buf[i].AddressFamily == syscall.AF_UNIX {
			return true
		}
	}
	return false
})
