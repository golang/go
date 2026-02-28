// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"errors"
	"sync"
	"syscall"
	"unsafe"
)

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/ns-wdm-_osversioninfoexw
type _OSVERSIONINFOEXW struct {
	osVersionInfoSize uint32
	majorVersion      uint32
	minorVersion      uint32
	buildNumber       uint32
	platformId        uint32
	csdVersion        [128]uint16
	servicePackMajor  uint16
	servicePackMinor  uint16
	suiteMask         uint16
	productType       byte
	reserved          byte
}

// According to documentation, RtlGetVersion function always succeeds.
//sys	rtlGetVersion(info *_OSVERSIONINFOEXW) = ntdll.RtlGetVersion

// Retrieves version information of the current Windows OS
// from the RtlGetVersion API.
func getVersionInfo() *_OSVERSIONINFOEXW {
	info := _OSVERSIONINFOEXW{}
	info.osVersionInfoSize = uint32(unsafe.Sizeof(info))
	rtlGetVersion(&info)
	return &info
}

// Version retrieves the major, minor, and build version numbers
// of the current Windows OS from the RtlGetVersion API.
func Version() (major, minor, build uint32) {
	info := getVersionInfo()
	return info.majorVersion, info.minorVersion, info.buildNumber
}

// SupportUnlimitedTransmitFile indicates whether the current
// Windows version's TransmitFile function imposes any
// concurrent operation limits.
// Workstation and client versions of Windows limit the number
// of concurrent TransmitFile operations allowed on the system
// to a maximum of two. Please see:
// https://learn.microsoft.com/en-us/windows/win32/api/mswsock/nf-mswsock-transmitfile
// https://golang.org/issue/73746
var SupportUnlimitedTransmitFile = sync.OnceValue(func() bool {
	info := getVersionInfo()
	return info.productType != VER_NT_WORKSTATION
})

var (
	supportTCPKeepAliveIdle     bool
	supportTCPKeepAliveInterval bool
	supportTCPKeepAliveCount    bool
)

var initTCPKeepAlive = sync.OnceFunc(func() {
	s, err := WSASocket(syscall.AF_INET, syscall.SOCK_STREAM, syscall.IPPROTO_TCP, nil, 0, WSA_FLAG_NO_HANDLE_INHERIT)
	if err != nil {
		// Fallback to checking the Windows version.
		major, _, build := Version()
		supportTCPKeepAliveIdle = major >= 10 && build >= 16299
		supportTCPKeepAliveInterval = major >= 10 && build >= 16299
		supportTCPKeepAliveCount = major >= 10 && build >= 15063
		return
	}
	defer syscall.Closesocket(s)
	var optSupported = func(opt int) bool {
		err := syscall.SetsockoptInt(s, syscall.IPPROTO_TCP, opt, 1)
		return !errors.Is(err, syscall.WSAENOPROTOOPT)
	}
	supportTCPKeepAliveIdle = optSupported(TCP_KEEPIDLE)
	supportTCPKeepAliveInterval = optSupported(TCP_KEEPINTVL)
	supportTCPKeepAliveCount = optSupported(TCP_KEEPCNT)
})

// SupportTCPKeepAliveIdle indicates whether TCP_KEEPIDLE is supported.
// The minimal requirement is Windows 10.0.16299.
func SupportTCPKeepAliveIdle() bool {
	initTCPKeepAlive()
	return supportTCPKeepAliveIdle
}

// SupportTCPKeepAliveInterval indicates whether TCP_KEEPINTVL is supported.
// The minimal requirement is Windows 10.0.16299.
func SupportTCPKeepAliveInterval() bool {
	initTCPKeepAlive()
	return supportTCPKeepAliveInterval
}

// SupportTCPKeepAliveCount indicates whether TCP_KEEPCNT is supported.
// supports TCP_KEEPCNT.
// The minimal requirement is Windows 10.0.15063.
func SupportTCPKeepAliveCount() bool {
	initTCPKeepAlive()
	return supportTCPKeepAliveCount
}

// SupportTCPInitialRTONoSYNRetransmissions indicates whether the current
// Windows version supports the TCP_INITIAL_RTO_NO_SYN_RETRANSMISSIONS.
// The minimal requirement is Windows 10.0.16299.
var SupportTCPInitialRTONoSYNRetransmissions = sync.OnceValue(func() bool {
	major, _, build := Version()
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
