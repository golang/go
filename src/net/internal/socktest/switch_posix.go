// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9

package socktest

import (
	"fmt"
	"syscall"
)

func familyString(family int) string {
	switch family {
	case syscall.AF_INET:
		return "inet4"
	case syscall.AF_INET6:
		return "inet6"
	case syscall.AF_UNIX:
		return "local"
	default:
		return fmt.Sprintf("%d", family)
	}
}

func typeString(sotype int) string {
	var s string
	switch sotype & 0xff {
	case syscall.SOCK_STREAM:
		s = "stream"
	case syscall.SOCK_DGRAM:
		s = "datagram"
	case syscall.SOCK_RAW:
		s = "raw"
	case syscall.SOCK_SEQPACKET:
		s = "seqpacket"
	default:
		s = fmt.Sprintf("%d", sotype&0xff)
	}
	if flags := uint(sotype) & ^uint(0xff); flags != 0 {
		s += fmt.Sprintf("|%#x", flags)
	}
	return s
}

func protocolString(proto int) string {
	switch proto {
	case 0:
		return "default"
	case syscall.IPPROTO_TCP:
		return "tcp"
	case syscall.IPPROTO_UDP:
		return "udp"
	default:
		return fmt.Sprintf("%d", proto)
	}
}
