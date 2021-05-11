// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !plan9
// +build !js,!plan9

package net

import (
	"net/internal/socktest"
	"strings"
	"syscall"
)

func enableSocketConnect() {
	sw.Set(socktest.FilterConnect, nil)
}

func disableSocketConnect(network string) {
	ss := strings.Split(network, ":")
	sw.Set(socktest.FilterConnect, func(so *socktest.Status) (socktest.AfterFilter, error) {
		switch ss[0] {
		case "tcp4":
			if so.Cookie.Family() == syscall.AF_INET && so.Cookie.Type() == syscall.SOCK_STREAM {
				return nil, syscall.EHOSTUNREACH
			}
		case "udp4":
			if so.Cookie.Family() == syscall.AF_INET && so.Cookie.Type() == syscall.SOCK_DGRAM {
				return nil, syscall.EHOSTUNREACH
			}
		case "ip4":
			if so.Cookie.Family() == syscall.AF_INET && so.Cookie.Type() == syscall.SOCK_RAW {
				return nil, syscall.EHOSTUNREACH
			}
		case "tcp6":
			if so.Cookie.Family() == syscall.AF_INET6 && so.Cookie.Type() == syscall.SOCK_STREAM {
				return nil, syscall.EHOSTUNREACH
			}
		case "udp6":
			if so.Cookie.Family() == syscall.AF_INET6 && so.Cookie.Type() == syscall.SOCK_DGRAM {
				return nil, syscall.EHOSTUNREACH
			}
		case "ip6":
			if so.Cookie.Family() == syscall.AF_INET6 && so.Cookie.Type() == syscall.SOCK_RAW {
				return nil, syscall.EHOSTUNREACH
			}
		}
		return nil, nil
	})
}
