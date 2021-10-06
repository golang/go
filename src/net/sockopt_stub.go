// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm
// +build js,wasm

package net

import "syscall"

func setDefaultSockopts(s, family, sotype int, ipv6only bool) error {
	return nil
}

func setDefaultListenerSockopts(s int) error {
	return nil
}

func setDefaultMulticastSockopts(s int) error {
	return nil
}

func setReadBuffer(fd *netFD, bytes int) error {
	return syscall.ENOPROTOOPT
}

func setWriteBuffer(fd *netFD, bytes int) error {
	return syscall.ENOPROTOOPT
}

func setKeepAlive(fd *netFD, keepalive bool) error {
	return syscall.ENOPROTOOPT
}

func setLinger(fd *netFD, sec int) error {
	return syscall.ENOPROTOOPT
}
