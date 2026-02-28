// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js || plan9 || wasip1

package net

import (
	"errors"
	"syscall"
)

func readRawConn(c syscall.RawConn, b []byte) (int, error) {
	return 0, errors.New("not supported")
}

func writeRawConn(c syscall.RawConn, b []byte) error {
	return errors.New("not supported")
}

func controlRawConn(c syscall.RawConn, addr Addr) error {
	return errors.New("not supported")
}

func controlOnConnSetup(network string, address string, c syscall.RawConn) error {
	return nil
}
