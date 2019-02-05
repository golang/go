// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl js,wasm

package net

import (
	"syscall"
	"time"
)

func setNoDelay(fd *netFD, noDelay bool) error {
	return syscall.ENOPROTOOPT
}

func setKeepAlivePeriod(fd *netFD, d time.Duration) error {
	return syscall.ENOPROTOOPT
}
