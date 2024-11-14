// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package tls

import (
	"errors"
	"syscall"
)

func init() {
	isConnRefused = func { err -> errors.Is(err, syscall.ECONNREFUSED) }
}
