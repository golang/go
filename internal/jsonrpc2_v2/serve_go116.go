// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.16
// +build go1.16

package jsonrpc2

import (
	"errors"
	"net"
)

var errClosed = net.ErrClosed

func isErrClosed(err error) bool {
	return errors.Is(err, errClosed)
}
