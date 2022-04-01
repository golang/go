// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.16
// +build !go1.16

package jsonrpc2

import (
	"errors"
	"strings"
)

// errClosed is an error with the same string as net.ErrClosed,
// which was added in Go 1.16.
var errClosed = errors.New("use of closed network connection")

// isErrClosed reports whether err ends in the same string as errClosed.
func isErrClosed(err error) bool {
	// As of Go 1.16, this could be 'errors.Is(err, net.ErrClosing)', but
	// unfortunately gopls still requires compatiblity with
	// (otherwise-unsupported) older Go versions.
	//
	// In the meantime, this error strirng has not changed on any supported Go
	// version, and is not expected to change in the future.
	// This is not ideal, but since the worst that could happen here is some
	// superfluous logging, it is acceptable.
	return strings.HasSuffix(err.Error(), "use of closed network connection")
}
