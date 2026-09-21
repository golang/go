// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import "errors"

var (
	ErrAbortHandler    = errors.New("net/http: abort Handler")
	ErrBodyNotAllowed  = errors.New("http: request method or response status code does not allow body")
	ErrRequestCanceled = errors.New("net/http: request canceled")
	ErrSkipAltProtocol = errors.New("net/http: skip alternate protocol")
)
