// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jsonrpc2 is a minimal implementation of the JSON RPC 2 spec.
// https://www.jsonrpc.org/specification
// It is intended to be compatible with other implementations at the wire level.
package jsonrpc2

const (
	// ErrIdleTimeout is returned when serving timed out waiting for new connections.
	ErrIdleTimeout = constError("timed out waiting for new connections")
)

type constError string

func (e constError) Error() string { return string(e) }
