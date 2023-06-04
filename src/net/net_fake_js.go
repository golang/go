// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fake networking for js/wasm. It is intended to allow tests of other package to pass.

//go:build js && wasm

package net

import "internal/poll"

// Network file descriptor.
type netFD struct {
	*fakeNetFD

	// immutable until Close
	family int
	sotype int
	net    string
	laddr  Addr
	raddr  Addr

	// unused
	pfd         poll.FD
	isConnected bool // handshake completed or use of association with peer
}
