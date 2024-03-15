// -lang=go1.21

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: Downgrading to go1.13 requires at least go1.21,
//       hence the need for -lang=go1.21 at the top.

//go:build go1.13

package p

import "io"

// A "duplicate method" error should be reported for
// all these interfaces, irrespective of which package
// the embedded Reader is coming from.

type _ interface {
	Reader
	Reader // ERROR "duplicate method Read"
}

type Reader interface {
	Read(p []byte) (n int, err error)
}

type _ interface {
	io.Reader
	Reader // ERROR "duplicate method Read"
}

type _ interface {
	io.Reader
	io /* ERROR "duplicate method Read" */ .Reader
}
