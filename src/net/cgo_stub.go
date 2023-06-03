// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file holds stub versions of the cgo functions called on Unix systems.
// We build this file if using the netgo build tag, or if cgo is not
// enabled and we are using a Unix system other than Darwin, or if it's
// wasip1 where cgo is never available.
// Darwin is exempted because it always provides the cgo routines,
// in cgo_unix_syscall.go.

//go:build netgo || (!cgo && unix && !darwin) || wasip1

package net

import "context"

// cgoAvailable set to false to indicate that the cgo resolver
// is not available on this system.
const cgoAvailable = false

func cgoLookupHost(ctx context.Context, name string) (addrs []string, err error) {
	panic("cgo stub: cgo not available")
}

func cgoLookupPort(ctx context.Context, network, service string) (port int, err error) {
	panic("cgo stub: cgo not available")
}

func cgoLookupIP(ctx context.Context, network, name string) (addrs []IPAddr, err error) {
	panic("cgo stub: cgo not available")
}

func cgoLookupCNAME(ctx context.Context, name string) (cname string, err error, completed bool) {
	panic("cgo stub: cgo not available")
}

func cgoLookupPTR(ctx context.Context, addr string) (ptrs []string, err error) {
	panic("cgo stub: cgo not available")
}
