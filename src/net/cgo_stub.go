// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cgo || netgo
// +build !cgo netgo

package net

import "context"

func init() { netGo = true }

type addrinfoErrno int

func (eai addrinfoErrno) Error() string   { return "<nil>" }
func (eai addrinfoErrno) Temporary() bool { return false }
func (eai addrinfoErrno) Timeout() bool   { return false }

func cgoLookupHost(ctx context.Context, name string) (addrs []string, err error, completed bool) {
	return nil, nil, false
}

func cgoLookupPort(ctx context.Context, network, service string) (port int, err error, completed bool) {
	return 0, nil, false
}

func cgoLookupIP(ctx context.Context, network, name string) (addrs []IPAddr, err error, completed bool) {
	return nil, nil, false
}

func cgoLookupCNAME(ctx context.Context, name string) (cname string, err error, completed bool) {
	return "", nil, false
}

func cgoLookupPTR(ctx context.Context, addr string) (ptrs []string, err error, completed bool) {
	return nil, nil, false
}
