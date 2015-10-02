// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !cgo netgo

package net

func init() { netGo = true }

type addrinfoErrno int

func (eai addrinfoErrno) Error() string   { return "<nil>" }
func (eai addrinfoErrno) Temporary() bool { return false }
func (eai addrinfoErrno) Timeout() bool   { return false }

func cgoLookupHost(name string) (addrs []string, err error, completed bool) {
	return nil, nil, false
}

func cgoLookupPort(network, service string) (port int, err error, completed bool) {
	return 0, nil, false
}

func cgoLookupIP(name string) (addrs []IPAddr, err error, completed bool) {
	return nil, nil, false
}

func cgoLookupCNAME(name string) (cname string, err error, completed bool) {
	return "", nil, false
}

func cgoLookupPTR(addr string) (ptrs []string, err error, completed bool) {
	return nil, nil, false
}
