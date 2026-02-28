// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || dragonfly || freebsd || linux || netbsd || solaris

package cgosotest

/*
extern int __thread tlsvar;
int *getTLS() { return &tlsvar; }
*/
import "C"

func init() {
	if v := *C.getTLS(); v != 12345 {
		println("got", v)
		panic("BAD TLS value")
	}
}
