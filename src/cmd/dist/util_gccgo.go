// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gccgo

package main

/*
int supports_sse2() {
#if defined(__i386__) || defined(__x86_64__)
	return __builtin_cpu_supports("sse2");
#else
	return 0;
#endif
}
*/
import "C"

func cansse2() bool { return C.supports_sse2() != 0 }

func useVFPv1() {}

func useVFPv3() {}

func useARMv6K() {}
