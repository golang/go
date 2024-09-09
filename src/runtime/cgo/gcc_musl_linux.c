// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

// x_cgo_is_musl reports whether the C library is musl.
int x_cgo_is_musl() {
	#if defined(__GLIBC__) || defined(__UCLIBC__)
		return 0;
	#else
		return 1;
	#endif
}
