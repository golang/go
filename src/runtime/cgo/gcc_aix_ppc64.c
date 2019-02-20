// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix
// +build ppc64 ppc64le

/*
 * On AIX, call to _cgo_topofstack and Go main are forced to be a longcall.
 * Without it, ld might add trampolines in the middle of .text section
 * to reach these functions which are normally declared in runtime package.
 */
extern int __attribute__((longcall)) __cgo_topofstack(void);
extern int __attribute__((longcall)) runtime_rt0_go(int argc, char **argv);

int _cgo_topofstack(void) {
	return __cgo_topofstack();
}

int main(int argc, char **argv) {
	return runtime_rt0_go(argc, argv);
}
