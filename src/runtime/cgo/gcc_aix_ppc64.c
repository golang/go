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
extern void __attribute__((longcall)) _rt0_ppc64_aix_lib(void);

int _cgo_topofstack(void) {
	return __cgo_topofstack();
}

int main(int argc, char **argv) {
	return runtime_rt0_go(argc, argv);
}

static void libinit(void) __attribute__ ((constructor));

/*
 * libinit aims to replace .init_array section which isn't available on aix.
 * Using __attribute__ ((constructor)) let gcc handles this instead of
 * adding special code in cmd/link.
 * However, it will be called for every Go programs which has cgo.
 * Inside _rt0_ppc64_aix_lib(), runtime.isarchive is checked in order
 * to know if this program is a c-archive or a simple cgo program.
 * If it's not set, _rt0_ppc64_ax_lib() returns directly.
 */
static void libinit() {
	_rt0_ppc64_aix_lib();
}
