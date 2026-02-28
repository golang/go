// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sysrand

import "syscall"

func read(b []byte) error {
	// This uses the wasi_snapshot_preview1 random_get syscall defined in
	// https://github.com/WebAssembly/WASI/blob/23a52736049f4327dd335434851d5dc40ab7cad1/legacy/preview1/docs.md#-random_getbuf-pointeru8-buf_len-size---result-errno.
	// The definition does not explicitly guarantee that the entire buffer will
	// be filled, but this appears to be the case in all runtimes tested.
	return syscall.RandomGet(b)
}
