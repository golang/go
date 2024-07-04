// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

#include "libcgo.h"

#include <stdlib.h>

/* Stub for calling clearenv */
void
x_cgo_clearenv(void **_unused)
{
	_cgo_tsan_acquire();
	clearenv();
	_cgo_tsan_release();
}
