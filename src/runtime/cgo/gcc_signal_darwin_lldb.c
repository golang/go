// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !lldb
// +build darwin
// +build arm arm64

#include <stdint.h>

uintptr_t x_cgo_panicmem;

void darwin_arm_init_thread_exception_port() {}
void darwin_arm_init_mach_exception_handler() {}
