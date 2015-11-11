// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gccgo

#include <errno.h>
#include <stdint.h>
#include <unistd.h>

#define _STRINGIFY2_(x) #x
#define _STRINGIFY_(x) _STRINGIFY2_(x)
#define GOSYM_PREFIX _STRINGIFY_(__USER_LABEL_PREFIX__)

extern intptr_t popcount(uintptr_t x) __asm__(GOSYM_PREFIX GOPKGPATH ".popcount");

intptr_t popcount(uintptr_t x) {
	return __builtin_popcountl((unsigned long)(x));
}
