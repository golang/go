// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (386 || amd64 || amd64p32) && gccgo

#include <cpuid.h>
#include <stdint.h>
#include <x86intrin.h>

// Need to wrap __get_cpuid_count because it's declared as static.
int
gccgoGetCpuidCount(uint32_t leaf, uint32_t subleaf,
                   uint32_t *eax, uint32_t *ebx,
                   uint32_t *ecx, uint32_t *edx)
{
	return __get_cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
}

#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC push_options
#pragma GCC target("xsave")
#pragma clang attribute push (__attribute__((target("xsave"))), apply_to=function)

// xgetbv reads the contents of an XCR (Extended Control Register)
// specified in the ECX register into registers EDX:EAX.
// Currently, the only supported value for XCR is 0.
void
gccgoXgetbv(uint32_t *eax, uint32_t *edx)
{
	uint64_t v = _xgetbv(0);
	*eax = v & 0xffffffff;
	*edx = v >> 32;
}

#pragma clang attribute pop
#pragma GCC pop_options
