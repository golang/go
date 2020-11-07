// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 amd64 amd64p32
// +build gccgo

#include <cpuid.h>
#include <stdint.h>

// Need to wrap __get_cpuid_count because it's declared as static.
int
gccgoGetCpuidCount(uint32_t leaf, uint32_t subleaf,
                   uint32_t *eax, uint32_t *ebx,
                   uint32_t *ecx, uint32_t *edx)
{
	return __get_cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
}

// xgetbv reads the contents of an XCR (Extended Control Register)
// specified in the ECX register into registers EDX:EAX.
// Currently, the only supported value for XCR is 0.
//
// TODO: Replace with a better alternative:
//
//     #include <xsaveintrin.h>
//
//     #pragma GCC target("xsave")
//
//     void gccgoXgetbv(uint32_t *eax, uint32_t *edx) {
//       unsigned long long x = _xgetbv(0);
//       *eax = x & 0xffffffff;
//       *edx = (x >> 32) & 0xffffffff;
//     }
//
// Note that _xgetbv is defined starting with GCC 8.
void
gccgoXgetbv(uint32_t *eax, uint32_t *edx)
{
	__asm("  xorl %%ecx, %%ecx\n"
	      "  xgetbv"
	    : "=a"(*eax), "=d"(*edx));
}
