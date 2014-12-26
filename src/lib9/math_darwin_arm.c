// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Some Darwin/ARM libc versions fail to provide a standard compliant version
// of frexp and ldexp that could handle denormal floating point numbers.
// The frexp and ldexp implementations are translated from their Go version.

#include <stdint.h>

// Assume double and uint64_t are using the same endian.
union dint64 {
	double d;
	uint64_t u;
};

static const uint64_t mask = 0x7FF, bias = 1023;
static const int shift = 64 - 11 - 1;
static const uint64_t uvnan = 0x7FF8000000000001ULL, uvinf = 0x7FF0000000000000ULL,
	  uvneginf = 0xFFF0000000000000ULL;
static const double smallestnormal = 2.2250738585072014e-308; // 2**-1022

static inline uint64_t float64bits(double x) {
	union dint64 u;
	u.d = x;
	return u.u;
}
static inline double float64frombits(uint64_t x) {
	union dint64 u;
	u.u = x;
	return u.d;
}
static inline int isinf(double x) {
	return float64bits(x) == uvinf || float64bits(x) == uvneginf;
}
static inline int isnan(double x) {
	return x != x;
}
extern double fabs(double);
static double normalize(double x, int *exp) {
	if (fabs(x) < smallestnormal) {
		*exp = -52;
		return x * (double)(1LL<<52);
	}
	*exp = 0;
	return x;
}

double ldexp(double frac, int exp) {
	// special cases
	if (frac == 0.0) return frac;
	if (isinf(frac) || isnan(frac)) return frac;

	int e;
	frac = normalize(frac, &e);
	exp += e;
	uint64_t x = float64bits(frac);
	exp += (int)((x>>shift)&mask) - bias;
	if (exp < -1074) { // underflow
		if (frac < 0.0) return float64frombits(1ULL<<63); // -0.0
		return 0.0;
	}
	if (exp > 1023) { // overflow
		if (frac < 0.0) return float64frombits(uvneginf);
		return float64frombits(uvinf);
	}
	double m = 1;
	if (exp < -1022) { // denormal
		exp += 52;
		m = 1.0 / (double)(1ULL<<52);
	}
	x &= ~(mask << shift);
	x |= (uint64_t)(exp+bias) << shift;
	return m * float64frombits(x);
}

double frexp(double f, int *exp) {
	*exp = 0;
	// special cases
	if (f == 0.0) return f;
	if (isinf(f) || isnan(f)) return f;

	f = normalize(f, exp);
	uint64_t x = float64bits(f);
	*exp += (int)((x>>shift)&mask) - bias + 1;
	x &= ~(mask << shift);
	x |= (-1 + bias) << shift;
	return float64frombits(x);
}

// On Darwin/ARM, the kernel insists on running VFP in runfast mode, and it
// cannot deal with denormal floating point numbers in that mode, so we have
// to disable the runfast mode if the client uses ldexp/frexp (i.e. 5g).
void disable_vfp_runfast(void) __attribute__((constructor));
void disable_vfp_runfast(void) {
    __asm__ volatile (
		      "fmrx r0, fpscr\n"
		      "bic r0, r0, $0x03000000\n"
		      "fmxr fpscr, r0\n"
		      : : : "r0"
		     );
}
