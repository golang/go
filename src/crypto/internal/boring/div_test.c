// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is a self-contained test for a copy of
// the division algorithm in build-goboring.sh,
// to verify that is correct. The real algorithm uses u128
// but this copy uses u32 for easier testing.
// s/32/128/g should be the only difference between the two.
//
// This is the dumbest possible division algorithm,
// but any crypto code that depends on the speed of
// division is equally dumb.

//go:build ignore

#include <stdio.h>
#include <stdint.h>

#define nelem(x) (sizeof(x)/sizeof((x)[0]))

typedef uint32_t u32;

static u32 div(u32 x, u32 y, u32 *rp) {
	int n = 0;
	while((y>>(32-1)) != 1 && y < x) {
		y<<=1;
		n++;
	}
	u32 q = 0;
	for(;; n--, y>>=1, q<<=1) {
		if(x>=y) {
			x -= y;
			q |= 1;
		}
		if(n == 0)
			break;
	}
	if(rp)
		*rp = x;
	return q;
}

u32 tests[] = {
	0,
	1,
	2,
	3,
	4,
	5,
	6,
	7,
	8,
	9,
	10,
	11,
	31,
	0xFFF,
	0x1000,
	0x1001,
	0xF0F0F0,
	0xFFFFFF,
	0x1000000,
	0xF0F0F0F0,
	0xFFFFFFFF,
};

int
main(void)
{
	for(int i=0; i<nelem(tests); i++)
	for(int j=0; j<nelem(tests); j++) {
		u32 n = tests[i];
		u32 d = tests[j];
		if(d == 0)
			continue;
		u32 r;
		u32 q = div(n, d, &r);
		if(q != n/d || r != n%d)
			printf("div(%x, %x) = %x, %x, want %x, %x\n", n, d, q, r, n/d, n%d);
	}
	return 0;
}
