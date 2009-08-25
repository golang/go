// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>

typedef int32_t int32;

static int32
fib1(int32 n)
{
	int32 a, b, t;

	a = 0;
	b = 1;
	for(; n>0; n--) {
		t = a;
		a = b;
		b += t;
	}
	return a;
}

void
fib(void *v)
{
	struct {	// 6g func(n int) int
		int32 n;
		int32 pad;
		int32 ret;
	} *args = v;

	args->ret = fib1(args->n);
}
