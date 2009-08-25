// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <gmp.h>
#include <string.h>

typedef int32_t int32;
typedef uint64_t uint64;

typedef struct Int Int;
struct Int
{
	mpz_t *mp;
};

void
gmp_newInt(void *v)
{
	struct {
		uint64 x;
		Int *z;
	} *a = v;

	a->z->mp = malloc(sizeof *a->z->mp);
	mpz_init_set_ui(*a->z->mp, a->x);
}

void
gmp_addInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *y;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_add(*a->z->mp, *a->x->mp, *a->y->mp);
}

void
gmp_stringInt(void *v)
{
	struct {
		Int *z;
		char *p;
	} *a = v;

	a->p = mpz_get_str(NULL, 10, *a->z->mp);
}

