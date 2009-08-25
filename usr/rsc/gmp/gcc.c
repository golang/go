// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <gmp.h>
#include <string.h>

typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;

typedef struct Slice Slice;
struct Slice
{
	void *data;
	uint32 len;
	uint32 cap;
};

typedef struct String String;
struct String
{
	void *data;
	uint32 len;
};

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
gmp_subInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *y;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_sub(*a->z->mp, *a->x->mp, *a->y->mp);
}

void
gmp_mulInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *y;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_mul(*a->z->mp, *a->x->mp, *a->y->mp);
}

void
gmp_setInt64Int(void *v)
{
	struct {
		Int *z;
		int64 x;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_set_si(*a->z->mp, a->x);
}

void
gmp_int64Int(void *v)
{
	struct {
		Int *z;
		int64 ret;
	} *a = v;

	a->ret = mpz_get_si(*a->z->mp);
}

void
gmp_divInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *y;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_div(*a->z->mp, *a->x->mp, *a->y->mp);
}

void
gmp_modInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *y;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_mod(*a->z->mp, *a->x->mp, *a->y->mp);
}

void
gmp_divModInt(void *v)
{
	struct {
		Int *d;
		Int *m;
		Int *x;
		Int *y;
	} *a = v;

	mpz_tdiv_qr(*a->d->mp, *a->m->mp, *a->x->mp, *a->y->mp);
}

void
gmp_lshInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		uint32 s;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_mul_2exp(*a->z->mp, *a->x->mp, a->s);
}

void
gmp_rshInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		int32 s;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_div_2exp(*a->z->mp, *a->x->mp, a->s);
}


void
gmp_expInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *y;
		Int *w;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_powm(*a->z->mp, *a->x->mp, *a->y->mp, *a->w->mp);
}

void
gmp_gcdInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *y;
		Int *a;
		Int *b;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_gcdext(*a->z->mp, *a->x->mp, *a->y->mp, *a->a->mp, *a->b->mp);
}

void
gmp_negInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_neg(*a->z->mp, *a->x->mp);
}

void
gmp_absInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_abs(*a->z->mp, *a->x->mp);
}

void
gmp_cmpInt(void *v)
{
	struct {
		Int *x;
		Int *y;
		int32 ret;
	} *a = v;

	a->ret = mpz_cmp(*a->x->mp, *a->y->mp);
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

void
gmp_setInt(void *v)
{
	struct {
		Int *z;
		Int *x;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_set(*a->z->mp, *a->x->mp);
}

void
gmp_setBytesInt(void *v)
{
	struct {
		Int *z;
		Slice b;
		Int *ret;
	} *a = v;

	a->ret = a->z;
	mpz_import(*a->z->mp, a->b.len, 1, 1, 1, 0, a->b.data);
}

void
gmp_lenInt(void *v)
{
	struct {
		Int *z;
		int32 ret;
	} *a = v;

	a->ret = mpz_sizeinbase(*a->z->mp, 2);
}

void
gmp_bytesInt(void *v)
{
	struct {
		Int *z;
		Slice b;
	} *a = v;
	size_t n;
	char *p;

	n = (mpz_sizeinbase(*a->z->mp, 2) + 7) >> 3;
	p = malloc(n);	// TODO: mallocgc
	mpz_export(p, &n, 1, 1, 1, 0, *a->z->mp);
	a->b.data = p;
	a->b.len = n;
	a->b.cap = n;
}

void
gmp_setStringInt(void *v)
{
	struct {
		Int *z;
		String s;
		int32 base;
		int32 pad;
		int32 ret;
	} *a = v;
	char *p;

	p = malloc(a->s.len+1);
	memmove(p, a->s.data, a->s.len);
	p[a->s.len] = 0;
	a->ret = mpz_set_str(*a->z->mp, p, a->base);
	free(p);
}

void
gmp_probablyPrimeInt(void *v)
{
	struct {
		Int *z;
		int32 nreps;
		int32 pad;
		int32 ret;
	} *a = v;

	a->ret = mpz_probab_prime_p(*a->z->mp, a->nreps);
}

