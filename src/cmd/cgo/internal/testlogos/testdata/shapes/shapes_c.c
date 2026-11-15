// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <errno.h>
#include <stdbool.h>
#include <string.h>

#include "_cgo_export.h"

int
shape_add_fast(int a, int b)
{
	return a + b;
}

int
shape_add_slow(int a, int b)
{
	return a + b;
}

unsigned long long
shape_u64_fast(unsigned long long a)
{
	return a * 2 + 1;
}

unsigned long long
shape_u64_slow(unsigned long long a)
{
	return a * 2 + 1;
}

float
shape_f32_fast(float a)
{
	return a + 1.5f;
}

float
shape_f32_slow(float a)
{
	return a + 1.5f;
}

double
shape_f64_fast(double a)
{
	return a * 2.0;
}

double
shape_f64_slow(double a)
{
	return a * 2.0;
}

bool
shape_bool_fast(bool b)
{
	return !b;
}

bool
shape_bool_slow(bool b)
{
	return !b;
}

void
shape_nop_fast(void)
{
}

void
shape_nop_slow(void)
{
}

void
shape_void_fast(int *p)
{
	*p = 42;
}

void
shape_void_slow(int *p)
{
	*p = 42;
}

int
shape_read_fast(int *p)
{
	return *p;
}

int
shape_read_slow(int *p)
{
	return *p;
}

int
shape_ppre_fast(int **pp)
{
	return **pp;
}

int
shape_ppre_slow(int **pp)
{
	return **pp;
}

int
shape_sum_fast(unsigned char *p, int n)
{
	int s = 0;
	int i;
	for (i = 0; i < n; i++) {
		s += p[i];
	}
	return s;
}

int
shape_sum_slow(unsigned char *p, int n)
{
	int s = 0;
	int i;
	for (i = 0; i < n; i++) {
		s += p[i];
	}
	return s;
}

int
shape_point_fast(struct ShapePoint p)
{
	return p.x + p.y;
}

int
shape_point_slow(struct ShapePoint p)
{
	return p.x + p.y;
}

int
shape_color_fast(enum ShapeColor c)
{
	return (int)c;
}

int
shape_color_slow(enum ShapeColor c)
{
	return (int)c;
}

int
shape_double(int x)
{
	return x * 2;
}

int
shape_apply_fast(shape_fn_t f, int x)
{
	return f(x);
}

int
shape_apply_slow(shape_fn_t f, int x)
{
	return f(x);
}

int
shape_errno_fast(int v)
{
	if (v < 0) {
		errno = 22;
		return -1;
	}
	errno = 0;
	return v * 2;
}

int
shape_errno_slow(int v)
{
	if (v < 0) {
		errno = 22;
		return -1;
	}
	errno = 0;
	return v * 2;
}

static int
shape_helper(int x)
{
	return x * 3 + 1;
}

int
shape_inline_fast(int x)
{
	return shape_helper(x);
}

int
shape_inline_slow(int x)
{
	return shape_helper(x);
}

int
cgoCheckPointer(int x)
{
	return x + 1;
}

int
cgocall(int x)
{
	return x + 2;
}

int
shape_strlen_fast(char *s)
{
	return strlen(s);
}

int
shape_strlen_slow(char *s)
{
	return strlen(s);
}

int
shape_vp_fast(void *p)
{
	return *(int *)p;
}

int
shape_vp_slow(void *p)
{
	return *(int *)p;
}

int
shape_cb_fast(void)
{
	return GoCallback(41);
}

int
shape_cb_plain(void)
{
	return GoCallback(7);
}
