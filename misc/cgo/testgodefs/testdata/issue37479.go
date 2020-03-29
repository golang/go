// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// +build ignore

package main

/*
typedef struct A A;

typedef struct {
	struct A *next;
	struct A **prev;
} N;

struct A
{
	N n;
};

typedef struct B
{
	A* a;
} B;
*/
import "C"

type N C.N

type A C.A

type B C.B
