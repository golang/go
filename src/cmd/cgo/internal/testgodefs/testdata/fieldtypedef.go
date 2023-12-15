// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package main

/*
struct S1 { int f1; };
struct S2 { struct S1 s1; };
typedef struct S1 S1Type;
typedef struct S2 S2Type;
*/
import "C"

type S1 C.S1Type
type S2 C.S2Type
