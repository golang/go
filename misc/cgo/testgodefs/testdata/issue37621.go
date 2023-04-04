// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package main

/*
struct tt {
	long long a;
	long long b;
};

struct s {
	struct tt ts[3];
};
*/
import "C"

type TT C.struct_tt

type S C.struct_s
