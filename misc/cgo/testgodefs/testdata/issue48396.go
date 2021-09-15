// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// +build ignore

package main

/*
// from <linux/kcm.h>
struct issue48396 {
	int fd;
	int bpf_fd;
};
*/
import "C"

type Issue48396 C.struct_issue48396
