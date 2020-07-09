// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// +build ignore

package main

// enum { ENUMVAL = 0x1 };
import "C"

const ENUMVAL = C.ENUMVAL
