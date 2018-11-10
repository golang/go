// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T1	// ERROR "unexpected newline in type declaration"

type T2 /* // ERROR "unexpected EOF in type declaration" */