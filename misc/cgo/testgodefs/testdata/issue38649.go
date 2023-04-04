// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package main

/*
struct Issue38649 { int x; };
#define issue38649 struct Issue38649
*/
import "C"

type issue38649 C.issue38649
