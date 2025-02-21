// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package main

import _ "unsafe"

//go:linkname a main.c
func a() {}

func c() {}
