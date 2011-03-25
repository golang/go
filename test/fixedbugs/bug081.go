// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const x x = 2 // ERROR "loop|type"

/*
bug081.go:3: first constant must evaluate an expression
Bus error
*/
