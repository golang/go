// $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type	I1	interface {}
type	I2	interface { pr() }

func	e()	I1;

var	i1	I1;
var	i2	I2;

func
main() {

	i2 = e().(I2);	// bug089.go:16: fatal error: agen_inter i2i
}
