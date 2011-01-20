// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type (
	Point struct {
		x, y float64
	}
	Polar Point
)

func main() {
}

/*
bug7.go:5: addtyp: renaming Point to Polar
main.go.c:14: error: redefinition of typedef ‘_T_2’
main.go.c:13: error: previous declaration of ‘_T_2’ was here
main.go.c:16: error: redefinition of ‘struct _T_2’
*/
