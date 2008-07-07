// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go && $L $F.$A && ./$A.out

// Forward declarations

package main

func f (x int) ;  // this works
func f (x int) {}

func i (x, y int) ;  // this works
func i (x, y int) {}

func g (x int) float ;  // BUG this doesn't
func g (x int) float { return 0.0 }

func h (x int) (u int, v int) ;  // BUG this doesn't
func h (x int) (u int, v int) {}

func main() {}
