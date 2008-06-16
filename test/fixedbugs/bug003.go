// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	switch ; { case true: return; default: return }
}
/*
bug003.go:6: fatal error: walkswitch: not case EMPTY
*/
