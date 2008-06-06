// ! $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var i int = 100;
	i = i << -3;  // BUG: should not compile (negative shift)
}

/*
bug016.go:7: fatal error: optoas: no entry LSH-<int32>INT32
*/
