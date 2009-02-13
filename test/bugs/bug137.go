// $G $D/$F.go || echo BUG should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
L1:
L2:	for i := 0; i < 10; i++ {
		print(i);
		break L2;
	}

L3: ;
L4:	for i := 0; i < 10; i++ {
		print(i);
		break L4;
	}
}

/*
bug137.go:9: break label is not defined: L2
bug137.go:15: break label is not defined: L4
*/
