// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// trivial malloc test

package main

import (
	"flag";
	"fmt";
	"malloc";
)

var chatty = flag.Bool("v", false, "chatty");

func main() {
	malloc.Free(malloc.Alloc(1));
	if *chatty {
		fmt.printf("%+v %v\n", *malloc.GetStats(), uint64(0));
	}
}

