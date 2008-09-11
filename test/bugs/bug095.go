// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG wrong result

package main

func main() {
   var x int = 1;
   if x != 1 { panic("found ", x, ", expected 1\n"); }
   {
	   var x int = x + 1;  // scope of x starts too late
	   if x != 1 { panic("found ", x, ", expected 1\n"); }
   }
   {
	   x := x + 1;  // scope of x starts too late
	   if x != 1 { panic("found ", x, ", expected 1\n"); }
   }
}

/*
uetli:~/Source/go1/test/bugs gri$ 6g bug095.go && 6l bug095.6 && 6.out
found 2, expected 1

panic on line 342 PC=0x139e
0x139e?zi
	main·main(1, 0, 1606416416, ...)
	main·main(0x1, 0x7fff5fbff820, 0x0, ...)
Trace/BPT trap
*/

/*
Example: If I write

type Tree struct {
	left, right *Tree
}

I expect the correct *Tree to picked up; i.e. the scope of the identifier
Tree starts immediately after the name is declared. There is no reason why
this should be different for vars.
*/
