// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	count := 0;
	if one := 1; {  // BUG if there is a simple stat, the condition must be present
		count = count + one;	
	}
}
