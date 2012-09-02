// skip

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package rand

/*
#include <stdlib.h>
*/
import "C"

// STOP OMIT
func Random() int {
	return int(C.random())
}

// STOP OMIT
func Seed(i int) {
	C.srandom(C.uint(i))
}

// END OMIT
