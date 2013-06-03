// skip

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package rand2

/*
#include <stdlib.h>
*/
import "C"

func Random() int {
	var r C.int = C.rand()
	return int(r)
}

// STOP OMIT
func Seed(i int) {
	C.srand(C.uint(i))
}

// END OMIT
