// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue8756

/*
#cgo !darwin LDFLAGS: -lm
#include <math.h>
*/
import "C"

func Pow() {
	C.pow(1, 2)
}
