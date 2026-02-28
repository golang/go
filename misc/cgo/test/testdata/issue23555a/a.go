// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue23555

// #include <stdlib.h>
import "C"

func X() {
	C.free(C.malloc(10))
}
