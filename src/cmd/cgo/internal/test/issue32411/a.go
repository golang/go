// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue32411

// #cgo CFLAGS: -Werror
// #include <stdlib.h>
import "C"

// Referring to a C function without calling it makes cgo declare it
// in _cgo_main.c without a prototype.
var Free = (*[0]byte)(C.free)
