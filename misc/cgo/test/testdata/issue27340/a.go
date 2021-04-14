// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Failed to resolve typedefs consistently.
// No runtime test; just make sure it compiles.
// In separate directory to isolate #pragma GCC diagnostic.

package issue27340

// We use the #pragma to avoid a compiler warning about incompatible
// pointer types, because we generate code passing a struct ptr rather
// than using the typedef. This warning is expected and does not break
// a normal build.
// We can only disable -Wincompatible-pointer-types starting with GCC 5.

// #if __GNU_MAJOR__ >= 5
//
// #pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
//
// typedef struct {
// 	int a;
// } issue27340Struct, *issue27340Ptr;
//
// static void issue27340CFunc(issue27340Ptr p) {}
//
// #else /* _GNU_MAJOR_ < 5 */
//
// typedef struct {
// 	int a;
// } issue27340Struct;
//
// static issue27340Struct* issue27340Ptr(issue27340Struct* p) { return p; }
//
// static void issue27340CFunc(issue27340Struct *p) {}
// #endif /* _GNU_MAJOR_ < 5 */
import "C"

func Issue27340GoFunc() {
	var s C.issue27340Struct
	C.issue27340CFunc(C.issue27340Ptr(&s))
}
