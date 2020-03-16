// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// typedef struct S ST;
// static ST* F() { return 0; }
import "C"

func F1() {
	C.F()
}
