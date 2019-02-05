// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is here just to cause problems.
// file.swig turns into a file also named file.go.
// Make sure cmd/go keeps them separate
// when both are passed to cgo.

package file

//int F(void) { return 1; }
import "C"

func F() int { return int(C.F()) }
