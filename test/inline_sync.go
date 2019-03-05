// +build !nacl,!386,!wasm,!arm,!gcflags_noopt
// errorcheck -0 -m

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that inlining of functions
// imported from the sync package is working.
// Compiles but does not run.
// FIXME: nacl-386 is excluded as inlining currently does not work there.

package foo

import (
	"sync"
)

var mutex *sync.Mutex

func small5() { // ERROR "can inline small5"
	// the Unlock fast path should be inlined
	mutex.Unlock() // ERROR "inlining call to sync\.\(\*Mutex\)\.Unlock" "&sync\.m\.state escapes to heap"
}
