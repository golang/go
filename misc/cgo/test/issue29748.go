// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Error handling a struct initializer that requires pointer checking.
// Compilation test only, nothing to run.

package cgotest

// typedef struct { char **p; } S29748;
// static int f29748(S29748 *p) { return 0; }
import "C"

var Vissue29748 = C.f29748(&C.S29748{
	nil,
})

func Fissue299748() {
	C.f29748(&C.S29748{
		nil,
	})
}
