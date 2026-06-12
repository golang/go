// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips || mipsle

package runtime

import _ "unsafe" // for linkname

// Called from assembly only; declared for go vet.
//
// load_g is also called from runtime/cgo.
//
// load_g should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/ebitengine/purego
//
//go:linkname load_g
func load_g()
func save_g()

// getfp returns the frame pointer register of its caller or 0 if not implemented.
// TODO: Make this a compiler intrinsic
//
//go:nosplit
func getfp() uintptr { return 0 }
