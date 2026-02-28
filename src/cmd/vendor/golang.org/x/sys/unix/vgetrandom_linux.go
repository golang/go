// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && go1.24

package unix

import _ "unsafe"

//go:linkname vgetrandom runtime.vgetrandom
//go:noescape
func vgetrandom(p []byte, flags uint32) (ret int, supported bool)
