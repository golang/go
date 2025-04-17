// errorcheck -0 -lang=go1.17

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Prior to Go 1.18, ineffectual //go:linkname directives were treated
// as noops. Ensure that modules that contain these directives (e.g.,
// x/sys prior to go.dev/cl/274573) continue to compile.

package p

import _ "unsafe"

//go:linkname nonexistent nonexistent

//go:linkname constant constant
const constant = 42

//go:linkname typename typename
type typename int
