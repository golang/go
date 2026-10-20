// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !logosejected

package rt

import _ "unsafe"

// Pull of the runtime group-slot primitives. The definitions live in
// src/runtime/logos_group.go and logos_group_netpoll.go and carry a
// push-style //go:linkname mark, so the linker's linkname check accepts
// these references. The signatures must match the runtime side exactly.

//go:linkname logosGroupAcquire runtime.logosGroupAcquire
func logosGroupAcquire() uint32

//go:linkname logosGroupRelease runtime.logosGroupRelease
func logosGroupRelease(slot uint32)

//go:linkname logosTagGroup runtime.logosSetGroup
func logosTagGroup(slot uint32)

//go:linkname logosCancelGroup runtime.logosCancelGroup
func logosCancelGroup(slot uint32)
