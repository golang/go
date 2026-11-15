// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !logosejected

package rt

import _ "unsafe"

//go:linkname logosGroupAcquire runtime.logosGroupAcquire
func logosGroupAcquire() uint32

//go:linkname logosGroupRelease runtime.logosGroupRelease
func logosGroupRelease(slot uint32)

//go:linkname logosTagGroup runtime.logosSetGroup
func logosTagGroup(slot uint32)

//go:linkname logosCancelGroup runtime.logosCancelGroup
func logosCancelGroup(slot uint32)
