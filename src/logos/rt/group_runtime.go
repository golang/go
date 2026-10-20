// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !logosejected

package rt

import _ "unsafe"

// Pull of the runtime group-tag primitive. The definition lives in
// src/runtime/logos_group.go and carries a push-style //go:linkname
// mark, so the linker's linkname check accepts this reference. The
// signature must match the runtime side exactly.

//go:linkname logosTagGroup runtime.logosSetGroup
func logosTagGroup(id uint64)

//go:linkname logosCancelGroup runtime.logosCancelGroup
func logosCancelGroup(id uint64)
