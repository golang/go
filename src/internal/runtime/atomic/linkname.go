// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic

import _ "unsafe" // for linkname

// Export some functions via linkname to assembly in sync/atomic.
//
//go:linkname Load
//go:linkname Loadp
//go:linkname Load64
//go:linkname Loaduintptr
//go:linkname Xadd
//go:linkname Xadd64
//go:linkname Xadduintptr
//go:linkname Xchg
//go:linkname Xchg64
//go:linkname Xchguintptr
//go:linkname Cas
//go:linkname Cas64
//go:linkname Casint32
//go:linkname Casint64
//go:linkname Casuintptr
//go:linkname Store
//go:linkname Store64
//go:linkname Storeuintptr
//go:linkname And32
//go:linkname And64
//go:linkname Anduintptr
//go:linkname Or32
//go:linkname Or64
//go:linkname Oruintptr
