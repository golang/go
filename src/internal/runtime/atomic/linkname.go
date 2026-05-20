// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic

import _ "unsafe" // for linkname

// Export some functions via linkname to assembly in sync/atomic.
//
//go:linknamestd Load
//go:linknamestd Loadp
//go:linknamestd Load64
//go:linknamestd Loaduintptr
//go:linknamestd Xadd
//go:linknamestd Xadd64
//go:linknamestd Xadduintptr
//go:linknamestd Xchg
//go:linknamestd Xchg64
//go:linknamestd Xchguintptr
//go:linknamestd Cas
//go:linknamestd Cas64
//go:linknamestd Casint32
//go:linknamestd Casint64
//go:linknamestd Casuintptr
//go:linknamestd Store
//go:linknamestd Store64
//go:linknamestd Storeuintptr
//go:linknamestd And32
//go:linknamestd And64
//go:linknamestd Anduintptr
//go:linknamestd Or32
//go:linknamestd Or64
//go:linknamestd Oruintptr
