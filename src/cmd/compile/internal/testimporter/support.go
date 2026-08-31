// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(mark): Remove fork in cmd/compile/internal/importer.

package testimporter

import (
	"cmd/compile/internal/base"
	"internal/pkgbits"
)

func assert(p bool) {
	base.Assert(p)
}

// See cmd/compile/internal/noder.derivedInfo.
type derivedInfo struct {
	idx    pkgbits.Index
	needed bool
}

// See cmd/compile/internal/noder.typeInfo.
type typeInfo struct {
	idx     pkgbits.Index
	derived bool
}
