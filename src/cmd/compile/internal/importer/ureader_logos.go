// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"cmd/compile/internal/types2"
)

// LogosOverlayEnabled gates the option overlay for direct importer
// users. Noder applies its own overlay after ReadPackage and leaves
// this false, so this hook stays a no-op there; direct users set it
// before ReadPackage to request the .lgo view.
var LogosOverlayEnabled bool

// logosOverlayHook runs after ReadPackage. The .lgo overlay lives in
// noder, which owns the import cache and the facts lookup; this hook
// only consumes the enable flag so direct importer users keep a hook
// point without pulling new dependencies here.
func logosOverlayHook(pkg *types2.Package) *types2.Package {
	LogosOverlayEnabled = false
	return pkg
}
