// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"flag"
)

// LogosEject is the -logos.eject output directory. After checkFiles
// succeeds, the compiler writes one ejected Go file per logos source by
// applying the lowering edits to the original bytes. Empty means no eject.
var LogosEject string

// logosRegisterFlags registers the loGo/s compiler flags.
func logosRegisterFlags() {
	flag.StringVar(&LogosEject, "logos.eject", "", "write ejected Go sources to `directory`")
}
