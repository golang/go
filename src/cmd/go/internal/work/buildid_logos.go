// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"slices"
	"strings"
)

// logosReportsBuildID reports whether the -V=full fields f end with a
// build ID. The loGos tools print one under the release version go1.27.2
// (cmd/internal/objabi forkReportsBuildID), so the go command keys them
// on the content ID, as it keys development builds. The whole line would
// also carry the action ID, which differs between toolchain stages that
// build byte-identical tools, so dist would report false stale targets.
func logosReportsBuildID(f []string) bool {
	return strings.HasPrefix(f[len(f)-1], "buildID=")
}

// logosEjectNoCache reports whether action a compiles a package marked
// for logos eject, or a package whose -gcflags carry -logos.eject=DIR
// (the language server proxy passes it per package). Such packages skip
// the build cache like -a builds, so the compiler always runs and writes
// fresh -logos.eject output. It is false for every other action, keeping
// plain go and ordinary logos builds on the cache.
func logosEjectNoCache(a *Action) bool {
	if a == nil || a.Package == nil {
		return false
	}
	return a.Package.Internal.LogosEject || slices.ContainsFunc(a.Package.Internal.Gcflags, logosEjectFlag)
}

// logosEjectFlag reports whether the compiler flag f is -logos.eject=DIR.
func logosEjectFlag(f string) bool {
	return strings.HasPrefix(f, "-logos.eject=")
}

// logosNextNoCache reports whether action a compiles a package marked
// for logos next. Scanned packages skip the build cache like -a builds,
// so the compiler always runs and appends fresh -logos.nextout output.
func logosNextNoCache(a *Action) bool {
	return a != nil && a.Package != nil && a.Package.Internal.LogosNext
}
