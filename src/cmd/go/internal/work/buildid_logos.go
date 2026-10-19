// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

// logosEjectNoCache reports whether action a compiles a package marked
// for logos eject. Ejected packages skip the build cache like -a builds,
// so the compiler always runs and writes fresh -logos.eject output.
// It is false for every other action, keeping plain go and ordinary
// logos builds on the cache.
func logosEjectNoCache(a *Action) bool {
	return a != nil && a.Package != nil && a.Package.Internal.LogosEject
}
