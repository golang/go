// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (ppc64 || ppc64le) && !purego

package sha512

import (
	"crypto/internal/impl"
	"internal/godebug"
)

// The POWER architecture doesn't have a way to turn off SHA-512 support at
// runtime with GODEBUG=cpu.something=off, so introduce a new GODEBUG knob for
// that. It's intentionally only checked at init() time, to avoid the
// performance overhead of checking it on every block.
var ppc64sha512 = godebug.New("#ppc64sha512").Value() != "off"

func init() {
	impl.Register("sha512", "POWER8", &ppc64sha512)
}

//go:noescape
func blockPOWER(dig *Digest, p []byte)

func block(dig *Digest, p []byte) {
	if ppc64sha512 {
		blockPOWER(dig, p)
	} else {
		blockGeneric(dig, p)
	}
}
