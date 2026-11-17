// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "internal/goarch"

// Offset of runtime.g.logosArena: stack(2P) + stackguard0(P) + stackguard1(P)
// + _panic(P) + _defer(P) + m(P) + sched gobuf(6P). The arena block sits
// after sched so upstream prefix offsets are preserved.
const logosArenaGOffset = 13 * goarch.PtrSize
const LogosArenaGOffset = logosArenaGOffset

const (
	logosRegionChunkOffset = 0
	logosRegionGenOffset   = goarch.PtrSize
	LogosRegionChunkOffset = logosRegionChunkOffset
	LogosRegionGenOffset   = logosRegionGenOffset
	logosRegionOwnerOffset = 3 * goarch.PtrSize
	LogosRegionOwnerOffset = logosRegionOwnerOffset

	logosRegionNoscanCurOffset = 2 * goarch.PtrSize
	logosRegionScanCurOffset   = 3 * goarch.PtrSize
	LogosRegionNoscanCurOffset = logosRegionNoscanCurOffset
	LogosRegionScanCurOffset   = logosRegionScanCurOffset
)
