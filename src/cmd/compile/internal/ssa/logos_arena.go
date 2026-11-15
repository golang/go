// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "internal/goarch"

const logosArenaGOffset = 4 * goarch.PtrSize
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
