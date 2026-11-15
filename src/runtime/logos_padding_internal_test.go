// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/stringslite"
	"unsafe"
)

var (
	LogosMFrameSize          = unsafe.Sizeof(m{}.fastCgoFrame)
	LogosMFlagSize           = unsafe.Sizeof(m{}.inFastCall)
	LogosMFrameOff           = unsafe.Offsetof(m{}.fastCgoFrame)
	LogosMSize               = unsafe.Sizeof(m{})
	LogosGArenaSize          = unsafe.Sizeof(g{}.logosArena)
	LogosGRetOKSize          = unsafe.Sizeof(g{}.logosArenaRetOK)
	LogosGRegionsSize        = unsafe.Sizeof(g{}.logosRegions)
	LogosGFreeSize           = unsafe.Sizeof(g{}.logosRegionFree)
	LogosGSiteSize           = unsafe.Sizeof(g{}.logosErrSite)
	LogosGGroupSize          = unsafe.Sizeof(g{}.logosGroup)
	LogosGLastSize           = unsafe.Sizeof(g{}.logosErrLast)
	LogosGLastTypSize        = unsafe.Sizeof(g{}.logosErrLastTyp)
	LogosGArenaOff           = unsafe.Offsetof(g{}.logosArena)
	LogosGRetOKOff           = unsafe.Offsetof(g{}.logosArenaRetOK)
	LogosGRegionsOff         = unsafe.Offsetof(g{}.logosRegions)
	LogosGFreeOff            = unsafe.Offsetof(g{}.logosRegionFree)
	LogosGSiteOff            = unsafe.Offsetof(g{}.logosErrSite)
	LogosGGroupOff           = unsafe.Offsetof(g{}.logosGroup)
	LogosGLastOff            = unsafe.Offsetof(g{}.logosErrLast)
	LogosGLastTypOff         = unsafe.Offsetof(g{}.logosErrLastTyp)
	LogosGSize               = unsafe.Sizeof(g{})
	LogosPPoolSize           = unsafe.Sizeof(p{}.logosArenaPool)
	LogosPNSize              = unsafe.Sizeof(p{}.logosArenaPoolN)
	LogosPRegionSize         = unsafe.Sizeof(p{}.logosRegionPool)
	LogosPPoolOff            = unsafe.Offsetof(p{}.logosArenaPool)
	LogosPNOff               = unsafe.Offsetof(p{}.logosArenaPoolN)
	LogosPRegionOff          = unsafe.Offsetof(p{}.logosRegionPool)
	LogosPSize               = unsafe.Sizeof(p{})
	LogosGAlign              = unsafe.Alignof(g{})
	LogosMAlign              = unsafe.Alignof(m{})
	LogosPAlign              = unsafe.Alignof(p{})
	LogosSpanNeedzeroSize    = unsafe.Sizeof(mspan{}.needzero)
	LogosSpanIsChunkSize     = unsafe.Sizeof(mspan{}.isLogosArenaChunk)
	LogosSpanBeforeCacheSize = unsafe.Sizeof(mspan{}.allocCountBeforeCache)
	LogosSpanNeedzeroOff     = unsafe.Offsetof(mspan{}.needzero)
	LogosSpanIsChunkOff      = unsafe.Offsetof(mspan{}.isLogosArenaChunk)
	LogosSpanBeforeCacheOff  = unsafe.Offsetof(mspan{}.allocCountBeforeCache)
)

var (
	LogosGUpstreamAlign = logosUpstreamAlign(abi.TypeFor[g]())
	LogosMUpstreamAlign = logosUpstreamAlign(abi.TypeFor[m](), "inFastCall", "fastCgoFrame")
	LogosPUpstreamAlign = logosUpstreamAlign(abi.TypeFor[p]())
)

func logosUpstreamAlign(t *abi.Type, forkNames ...string) uintptr {
	align := 1
	for _, f := range t.StructType().Fields {
		if !logosForkField(f.Name.Name(), forkNames) {
			align = max(align, f.Typ.Align())
		}
	}
	return uintptr(align)
}

func logosForkField(name string, forkNames []string) bool {
	if stringslite.HasPrefix(name, "logos") {
		return true
	}
	for _, n := range forkNames {
		if name == n {
			return true
		}
	}
	return false
}
