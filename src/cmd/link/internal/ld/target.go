// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"encoding/binary"
)

// Target holds the configuration we're building for.
type Target struct {
	Arch *sys.Arch

	HeadType objabi.HeadType

	LinkMode  LinkMode
	BuildMode BuildMode

	linkShared    bool
	canUsePlugins bool
	IsELF         bool
}

//
// Target type functions
//

func (t *Target) IsExe() bool {
	return t.BuildMode == BuildModeExe
}

func (t *Target) IsShared() bool {
	return t.BuildMode == BuildModeShared
}

func (t *Target) IsPlugin() bool {
	return t.BuildMode == BuildModePlugin
}

func (t *Target) IsInternal() bool {
	return t.LinkMode == LinkInternal
}

func (t *Target) IsExternal() bool {
	return t.LinkMode == LinkExternal
}

func (t *Target) IsPIE() bool {
	return t.BuildMode == BuildModePIE
}

func (t *Target) IsSharedGoLink() bool {
	return t.linkShared
}

func (t *Target) CanUsePlugins() bool {
	return t.canUsePlugins
}

func (t *Target) IsElf() bool {
	t.mustSetHeadType()
	return t.IsELF
}

func (t *Target) IsDynlinkingGo() bool {
	return t.IsShared() || t.IsSharedGoLink() || t.IsPlugin() || t.CanUsePlugins()
}

// UseRelro reports whether to make use of "read only relocations" aka
// relro.
func (t *Target) UseRelro() bool {
	switch t.BuildMode {
	case BuildModeCArchive, BuildModeCShared, BuildModeShared, BuildModePIE, BuildModePlugin:
		return t.IsELF || t.HeadType == objabi.Haix || t.HeadType == objabi.Hdarwin
	default:
		if t.HeadType == objabi.Hdarwin && t.IsARM64() {
			// On darwin/ARM64, everything is PIE.
			return true
		}
		return t.linkShared || (t.HeadType == objabi.Haix && t.LinkMode == LinkExternal)
	}
}

//
// Processor functions
//

func (t *Target) Is386() bool {
	return t.Arch.Family == sys.I386
}

func (t *Target) IsARM() bool {
	return t.Arch.Family == sys.ARM
}

func (t *Target) IsARM64() bool {
	return t.Arch.Family == sys.ARM64
}

func (t *Target) IsAMD64() bool {
	return t.Arch.Family == sys.AMD64
}

func (t *Target) IsMIPS() bool {
	return t.Arch.Family == sys.MIPS
}

func (t *Target) IsMIPS64() bool {
	return t.Arch.Family == sys.MIPS64
}

func (t *Target) IsPPC64() bool {
	return t.Arch.Family == sys.PPC64
}

func (t *Target) IsRISCV64() bool {
	return t.Arch.Family == sys.RISCV64
}

func (t *Target) IsS390X() bool {
	return t.Arch.Family == sys.S390X
}

func (t *Target) IsWasm() bool {
	return t.Arch.Family == sys.Wasm
}

//
// OS Functions
//

func (t *Target) IsLinux() bool {
	t.mustSetHeadType()
	return t.HeadType == objabi.Hlinux
}

func (t *Target) IsDarwin() bool {
	t.mustSetHeadType()
	return t.HeadType == objabi.Hdarwin
}

func (t *Target) IsWindows() bool {
	t.mustSetHeadType()
	return t.HeadType == objabi.Hwindows
}

func (t *Target) IsPlan9() bool {
	t.mustSetHeadType()
	return t.HeadType == objabi.Hplan9
}

func (t *Target) IsAIX() bool {
	t.mustSetHeadType()
	return t.HeadType == objabi.Haix
}

func (t *Target) IsSolaris() bool {
	t.mustSetHeadType()
	return t.HeadType == objabi.Hsolaris
}

func (t *Target) IsNetbsd() bool {
	t.mustSetHeadType()
	return t.HeadType == objabi.Hnetbsd
}

func (t *Target) IsOpenbsd() bool {
	t.mustSetHeadType()
	return t.HeadType == objabi.Hopenbsd
}

func (t *Target) mustSetHeadType() {
	if t.HeadType == objabi.Hunknown {
		panic("HeadType is not set")
	}
}

//
// MISC
//

func (t *Target) IsBigEndian() bool {
	return t.Arch.ByteOrder == binary.BigEndian
}
