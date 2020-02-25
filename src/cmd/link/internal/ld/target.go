// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
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

func (t *Target) IsShared() bool {
	return t.BuildMode == BuildModeShared
}

func (t *Target) IsPlugin() bool {
	return t.BuildMode == BuildModePlugin
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
	return t.IsELF
}

func (t *Target) IsDynlinkingGo() bool {
	return t.IsShared() || t.IsSharedGoLink() || t.IsPlugin() || t.CanUsePlugins()
}

//
// Processor functions
//

func (t *Target) IsARM() bool {
	return t.Arch.Family == sys.ARM
}

func (t *Target) IsAMD64() bool {
	return t.Arch.Family == sys.AMD64
}

func (t *Target) IsPPC64() bool {
	return t.Arch.Family == sys.PPC64
}

func (t *Target) IsS390X() bool {
	return t.Arch.Family == sys.S390X
}

//
// OS Functions
//

func (t *Target) IsDarwin() bool {
	return t.HeadType == objabi.Hdarwin
}

func (t *Target) IsWindows() bool {
	return t.HeadType == objabi.Hwindows
}

func (t *Target) IsPlan9() bool {
	return t.HeadType == objabi.Hplan9
}

func (t *Target) IsAIX() bool {
	return t.HeadType == objabi.Haix
}

func (t *Target) IsSolaris() bool {
	return t.HeadType == objabi.Hsolaris
}
