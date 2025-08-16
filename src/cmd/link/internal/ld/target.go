// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"encoding/binary"
	"fmt"
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

	TLSModel TLSModel // TLS model selection
}

// TLSModel represents different Thread Local Storage models
type TLSModel int

const (
	TLSModelAuto TLSModel = iota // Automatic selection based on platform and build mode
	TLSModelLE                   // Local Exec - fastest, static executables only
	TLSModelIE                   // Initial Exec - fast, may not work with dlopen on non-glibc
	TLSModelGD                   // General Dynamic - compatible with all dlopen scenarios
)

func (t TLSModel) String() string {
	switch t {
	case TLSModelAuto:
		return "auto"
	case TLSModelLE:
		return "LE"
	case TLSModelIE:
		return "IE"
	case TLSModelGD:
		return "GD"
	default:
		return "unknown"
	}
}

// Set implements flag.Value interface for TLS model parsing
func (t *TLSModel) Set(s string) error {
	switch s {
	case "auto":
		*t = TLSModelAuto
	case "LE":
		*t = TLSModelLE
	case "IE":
		*t = TLSModelIE
	case "GD":
		*t = TLSModelGD
	default:
		return fmt.Errorf("invalid TLS model %q; valid values are auto, LE, IE, GD", s)
	}
	return nil
}

// ValidateTLSModel checks if the TLS model is valid for the target platform and build mode
func (t *Target) ValidateTLSModel() error {
	switch t.TLSModel {
	case TLSModelAuto:
		// Auto is always valid
		return nil
	case TLSModelLE:
		// LE model is only valid for static executables
		if t.BuildMode == BuildModeShared || t.BuildMode == BuildModeCArchive || t.BuildMode == BuildModeCShared {
			return fmt.Errorf("LE TLS model invalid for shared libraries (requires static TLS allocation)")
		}
		// LE not supported on Windows/Plan9
		if t.HeadType == objabi.Hwindows || t.HeadType == objabi.Hplan9 {
			return fmt.Errorf("LE TLS model not supported on %s", t.HeadType)
		}
	case TLSModelGD:
		// GD model requires __tls_get_addr support
		if t.HeadType == objabi.Hwindows || t.HeadType == objabi.Hplan9 {
			return fmt.Errorf("GD TLS model not supported on %s (no __tls_get_addr support)", t.HeadType)
		}
	case TLSModelIE:
		// IE model may fail on non-glibc systems when used with shared libraries
		if (t.BuildMode == BuildModeShared || t.BuildMode == BuildModeCArchive || t.BuildMode == BuildModeCShared) &&
			(t.HeadType == objabi.Hlinux || t.HeadType == objabi.Hfreebsd || t.HeadType == objabi.Hopenbsd) {
			// This is a warning case, not an error - user might know their deployment environment
			// We'll print a warning during linking but allow it
		}
	}
	return nil
}

// GetEffectiveTLSModel returns the actual TLS model to use, resolving "auto" to a concrete model
func (t *Target) GetEffectiveTLSModel() TLSModel {
	if t.TLSModel != TLSModelAuto {
		return t.TLSModel
	}

	// Auto selection logic - matches our current implementation
	switch {
	case t.BuildMode == BuildModeShared || t.BuildMode == BuildModeCArchive || t.BuildMode == BuildModeCShared:
		// For shared libraries, use GD on Unix platforms for compatibility
		if t.HeadType == objabi.Hlinux || t.HeadType == objabi.Hfreebsd || t.HeadType == objabi.Hopenbsd {
			return TLSModelGD
		}
		// For other platforms (Darwin, Windows), use IE
		return TLSModelIE
	default:
		// For static executables, use LE where supported, otherwise IE
		if t.HeadType == objabi.Hwindows || t.HeadType == objabi.Hplan9 {
			return TLSModelIE
		}
		return TLSModelLE
	}
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

func (t *Target) IsLOONG64() bool {
	return t.Arch.Family == sys.Loong64
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

func (t *Target) IsFreebsd() bool {
	t.mustSetHeadType()
	return t.HeadType == objabi.Hfreebsd
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

func (t *Target) UsesLibc() bool {
	t.mustSetHeadType()
	switch t.HeadType {
	case objabi.Haix, objabi.Hdarwin, objabi.Hopenbsd, objabi.Hsolaris, objabi.Hwindows:
		// platforms where we use libc for syscalls.
		return true
	}
	return false
}
