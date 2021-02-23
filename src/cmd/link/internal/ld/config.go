// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"fmt"
	"log"
)

// A BuildMode indicates the sort of object we are building.
//
// Possible build modes are the same as those for the -buildmode flag
// in cmd/go, and are documented in 'go help buildmode'.
type BuildMode uint8

const (
	BuildModeUnset BuildMode = iota
	BuildModeExe
	BuildModePIE
	BuildModeCArchive
	BuildModeCShared
	BuildModeShared
	BuildModePlugin
)

func (mode *BuildMode) Set(s string) error {
	badmode := func() error {
		return fmt.Errorf("buildmode %s not supported on %s/%s", s, objabi.GOOS, objabi.GOARCH)
	}
	switch s {
	default:
		return fmt.Errorf("invalid buildmode: %q", s)
	case "exe":
		switch objabi.GOOS + "/" + objabi.GOARCH {
		case "darwin/arm64", "windows/arm": // On these platforms, everything is PIE
			*mode = BuildModePIE
		default:
			*mode = BuildModeExe
		}
	case "pie":
		switch objabi.GOOS {
		case "aix", "android", "linux", "windows", "darwin", "ios":
		case "freebsd":
			switch objabi.GOARCH {
			case "amd64":
			default:
				return badmode()
			}
		default:
			return badmode()
		}
		*mode = BuildModePIE
	case "c-archive":
		switch objabi.GOOS {
		case "aix", "darwin", "ios", "linux":
		case "freebsd":
			switch objabi.GOARCH {
			case "amd64":
			default:
				return badmode()
			}
		case "windows":
			switch objabi.GOARCH {
			case "amd64", "386", "arm":
			default:
				return badmode()
			}
		default:
			return badmode()
		}
		*mode = BuildModeCArchive
	case "c-shared":
		switch objabi.GOARCH {
		case "386", "amd64", "arm", "arm64", "ppc64le", "s390x":
		default:
			return badmode()
		}
		*mode = BuildModeCShared
	case "shared":
		switch objabi.GOOS {
		case "linux":
			switch objabi.GOARCH {
			case "386", "amd64", "arm", "arm64", "ppc64le", "s390x":
			default:
				return badmode()
			}
		default:
			return badmode()
		}
		*mode = BuildModeShared
	case "plugin":
		switch objabi.GOOS {
		case "linux":
			switch objabi.GOARCH {
			case "386", "amd64", "arm", "arm64", "s390x", "ppc64le":
			default:
				return badmode()
			}
		case "darwin":
			switch objabi.GOARCH {
			case "amd64", "arm64":
			default:
				return badmode()
			}
		case "freebsd":
			switch objabi.GOARCH {
			case "amd64":
			default:
				return badmode()
			}
		default:
			return badmode()
		}
		*mode = BuildModePlugin
	}
	return nil
}

func (mode *BuildMode) String() string {
	switch *mode {
	case BuildModeUnset:
		return "" // avoid showing a default in usage message
	case BuildModeExe:
		return "exe"
	case BuildModePIE:
		return "pie"
	case BuildModeCArchive:
		return "c-archive"
	case BuildModeCShared:
		return "c-shared"
	case BuildModeShared:
		return "shared"
	case BuildModePlugin:
		return "plugin"
	}
	return fmt.Sprintf("BuildMode(%d)", uint8(*mode))
}

// LinkMode indicates whether an external linker is used for the final link.
type LinkMode uint8

const (
	LinkAuto LinkMode = iota
	LinkInternal
	LinkExternal
)

func (mode *LinkMode) Set(s string) error {
	switch s {
	default:
		return fmt.Errorf("invalid linkmode: %q", s)
	case "auto":
		*mode = LinkAuto
	case "internal":
		*mode = LinkInternal
	case "external":
		*mode = LinkExternal
	}
	return nil
}

func (mode *LinkMode) String() string {
	switch *mode {
	case LinkAuto:
		return "auto"
	case LinkInternal:
		return "internal"
	case LinkExternal:
		return "external"
	}
	return fmt.Sprintf("LinkMode(%d)", uint8(*mode))
}

// mustLinkExternal reports whether the program being linked requires
// the external linker be used to complete the link.
func mustLinkExternal(ctxt *Link) (res bool, reason string) {
	if ctxt.Debugvlog > 1 {
		defer func() {
			if res {
				log.Printf("external linking is forced by: %s\n", reason)
			}
		}()
	}

	if sys.MustLinkExternal(objabi.GOOS, objabi.GOARCH) {
		return true, fmt.Sprintf("%s/%s requires external linking", objabi.GOOS, objabi.GOARCH)
	}

	if *flagMsan {
		return true, "msan"
	}

	// Internally linking cgo is incomplete on some architectures.
	// https://golang.org/issue/14449
	// https://golang.org/issue/21961
	if iscgo && ctxt.Arch.InFamily(sys.MIPS64, sys.MIPS, sys.PPC64, sys.RISCV64) {
		return true, objabi.GOARCH + " does not support internal cgo"
	}
	if iscgo && objabi.GOOS == "android" {
		return true, objabi.GOOS + " does not support internal cgo"
	}

	// When the race flag is set, the LLVM tsan relocatable file is linked
	// into the final binary, which means external linking is required because
	// internal linking does not support it.
	if *flagRace && ctxt.Arch.InFamily(sys.PPC64) {
		return true, "race on " + objabi.GOARCH
	}

	// Some build modes require work the internal linker cannot do (yet).
	switch ctxt.BuildMode {
	case BuildModeCArchive:
		return true, "buildmode=c-archive"
	case BuildModeCShared:
		return true, "buildmode=c-shared"
	case BuildModePIE:
		switch objabi.GOOS + "/" + objabi.GOARCH {
		case "linux/amd64", "linux/arm64", "android/arm64":
		case "windows/386", "windows/amd64", "windows/arm":
		case "darwin/amd64", "darwin/arm64":
		default:
			// Internal linking does not support TLS_IE.
			return true, "buildmode=pie"
		}
	case BuildModePlugin:
		return true, "buildmode=plugin"
	case BuildModeShared:
		return true, "buildmode=shared"
	}
	if ctxt.linkShared {
		return true, "dynamically linking with a shared library"
	}

	return false, ""
}

// determineLinkMode sets ctxt.LinkMode.
//
// It is called after flags are processed and inputs are processed,
// so the ctxt.LinkMode variable has an initial value from the -linkmode
// flag and the iscgo externalobj variables are set.
func determineLinkMode(ctxt *Link) {
	extNeeded, extReason := mustLinkExternal(ctxt)
	via := ""

	if ctxt.LinkMode == LinkAuto {
		// The environment variable GO_EXTLINK_ENABLED controls the
		// default value of -linkmode. If it is not set when the
		// linker is called we take the value it was set to when
		// cmd/link was compiled. (See make.bash.)
		switch objabi.Getgoextlinkenabled() {
		case "0":
			ctxt.LinkMode = LinkInternal
			via = "via GO_EXTLINK_ENABLED "
		case "1":
			ctxt.LinkMode = LinkExternal
			via = "via GO_EXTLINK_ENABLED "
		default:
			if extNeeded || (iscgo && externalobj) {
				ctxt.LinkMode = LinkExternal
			} else {
				ctxt.LinkMode = LinkInternal
			}
		}
	}

	switch ctxt.LinkMode {
	case LinkInternal:
		if extNeeded {
			Exitf("internal linking requested %sbut external linking required: %s", via, extReason)
		}
	case LinkExternal:
		switch {
		case objabi.GOARCH == "ppc64" && objabi.GOOS != "aix":
			Exitf("external linking not supported for %s/ppc64", objabi.GOOS)
		}
	}
}
