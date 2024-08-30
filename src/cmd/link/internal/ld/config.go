// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"fmt"
	"internal/buildcfg"
	"internal/platform"
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

// Set implements flag.Value to set the build mode based on the argument
// to the -buildmode flag.
func (mode *BuildMode) Set(s string) error {
	switch s {
	default:
		return fmt.Errorf("invalid buildmode: %q", s)
	case "exe":
		switch buildcfg.GOOS + "/" + buildcfg.GOARCH {
		case "darwin/arm64", "windows/arm", "windows/arm64": // On these platforms, everything is PIE
			*mode = BuildModePIE
		default:
			*mode = BuildModeExe
		}
	case "pie":
		*mode = BuildModePIE
	case "c-archive":
		*mode = BuildModeCArchive
	case "c-shared":
		*mode = BuildModeCShared
	case "shared":
		*mode = BuildModeShared
	case "plugin":
		*mode = BuildModePlugin
	}

	if !platform.BuildModeSupported("gc", s, buildcfg.GOOS, buildcfg.GOARCH) {
		return fmt.Errorf("buildmode %s not supported on %s/%s", s, buildcfg.GOOS, buildcfg.GOARCH)
	}

	return nil
}

func (mode BuildMode) String() string {
	switch mode {
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
	return fmt.Sprintf("BuildMode(%d)", uint8(mode))
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
				ctxt.Logf("external linking is forced by: %s\n", reason)
			}
		}()
	}

	if platform.MustLinkExternal(buildcfg.GOOS, buildcfg.GOARCH, false) {
		return true, fmt.Sprintf("%s/%s requires external linking", buildcfg.GOOS, buildcfg.GOARCH)
	}

	if *flagMsan {
		return true, "msan"
	}

	if *flagAsan {
		return true, "asan"
	}

	if iscgo && platform.MustLinkExternal(buildcfg.GOOS, buildcfg.GOARCH, true) {
		return true, buildcfg.GOARCH + " does not support internal cgo"
	}

	// Some build modes require work the internal linker cannot do (yet).
	switch ctxt.BuildMode {
	case BuildModeCArchive:
		return true, "buildmode=c-archive"
	case BuildModeCShared:
		if buildcfg.GOARCH == "wasm" {
			break
		}
		return true, "buildmode=c-shared"
	case BuildModePIE:
		if !platform.InternalLinkPIESupported(buildcfg.GOOS, buildcfg.GOARCH) {
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

	if unknownObjFormat {
		return true, "some input objects have an unrecognized file format"
	}

	if len(dynimportfail) > 0 {
		// This error means that we were unable to generate
		// the _cgo_import.go file for some packages.
		// This typically means that there are some dependencies
		// that the cgo tool could not figure out.
		// See issue #52863.
		return true, fmt.Sprintf("some packages could not be built to support internal linking (%v)", dynimportfail)
	}

	return false, ""
}

// determineLinkMode sets ctxt.LinkMode.
//
// It is called after flags are processed and inputs are processed,
// so the ctxt.LinkMode variable has an initial value from the -linkmode
// flag and the iscgo, externalobj, and unknownObjFormat variables are set.
func determineLinkMode(ctxt *Link) {
	extNeeded, extReason := mustLinkExternal(ctxt)
	via := ""

	if ctxt.LinkMode == LinkAuto {
		// The environment variable GO_EXTLINK_ENABLED controls the
		// default value of -linkmode. If it is not set when the
		// linker is called we take the value it was set to when
		// cmd/link was compiled. (See make.bash.)
		switch buildcfg.Getgoextlinkenabled() {
		case "0":
			ctxt.LinkMode = LinkInternal
			via = "via GO_EXTLINK_ENABLED "
		case "1":
			ctxt.LinkMode = LinkExternal
			via = "via GO_EXTLINK_ENABLED "
		default:
			preferExternal := len(preferlinkext) != 0
			if preferExternal && ctxt.Debugvlog > 0 {
				ctxt.Logf("external linking prefer list is %v\n", preferlinkext)
			}
			if extNeeded || (iscgo && (externalobj || preferExternal)) {
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
		case buildcfg.GOARCH == "ppc64" && buildcfg.GOOS == "linux":
			Exitf("external linking not supported for %s/ppc64", buildcfg.GOOS)
		}
	}
}
