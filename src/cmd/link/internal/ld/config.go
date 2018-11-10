// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/obj"
	"cmd/internal/sys"
	"fmt"
	"log"
)

var (
	Linkmode  LinkMode
	Buildmode BuildMode
)

// A BuildMode indicates the sort of object we are building.
//
// Possible build modes are the same as those for the -buildmode flag
// in cmd/go, and are documented in 'go help buildmode'.
type BuildMode uint8

const (
	BuildmodeUnset BuildMode = iota
	BuildmodeExe
	BuildmodePIE
	BuildmodeCArchive
	BuildmodeCShared
	BuildmodeShared
	BuildmodePlugin
)

func (mode *BuildMode) Set(s string) error {
	badmode := func() error {
		return fmt.Errorf("buildmode %s not supported on %s/%s", s, obj.GOOS, obj.GOARCH)
	}
	switch s {
	default:
		return fmt.Errorf("invalid buildmode: %q", s)
	case "exe":
		*mode = BuildmodeExe
	case "pie":
		switch obj.GOOS {
		case "android", "linux":
		default:
			return badmode()
		}
		*mode = BuildmodePIE
	case "c-archive":
		switch obj.GOOS {
		case "darwin", "linux":
		case "windows":
			switch obj.GOARCH {
			case "amd64", "386":
			default:
				return badmode()
			}
		default:
			return badmode()
		}
		*mode = BuildmodeCArchive
	case "c-shared":
		switch obj.GOARCH {
		case "386", "amd64", "arm", "arm64":
		default:
			return badmode()
		}
		*mode = BuildmodeCShared
	case "shared":
		switch obj.GOOS {
		case "linux":
			switch obj.GOARCH {
			case "386", "amd64", "arm", "arm64", "ppc64le", "s390x":
			default:
				return badmode()
			}
		default:
			return badmode()
		}
		*mode = BuildmodeShared
	case "plugin":
		switch obj.GOOS {
		case "linux":
			switch obj.GOARCH {
			case "386", "amd64", "arm", "arm64":
			default:
				return badmode()
			}
		case "darwin":
			switch obj.GOARCH {
			case "amd64":
			default:
				return badmode()
			}
		default:
			return badmode()
		}
		*mode = BuildmodePlugin
	}
	return nil
}

func (mode *BuildMode) String() string {
	switch *mode {
	case BuildmodeUnset:
		return "" // avoid showing a default in usage message
	case BuildmodeExe:
		return "exe"
	case BuildmodePIE:
		return "pie"
	case BuildmodeCArchive:
		return "c-archive"
	case BuildmodeCShared:
		return "c-shared"
	case BuildmodeShared:
		return "shared"
	case BuildmodePlugin:
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

	switch obj.GOOS {
	case "android":
		return true, "android"
	case "darwin":
		if SysArch.InFamily(sys.ARM, sys.ARM64) {
			return true, "iOS"
		}
	}

	if *flagMsan {
		return true, "msan"
	}

	// Internally linking cgo is incomplete on some architectures.
	// https://golang.org/issue/10373
	// https://golang.org/issue/14449
	if iscgo && SysArch.InFamily(sys.ARM64, sys.MIPS64, sys.MIPS) {
		return true, obj.GOARCH + " does not support internal cgo"
	}

	// Some build modes require work the internal linker cannot do (yet).
	switch Buildmode {
	case BuildmodeCArchive:
		return true, "buildmode=c-archive"
	case BuildmodeCShared:
		return true, "buildmode=c-shared"
	case BuildmodePIE:
		switch obj.GOOS + "/" + obj.GOARCH {
		case "linux/amd64":
		default:
			// Internal linking does not support TLS_IE.
			return true, "buildmode=pie"
		}
	case BuildmodePlugin:
		return true, "buildmode=plugin"
	case BuildmodeShared:
		return true, "buildmode=shared"
	}
	if *FlagLinkshared {
		return true, "dynamically linking with a shared library"
	}

	return false, ""
}

// determineLinkMode sets Linkmode.
//
// It is called after flags are processed and inputs are processed,
// so the Linkmode variable has an initial value from the -linkmode
// flag and the iscgo externalobj variables are set.
func determineLinkMode(ctxt *Link) {
	switch Linkmode {
	case LinkAuto:
		// The environment variable GO_EXTLINK_ENABLED controls the
		// default value of -linkmode. If it is not set when the
		// linker is called we take the value it was set to when
		// cmd/link was compiled. (See make.bash.)
		switch obj.Getgoextlinkenabled() {
		case "0":
			if needed, reason := mustLinkExternal(ctxt); needed {
				Exitf("internal linking requested via GO_EXTLINK_ENABLED, but external linking required: %s", reason)
			}
			Linkmode = LinkInternal
		case "1":
			Linkmode = LinkExternal
		default:
			if needed, _ := mustLinkExternal(ctxt); needed {
				Linkmode = LinkExternal
			} else if iscgo && externalobj {
				Linkmode = LinkExternal
			} else if Buildmode == BuildmodePIE {
				Linkmode = LinkExternal // https://golang.org/issue/18968
			} else {
				Linkmode = LinkInternal
			}
		}
	case LinkInternal:
		if needed, reason := mustLinkExternal(ctxt); needed {
			Exitf("internal linking requested but external linking required: %s", reason)
		}
	}
}
