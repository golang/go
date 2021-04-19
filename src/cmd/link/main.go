// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/link/internal/amd64"
	"cmd/link/internal/arm"
	"cmd/link/internal/arm64"
	"cmd/link/internal/ld"
	"cmd/link/internal/mips"
	"cmd/link/internal/mips64"
	"cmd/link/internal/ppc64"
	"cmd/link/internal/s390x"
	"cmd/link/internal/x86"
	"fmt"
	"os"
)

// The bulk of the linker implementation lives in cmd/link/internal/ld.
// Architecture-specific code lives in cmd/link/internal/GOARCH.
//
// Program initialization:
//
// Before any argument parsing is done, the Init function of relevant
// architecture package is called. The only job done in Init is
// configuration of the ld.Thearch and ld.SysArch variables.
//
// Then control flow passes to ld.Main, which parses flags, makes
// some configuration decisions, and then gives the architecture
// packages a second chance to modify the linker's configuration
// via the ld.Thearch.Archinit function.

func main() {
	switch obj.GOARCH {
	default:
		fmt.Fprintf(os.Stderr, "link: unknown architecture %q\n", obj.GOARCH)
		os.Exit(2)
	case "386":
		x86.Init()
	case "amd64", "amd64p32":
		amd64.Init()
	case "arm":
		arm.Init()
	case "arm64":
		arm64.Init()
	case "mips", "mipsle":
		mips.Init()
	case "mips64", "mips64le":
		mips64.Init()
	case "ppc64", "ppc64le":
		ppc64.Init()
	case "s390x":
		s390x.Init()
	}
	ld.Main()
}
