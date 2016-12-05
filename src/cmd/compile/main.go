// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/compile/internal/amd64"
	"cmd/compile/internal/arm"
	"cmd/compile/internal/arm64"
	"cmd/compile/internal/gc"
	"cmd/compile/internal/mips"
	"cmd/compile/internal/mips64"
	"cmd/compile/internal/ppc64"
	"cmd/compile/internal/s390x"
	"cmd/compile/internal/x86"
	"cmd/internal/obj"
	"fmt"
	"log"
	"os"
)

func main() {
	// disable timestamps for reproducible output
	log.SetFlags(0)
	log.SetPrefix("compile: ")

	switch obj.GOARCH {
	default:
		fmt.Fprintf(os.Stderr, "compile: unknown architecture %q\n", obj.GOARCH)
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

	gc.Main()
	gc.Exit(0)
}
