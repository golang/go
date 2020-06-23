// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"fmt"
	"log"
	"os"
	"strings"
)

func envOr(key, value string) string {
	if x := os.Getenv(key); x != "" {
		return x
	}
	return value
}

var (
	defaultGOROOT string // set by linker

	GOROOT   = envOr("GOROOT", defaultGOROOT)
	GOARCH   = envOr("GOARCH", defaultGOARCH)
	GOOS     = envOr("GOOS", defaultGOOS)
	GO386    = envOr("GO386", defaultGO386)
	GOAMD64  = goamd64()
	GOARM    = goarm()
	GOMIPS   = gomips()
	GOMIPS64 = gomips64()
	GOPPC64  = goppc64()
	GOWASM   = gowasm()
	GO_LDSO  = defaultGO_LDSO
	Version  = version
)

const (
	ElfRelocOffset   = 256
	MachoRelocOffset = 2048           // reserve enough space for ELF relocations
	Go115AMD64       = "alignedjumps" // Should be "alignedjumps" or "normaljumps"; this replaces environment variable introduced in CL 219357.
)

// TODO(1.16): assuming no issues in 1.15 release, remove this and related constant.
func goamd64() string {
	return Go115AMD64
}

func goarm() int {
	switch v := envOr("GOARM", defaultGOARM); v {
	case "5":
		return 5
	case "6":
		return 6
	case "7":
		return 7
	}
	// Fail here, rather than validate at multiple call sites.
	log.Fatalf("Invalid GOARM value. Must be 5, 6, or 7.")
	panic("unreachable")
}

func gomips() string {
	switch v := envOr("GOMIPS", defaultGOMIPS); v {
	case "hardfloat", "softfloat":
		return v
	}
	log.Fatalf("Invalid GOMIPS value. Must be hardfloat or softfloat.")
	panic("unreachable")
}

func gomips64() string {
	switch v := envOr("GOMIPS64", defaultGOMIPS64); v {
	case "hardfloat", "softfloat":
		return v
	}
	log.Fatalf("Invalid GOMIPS64 value. Must be hardfloat or softfloat.")
	panic("unreachable")
}

func goppc64() int {
	switch v := envOr("GOPPC64", defaultGOPPC64); v {
	case "power8":
		return 8
	case "power9":
		return 9
	}
	log.Fatalf("Invalid GOPPC64 value. Must be power8 or power9.")
	panic("unreachable")
}

type gowasmFeatures struct {
	SignExt bool
	SatConv bool
}

func (f gowasmFeatures) String() string {
	var flags []string
	if f.SatConv {
		flags = append(flags, "satconv")
	}
	if f.SignExt {
		flags = append(flags, "signext")
	}
	return strings.Join(flags, ",")
}

func gowasm() (f gowasmFeatures) {
	for _, opt := range strings.Split(envOr("GOWASM", ""), ",") {
		switch opt {
		case "satconv":
			f.SatConv = true
		case "signext":
			f.SignExt = true
		case "":
			// ignore
		default:
			log.Fatalf("Invalid GOWASM value. No such feature: " + opt)
		}
	}
	return
}

func Getgoextlinkenabled() string {
	return envOr("GO_EXTLINK_ENABLED", defaultGO_EXTLINK_ENABLED)
}

func init() {
	for _, f := range strings.Split(goexperiment, ",") {
		if f != "" {
			addexp(f)
		}
	}
}

func Framepointer_enabled(goos, goarch string) bool {
	return framepointer_enabled != 0 && (goarch == "amd64" || goarch == "arm64" && (goos == "linux" || goos == "darwin"))
}

func addexp(s string) {
	// Could do general integer parsing here, but the runtime copy doesn't yet.
	v := 1
	name := s
	if len(name) > 2 && name[:2] == "no" {
		v = 0
		name = name[2:]
	}
	for i := 0; i < len(exper); i++ {
		if exper[i].name == name {
			if exper[i].val != nil {
				*exper[i].val = v
			}
			return
		}
	}

	fmt.Printf("unknown experiment %s\n", s)
	os.Exit(2)
}

var (
	framepointer_enabled      int = 1
	Fieldtrack_enabled        int
	Preemptibleloops_enabled  int
	Staticlockranking_enabled int
)

// Toolchain experiments.
// These are controlled by the GOEXPERIMENT environment
// variable recorded when the toolchain is built.
// This list is also known to cmd/gc.
var exper = []struct {
	name string
	val  *int
}{
	{"fieldtrack", &Fieldtrack_enabled},
	{"framepointer", &framepointer_enabled},
	{"preemptibleloops", &Preemptibleloops_enabled},
	{"staticlockranking", &Staticlockranking_enabled},
}

var defaultExpstring = Expstring()

func DefaultExpstring() string {
	return defaultExpstring
}

func Expstring() string {
	buf := "X"
	for i := range exper {
		if *exper[i].val != 0 {
			buf += "," + exper[i].name
		}
	}
	if buf == "X" {
		buf += ",none"
	}
	return "X:" + buf[2:]
}
