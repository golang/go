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
	GOARM    = goarm()
	GOMIPS   = gomips()
	GOMIPS64 = gomips64()
	GOPPC64  = goppc64()
	GOWASM   = gowasm()
	GO_LDSO  = defaultGO_LDSO
	Version  = version

	// GOEXPERIMENT is a comma-separated list of enabled
	// experiments. This is derived from the GOEXPERIMENT
	// environment variable if set, or the value of GOEXPERIMENT
	// when make.bash was run if not.
	GOEXPERIMENT string // Set by package init
)

const (
	ElfRelocOffset   = 256
	MachoRelocOffset = 2048 // reserve enough space for ELF relocations
)

func goarm() int {
	def := defaultGOARM
	if GOOS == "android" && GOARCH == "arm" {
		// Android arm devices always support GOARM=7.
		def = "7"
	}
	switch v := envOr("GOARM", def); v {
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
	// Capture "default" experiments.
	defaultExpstring = Expstring()

	goexperiment := envOr("GOEXPERIMENT", defaultGOEXPERIMENT)

	for _, f := range strings.Split(goexperiment, ",") {
		if f != "" {
			addexp(f)
		}
	}

	// regabi is only supported on amd64.
	if GOARCH != "amd64" {
		Regabi_enabled = 0
	}

	// Set GOEXPERIMENT to the parsed and canonicalized set of experiments.
	GOEXPERIMENT = expList()
}

// Note: must agree with runtime.framepointer_enabled.
var Framepointer_enabled = GOARCH == "amd64" || GOARCH == "arm64"

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
	Fieldtrack_enabled        int
	Preemptibleloops_enabled  int
	Staticlockranking_enabled int
	Regabi_enabled            int
)

// Toolchain experiments.
// These are controlled by the GOEXPERIMENT environment
// variable recorded when the toolchain is built.
var exper = []struct {
	name string
	val  *int
}{
	{"fieldtrack", &Fieldtrack_enabled},
	{"preemptibleloops", &Preemptibleloops_enabled},
	{"staticlockranking", &Staticlockranking_enabled},
	{"regabi", &Regabi_enabled},
}

var defaultExpstring string

// expList returns the list of enabled GOEXPERIMENTS as a
// commas-separated list.
func expList() string {
	buf := ""
	for i := range exper {
		if *exper[i].val != 0 {
			buf += "," + exper[i].name
		}
	}
	if len(buf) == 0 {
		return ""
	}
	return buf[1:]
}

// Expstring returns the GOEXPERIMENT string that should appear in Go
// version signatures. This always starts with "X:".
func Expstring() string {
	list := expList()
	if list == "" {
		return "X:none"
	}
	return "X:" + list
}
