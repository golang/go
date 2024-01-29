// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package buildcfg provides access to the build configuration
// described by the current environment. It is for use by build tools
// such as cmd/go or cmd/compile and for setting up go/build's Default context.
//
// Note that it does NOT provide access to the build configuration used to
// build the currently-running binary. For that, use runtime.GOOS etc
// as well as internal/goexperiment.
package buildcfg

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

var (
	GOROOT    = runtime.GOROOT() // cached for efficiency
	GOARCH    = envOr("GOARCH", defaultGOARCH)
	GOOS      = envOr("GOOS", defaultGOOS)
	GO386     = envOr("GO386", defaultGO386)
	GOAMD64   = goamd64()
	GOARM     = goarm()
	GOMIPS    = gomips()
	GOMIPS64  = gomips64()
	GOPPC64   = goppc64()
	GORISCV64 = goriscv64()
	GOWASM    = gowasm()
	ToolTags  = toolTags()
	GO_LDSO   = defaultGO_LDSO
	Version   = version
)

// Error is one of the errors found (if any) in the build configuration.
var Error error

// Check exits the program with a fatal error if Error is non-nil.
func Check() {
	if Error != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", filepath.Base(os.Args[0]), Error)
		os.Exit(2)
	}
}

func envOr(key, value string) string {
	if x := os.Getenv(key); x != "" {
		return x
	}
	return value
}

func goamd64() int {
	switch v := envOr("GOAMD64", defaultGOAMD64); v {
	case "v1":
		return 1
	case "v2":
		return 2
	case "v3":
		return 3
	case "v4":
		return 4
	}
	Error = fmt.Errorf("invalid GOAMD64: must be v1, v2, v3, v4")
	return int(defaultGOAMD64[len("v")] - '0')
}

type goarmFeatures struct {
	Version   int
	SoftFloat bool
}

func (g goarmFeatures) String() string {
	armStr := strconv.Itoa(g.Version)
	if g.SoftFloat {
		armStr += ",softfloat"
	} else {
		armStr += ",hardfloat"
	}
	return armStr
}

func goarm() (g goarmFeatures) {
	const (
		softFloatOpt = ",softfloat"
		hardFloatOpt = ",hardfloat"
	)
	def := defaultGOARM
	if GOOS == "android" && GOARCH == "arm" {
		// Android arm devices always support GOARM=7.
		def = "7"
	}
	v := envOr("GOARM", def)

	floatSpecified := false
	if strings.HasSuffix(v, softFloatOpt) {
		g.SoftFloat = true
		floatSpecified = true
		v = v[:len(v)-len(softFloatOpt)]
	}
	if strings.HasSuffix(v, hardFloatOpt) {
		floatSpecified = true
		v = v[:len(v)-len(hardFloatOpt)]
	}

	switch v {
	case "5":
		g.Version = 5
	case "6":
		g.Version = 6
	case "7":
		g.Version = 7
	default:
		Error = fmt.Errorf("invalid GOARM: must start with 5, 6, or 7, and may optionally end in either %q or %q", hardFloatOpt, softFloatOpt)
		g.Version = int(def[0] - '0')
	}

	// 5 defaults to softfloat. 6 and 7 default to hardfloat.
	if !floatSpecified && g.Version == 5 {
		g.SoftFloat = true
	}
	return
}

func gomips() string {
	switch v := envOr("GOMIPS", defaultGOMIPS); v {
	case "hardfloat", "softfloat":
		return v
	}
	Error = fmt.Errorf("invalid GOMIPS: must be hardfloat, softfloat")
	return defaultGOMIPS
}

func gomips64() string {
	switch v := envOr("GOMIPS64", defaultGOMIPS64); v {
	case "hardfloat", "softfloat":
		return v
	}
	Error = fmt.Errorf("invalid GOMIPS64: must be hardfloat, softfloat")
	return defaultGOMIPS64
}

func goppc64() int {
	switch v := envOr("GOPPC64", defaultGOPPC64); v {
	case "power8":
		return 8
	case "power9":
		return 9
	case "power10":
		return 10
	}
	Error = fmt.Errorf("invalid GOPPC64: must be power8, power9, power10")
	return int(defaultGOPPC64[len("power")] - '0')
}

func goriscv64() int {
	switch v := envOr("GORISCV64", defaultGORISCV64); v {
	case "rva20u64":
		return 20
	case "rva22u64":
		return 22
	}
	Error = fmt.Errorf("invalid GORISCV64: must be rva20u64, rva22u64")
	v := defaultGORISCV64[len("rva"):]
	i := strings.IndexFunc(v, func(r rune) bool {
		return r < '0' || r > '9'
	})
	year, _ := strconv.Atoi(v[:i])
	return year
}

type gowasmFeatures struct {
	SatConv bool
	SignExt bool
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
			Error = fmt.Errorf("invalid GOWASM: no such feature %q", opt)
		}
	}
	return
}

func Getgoextlinkenabled() string {
	return envOr("GO_EXTLINK_ENABLED", defaultGO_EXTLINK_ENABLED)
}

func toolTags() []string {
	tags := experimentTags()
	tags = append(tags, gogoarchTags()...)
	return tags
}

func experimentTags() []string {
	var list []string
	// For each experiment that has been enabled in the toolchain, define a
	// build tag with the same name but prefixed by "goexperiment." which can be
	// used for compiling alternative files for the experiment. This allows
	// changes for the experiment, like extra struct fields in the runtime,
	// without affecting the base non-experiment code at all.
	for _, exp := range Experiment.Enabled() {
		list = append(list, "goexperiment."+exp)
	}
	return list
}

// GOGOARCH returns the name and value of the GO$GOARCH setting.
// For example, if GOARCH is "amd64" it might return "GOAMD64", "v2".
func GOGOARCH() (name, value string) {
	switch GOARCH {
	case "386":
		return "GO386", GO386
	case "amd64":
		return "GOAMD64", fmt.Sprintf("v%d", GOAMD64)
	case "arm":
		return "GOARM", GOARM.String()
	case "mips", "mipsle":
		return "GOMIPS", GOMIPS
	case "mips64", "mips64le":
		return "GOMIPS64", GOMIPS64
	case "ppc64", "ppc64le":
		return "GOPPC64", fmt.Sprintf("power%d", GOPPC64)
	case "wasm":
		return "GOWASM", GOWASM.String()
	}
	return "", ""
}

func gogoarchTags() []string {
	switch GOARCH {
	case "386":
		return []string{GOARCH + "." + GO386}
	case "amd64":
		var list []string
		for i := 1; i <= GOAMD64; i++ {
			list = append(list, fmt.Sprintf("%s.v%d", GOARCH, i))
		}
		return list
	case "arm":
		var list []string
		for i := 5; i <= GOARM.Version; i++ {
			list = append(list, fmt.Sprintf("%s.%d", GOARCH, i))
		}
		return list
	case "mips", "mipsle":
		return []string{GOARCH + "." + GOMIPS}
	case "mips64", "mips64le":
		return []string{GOARCH + "." + GOMIPS64}
	case "ppc64", "ppc64le":
		var list []string
		for i := 8; i <= GOPPC64; i++ {
			list = append(list, fmt.Sprintf("%s.power%d", GOARCH, i))
		}
		return list
	case "riscv64":
		list := []string{GOARCH + "." + "rva20u64"}
		if GORISCV64 >= 22 {
			list = append(list, GOARCH+"."+"rva22u64")
		}
		return list
	case "wasm":
		var list []string
		if GOWASM.SatConv {
			list = append(list, GOARCH+".satconv")
		}
		if GOWASM.SignExt {
			list = append(list, GOARCH+".signext")
		}
		return list
	}
	return nil
}
