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
	"strings"
)

var (
	GOROOT   = runtime.GOROOT() // cached for efficiency
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
	Error = fmt.Errorf("invalid GOARM: must be 5, 6, 7")
	return int(def[0] - '0')
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
	}
	Error = fmt.Errorf("invalid GOPPC64: must be power8, power9")
	return int(defaultGOPPC64[len("power")] - '0')
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
			Error = fmt.Errorf("invalid GOWASM: no such feature %q", opt)
		}
	}
	return
}

func Getgoextlinkenabled() string {
	return envOr("GO_EXTLINK_ENABLED", defaultGO_EXTLINK_ENABLED)
}
