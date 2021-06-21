// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildcfg

import (
	"fmt"
	"os"
	"reflect"
	"strings"

	"internal/goexperiment"
)

// Experiment contains the toolchain experiments enabled for the
// current build.
//
// (This is not necessarily the set of experiments the compiler itself
// was built with.)
var Experiment goexperiment.Flags = parseExperiments(GOARCH)

var regabiSupported = GOARCH == "amd64" && (GOOS == "android" || GOOS == "linux" || GOOS == "darwin" || GOOS == "windows")

// experimentBaseline specifies the experiment flags that are enabled by
// default in the current toolchain. This is, in effect, the "control"
// configuration and any variation from this is an experiment.
var experimentBaseline = goexperiment.Flags{
	RegabiWrappers: regabiSupported,
	RegabiG:        regabiSupported,
	RegabiReflect:  regabiSupported,
	RegabiDefer:    regabiSupported,
	RegabiArgs:     regabiSupported,
}

// FramePointerEnabled enables the use of platform conventions for
// saving frame pointers.
//
// This used to be an experiment, but now it's always enabled on
// platforms that support it.
//
// Note: must agree with runtime.framepointer_enabled.
var FramePointerEnabled = GOARCH == "amd64" || GOARCH == "arm64"

func parseExperiments(goarch string) goexperiment.Flags {
	// Start with the statically enabled set of experiments.
	flags := experimentBaseline

	// Pick up any changes to the baseline configuration from the
	// GOEXPERIMENT environment. This can be set at make.bash time
	// and overridden at build time.
	env := envOr("GOEXPERIMENT", defaultGOEXPERIMENT)

	if env != "" {
		// Create a map of known experiment names.
		names := make(map[string]func(bool))
		rv := reflect.ValueOf(&flags).Elem()
		rt := rv.Type()
		for i := 0; i < rt.NumField(); i++ {
			field := rv.Field(i)
			names[strings.ToLower(rt.Field(i).Name)] = field.SetBool
		}

		// "regabi" is an alias for all working regabi
		// subexperiments, and not an experiment itself. Doing
		// this as an alias make both "regabi" and "noregabi"
		// do the right thing.
		names["regabi"] = func(v bool) {
			flags.RegabiWrappers = v
			flags.RegabiG = v
			flags.RegabiReflect = v
			flags.RegabiDefer = v
			flags.RegabiArgs = v
		}

		// Parse names.
		for _, f := range strings.Split(env, ",") {
			if f == "" {
				continue
			}
			if f == "none" {
				// GOEXPERIMENT=none disables all experiment flags.
				// This is used by cmd/dist, which doesn't know how
				// to build with any experiment flags.
				flags = goexperiment.Flags{}
				continue
			}
			val := true
			if strings.HasPrefix(f, "no") {
				f, val = f[2:], false
			}
			set, ok := names[f]
			if !ok {
				fmt.Printf("unknown experiment %s\n", f)
				os.Exit(2)
			}
			set(val)
		}
	}

	// regabi is only supported on amd64.
	if goarch != "amd64" {
		flags.RegabiWrappers = false
		flags.RegabiG = false
		flags.RegabiReflect = false
		flags.RegabiDefer = false
		flags.RegabiArgs = false
	}
	// Check regabi dependencies.
	if flags.RegabiG && !flags.RegabiWrappers {
		Error = fmt.Errorf("GOEXPERIMENT regabig requires regabiwrappers")
	}
	if flags.RegabiArgs && !(flags.RegabiWrappers && flags.RegabiG && flags.RegabiReflect && flags.RegabiDefer) {
		Error = fmt.Errorf("GOEXPERIMENT regabiargs requires regabiwrappers,regabig,regabireflect,regabidefer")
	}
	return flags
}

// expList returns the list of lower-cased experiment names for
// experiments that differ from base. base may be nil to indicate no
// experiments. If all is true, then include all experiment flags,
// regardless of base.
func expList(exp, base *goexperiment.Flags, all bool) []string {
	var list []string
	rv := reflect.ValueOf(exp).Elem()
	var rBase reflect.Value
	if base != nil {
		rBase = reflect.ValueOf(base).Elem()
	}
	rt := rv.Type()
	for i := 0; i < rt.NumField(); i++ {
		name := strings.ToLower(rt.Field(i).Name)
		val := rv.Field(i).Bool()
		baseVal := false
		if base != nil {
			baseVal = rBase.Field(i).Bool()
		}
		if all || val != baseVal {
			if val {
				list = append(list, name)
			} else {
				list = append(list, "no"+name)
			}
		}
	}
	return list
}

// GOEXPERIMENT is a comma-separated list of enabled or disabled
// experiments that differ from the baseline experiment configuration.
// GOEXPERIMENT is exactly what a user would set on the command line
// to get the set of enabled experiments.
func GOEXPERIMENT() string {
	return strings.Join(expList(&Experiment, &experimentBaseline, false), ",")
}

// EnabledExperiments returns a list of enabled experiments, as
// lower-cased experiment names.
func EnabledExperiments() []string {
	return expList(&Experiment, nil, false)
}

// AllExperiments returns a list of all experiment settings.
// Disabled experiments appear in the list prefixed by "no".
func AllExperiments() []string {
	return expList(&Experiment, nil, true)
}

// UpdateExperiments updates the Experiment global based on a new GOARCH value.
// This is only required for cmd/go, which can change GOARCH after
// program startup due to use of "go env -w".
func UpdateExperiments(goarch string) {
	Experiment = parseExperiments(goarch)
}
