// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

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
var Experiment goexperiment.Flags

var defaultExpstring string // Set by package init

// FramePointerEnabled enables the use of platform conventions for
// saving frame pointers.
//
// This used to be an experiment, but now it's always enabled on
// platforms that support it.
//
// Note: must agree with runtime.framepointer_enabled.
var FramePointerEnabled = GOARCH == "amd64" || GOARCH == "arm64"

func init() {
	// Capture "default" experiments.
	defaultExpstring = Expstring()

	goexperiment := envOr("GOEXPERIMENT", defaultGOEXPERIMENT)

	// GOEXPERIMENT=none overrides all experiments enabled at dist time.
	if goexperiment != "none" {
		// Create a map of known experiment names.
		names := make(map[string]reflect.Value)
		rv := reflect.ValueOf(&Experiment).Elem()
		rt := rv.Type()
		for i := 0; i < rt.NumField(); i++ {
			field := rv.Field(i)
			names[strings.ToLower(rt.Field(i).Name)] = field
		}

		// Parse names.
		for _, f := range strings.Split(goexperiment, ",") {
			if f == "" {
				continue
			}
			val := true
			if strings.HasPrefix(f, "no") {
				f, val = f[2:], false
			}
			field, ok := names[f]
			if !ok {
				fmt.Printf("unknown experiment %s\n", f)
				os.Exit(2)
			}
			field.SetBool(val)
		}
	}

	// regabi is only supported on amd64.
	if GOARCH != "amd64" {
		Experiment.Regabi = false
		Experiment.RegabiWrappers = false
		Experiment.RegabiG = false
		Experiment.RegabiReflect = false
		Experiment.RegabiDefer = false
		Experiment.RegabiArgs = false
	}
	// Setting regabi sets working sub-experiments.
	if Experiment.Regabi {
		Experiment.RegabiWrappers = true
		Experiment.RegabiG = true
		Experiment.RegabiReflect = true
		Experiment.RegabiDefer = true
		// Not ready yet:
		//Experiment.RegabiArgs = true
	}
	// Check regabi dependencies.
	if Experiment.RegabiG && !Experiment.RegabiWrappers {
		panic("GOEXPERIMENT regabig requires regabiwrappers")
	}
	if Experiment.RegabiArgs && !(Experiment.RegabiWrappers && Experiment.RegabiG && Experiment.RegabiReflect && Experiment.RegabiDefer) {
		panic("GOEXPERIMENT regabiargs requires regabiwrappers,regabig,regabireflect,regabidefer")
	}
}

// expList returns the list of enabled GOEXPERIMENTs names.
func expList(flags *goexperiment.Flags) []string {
	var list []string
	rv := reflect.ValueOf(&Experiment).Elem()
	rt := rv.Type()
	for i := 0; i < rt.NumField(); i++ {
		val := rv.Field(i).Bool()
		if val {
			field := rt.Field(i)
			list = append(list, strings.ToLower(field.Name))
		}
	}
	return list
}

// Expstring returns the GOEXPERIMENT string that should appear in Go
// version signatures. This always starts with "X:".
func Expstring() string {
	list := expList(&Experiment)
	if len(list) == 0 {
		return "X:none"
	}
	return "X:" + strings.Join(list, ",")
}

// GOEXPERIMENT returns a comma-separated list of enabled experiments.
// This is derived from the GOEXPERIMENT environment variable if set,
// or the value of GOEXPERIMENT when make.bash was run if not.
func GOEXPERIMENT() string {
	return strings.Join(expList(&Experiment), ",")
}

// EnabledExperiments returns a list of enabled experiments, as
// lower-cased experiment names.
func EnabledExperiments() []string {
	return expList(&Experiment)
}
