// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildcfg

import (
	"fmt"
	"reflect"
	"strings"

	"internal/goexperiment"
)

// ExperimentFlags represents a set of GOEXPERIMENT flags relative to a baseline
// (platform-default) experiment configuration.
type ExperimentFlags struct {
	goexperiment.Flags
	baseline goexperiment.Flags
}

// Experiment contains the toolchain experiments enabled for the
// current build.
//
// (This is not necessarily the set of experiments the compiler itself
// was built with.)
//
// Experiment.baseline specifies the experiment flags that are enabled by
// default in the current toolchain. This is, in effect, the "control"
// configuration and any variation from this is an experiment.
var Experiment ExperimentFlags = func() ExperimentFlags {
	flags, err := ParseGOEXPERIMENT(GOOS, GOARCH, envOr("GOEXPERIMENT", defaultGOEXPERIMENT))
	if err != nil {
		Error = err
		return ExperimentFlags{}
	}
	return *flags
}()

// DefaultGOEXPERIMENT is the embedded default GOEXPERIMENT string.
// It is not guaranteed to be canonical.
const DefaultGOEXPERIMENT = defaultGOEXPERIMENT

// FramePointerEnabled enables the use of platform conventions for
// saving frame pointers.
//
// This used to be an experiment, but now it's always enabled on
// platforms that support it.
//
// Note: must agree with runtime.framepointer_enabled.
var FramePointerEnabled = GOARCH == "amd64" || GOARCH == "arm64"

// ParseGOEXPERIMENT parses a (GOOS, GOARCH, GOEXPERIMENT)
// configuration tuple and returns the enabled and baseline experiment
// flag sets.
//
// TODO(mdempsky): Move to [internal/goexperiment].
func ParseGOEXPERIMENT(goos, goarch, goexp string) (*ExperimentFlags, error) {
	// regabiSupported is set to true on platforms where register ABI is
	// supported and enabled by default.
	// regabiAlwaysOn is set to true on platforms where register ABI is
	// always on.
	var regabiSupported, regabiAlwaysOn bool
	switch goarch {
	case "amd64", "arm64", "loong64", "ppc64le", "ppc64", "riscv64":
		regabiAlwaysOn = true
		regabiSupported = true
	}

	// Older versions (anything before V16) of dsymutil don't handle
	// the .debug_rnglists section in DWARF5. See
	// https://github.com/golang/go/issues/26379#issuecomment-2677068742
	// for more context. This disables all DWARF5 on mac, which is not
	// ideal (would be better to disable just for cases where we know
	// the build will use external linking). In the GOOS=aix case, the
	// XCOFF format (as far as can be determined) doesn't seem to
	// support the necessary section subtypes for DWARF-specific
	// things like .debug_addr (needed for DWARF 5).
	dwarf5Supported := (goos != "darwin" && goos != "ios" && goos != "aix")

	baseline := goexperiment.Flags{
		RegabiWrappers:        regabiSupported,
		RegabiArgs:            regabiSupported,
		Dwarf5:                dwarf5Supported,
		RandomizedHeapBase64:  true,
		SizeSpecializedMalloc: true,
		GreenTeaGC:            true,
	}

	// Start with the statically enabled set of experiments.
	flags := &ExperimentFlags{
		Flags:    baseline,
		baseline: baseline,
	}

	// Pick up any changes to the baseline configuration from the
	// GOEXPERIMENT environment. This can be set at make.bash time
	// and overridden at build time.
	if goexp != "" {
		// Create a map of known experiment names.
		names := make(map[string]func(bool))
		rv := reflect.ValueOf(&flags.Flags).Elem()
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
			flags.RegabiArgs = v
		}

		// Parse names.
		for f := range strings.SplitSeq(goexp, ",") {
			if f == "" {
				continue
			}
			if f == "none" {
				// GOEXPERIMENT=none disables all experiment flags.
				// This is used by cmd/dist, which doesn't know how
				// to build with any experiment flags.
				flags.Flags = goexperiment.Flags{}
				continue
			}
			val := true
			if strings.HasPrefix(f, "no") {
				f, val = f[2:], false
			}
			set, ok := names[f]
			if !ok {
				return nil, fmt.Errorf("unknown GOEXPERIMENT %s", f)
			}
			set(val)
		}
	}

	if regabiAlwaysOn {
		flags.RegabiWrappers = true
		flags.RegabiArgs = true
	}
	// regabi is only supported on amd64, arm64, loong64, riscv64, ppc64 and ppc64le.
	if !regabiSupported {
		flags.RegabiWrappers = false
		flags.RegabiArgs = false
	}
	// Check regabi dependencies.
	if flags.RegabiArgs && !flags.RegabiWrappers {
		return nil, fmt.Errorf("GOEXPERIMENT regabiargs requires regabiwrappers")
	}
	return flags, nil
}

// String returns the canonical GOEXPERIMENT string to enable this experiment
// configuration. (Experiments in the same state as in the baseline are elided.)
func (exp *ExperimentFlags) String() string {
	return strings.Join(expList(&exp.Flags, &exp.baseline, false), ",")
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

// Enabled returns a list of enabled experiments, as
// lower-cased experiment names.
func (exp *ExperimentFlags) Enabled() []string {
	return expList(&exp.Flags, nil, false)
}

// All returns a list of all experiment settings.
// Disabled experiments appear in the list prefixed by "no".
func (exp *ExperimentFlags) All() []string {
	return expList(&exp.Flags, nil, true)
}
