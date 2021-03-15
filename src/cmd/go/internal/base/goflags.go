// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"flag"
	"fmt"
	"runtime"
	"strings"

	"cmd/go/internal/cfg"
)

var goflags []string // cached $GOFLAGS list; can be -x or --x form

// GOFLAGS returns the flags from $GOFLAGS.
// The list can be assumed to contain one string per flag,
// with each string either beginning with -name or --name.
func GOFLAGS() []string {
	InitGOFLAGS()
	return goflags
}

// InitGOFLAGS initializes the goflags list from $GOFLAGS.
// If goflags is already initialized, it does nothing.
func InitGOFLAGS() {
	if goflags != nil { // already initialized
		return
	}

	goflags = strings.Fields(cfg.Getenv("GOFLAGS"))
	if len(goflags) == 0 {
		// nothing to do; avoid work on later InitGOFLAGS call
		goflags = []string{}
		return
	}

	// Ignore bad flag in go env and go bug, because
	// they are what people reach for when debugging
	// a problem, and maybe they're debugging GOFLAGS.
	// (Both will show the GOFLAGS setting if let succeed.)
	hideErrors := cfg.CmdName == "env" || cfg.CmdName == "bug"

	// Each of the words returned by strings.Fields must be its own flag.
	// To set flag arguments use -x=value instead of -x value.
	// For boolean flags, -x is fine instead of -x=true.
	for _, f := range goflags {
		// Check that every flag looks like -x --x -x=value or --x=value.
		if !strings.HasPrefix(f, "-") || f == "-" || f == "--" || strings.HasPrefix(f, "---") || strings.HasPrefix(f, "-=") || strings.HasPrefix(f, "--=") {
			if hideErrors {
				continue
			}
			Fatalf("go: parsing $GOFLAGS: non-flag %q", f)
		}

		name := f[1:]
		if name[0] == '-' {
			name = name[1:]
		}
		if i := strings.Index(name, "="); i >= 0 {
			name = name[:i]
		}
		if !hasFlag(Go, name) {
			if hideErrors {
				continue
			}
			Fatalf("go: parsing $GOFLAGS: unknown flag -%s", name)
		}
	}
}

// boolFlag is the optional interface for flag.Value known to the flag package.
// (It is not clear why package flag does not export this interface.)
type boolFlag interface {
	flag.Value
	IsBoolFlag() bool
}

// SetFromGOFLAGS sets the flags in the given flag set using settings in $GOFLAGS.
func SetFromGOFLAGS(flags *flag.FlagSet) {
	InitGOFLAGS()

	// This loop is similar to flag.Parse except that it ignores
	// unknown flags found in goflags, so that setting, say, GOFLAGS=-ldflags=-w
	// does not break commands that don't have a -ldflags.
	// It also adjusts the output to be clear that the reported problem is from $GOFLAGS.
	where := "$GOFLAGS"
	if runtime.GOOS == "windows" {
		where = "%GOFLAGS%"
	}
	for _, goflag := range goflags {
		name, value, hasValue := goflag, "", false
		// Ignore invalid flags like '=' or '=value'.
		// If it is not reported in InitGOFlags it means we don't want to report it.
		if i := strings.Index(goflag, "="); i == 0 {
			continue
		} else if i > 0 {
			name, value, hasValue = goflag[:i], goflag[i+1:], true
		}
		if strings.HasPrefix(name, "--") {
			name = name[1:]
		}
		f := flags.Lookup(name[1:])
		if f == nil {
			continue
		}

		// Use flags.Set consistently (instead of f.Value.Set) so that a subsequent
		// call to flags.Visit will correctly visit the flags that have been set.

		if fb, ok := f.Value.(boolFlag); ok && fb.IsBoolFlag() {
			if hasValue {
				if err := flags.Set(f.Name, value); err != nil {
					fmt.Fprintf(flags.Output(), "go: invalid boolean value %q for flag %s (from %s): %v\n", value, name, where, err)
					flags.Usage()
				}
			} else {
				if err := flags.Set(f.Name, "true"); err != nil {
					fmt.Fprintf(flags.Output(), "go: invalid boolean flag %s (from %s): %v\n", name, where, err)
					flags.Usage()
				}
			}
		} else {
			if !hasValue {
				fmt.Fprintf(flags.Output(), "go: flag needs an argument: %s (from %s)\n", name, where)
				flags.Usage()
			}
			if err := flags.Set(f.Name, value); err != nil {
				fmt.Fprintf(flags.Output(), "go: invalid value %q for flag %s (from %s): %v\n", value, name, where, err)
				flags.Usage()
			}
		}
	}
}

// InGOFLAGS returns whether GOFLAGS contains the given flag, such as "-mod".
func InGOFLAGS(flag string) bool {
	for _, goflag := range GOFLAGS() {
		name := goflag
		if strings.HasPrefix(name, "--") {
			name = name[1:]
		}
		if i := strings.Index(name, "="); i >= 0 {
			name = name[:i]
		}
		if name == flag {
			return true
		}
	}
	return false
}
