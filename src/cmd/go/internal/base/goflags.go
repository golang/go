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

var (
	goflags   []string                // cached $GOFLAGS list; can be -x or --x form
	knownFlag = make(map[string]bool) // flags allowed to appear in $GOFLAGS; no leading dashes
)

// AddKnownFlag adds name to the list of known flags for use in $GOFLAGS.
func AddKnownFlag(name string) {
	knownFlag[name] = true
}

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

	// Build list of all flags for all commands.
	// If no command has that flag, then we report the problem.
	// This catches typos while still letting users record flags in GOFLAGS
	// that only apply to a subset of go commands.
	// Commands using CustomFlags can report their flag names
	// by calling AddKnownFlag instead.
	var walkFlags func(*Command)
	walkFlags = func(cmd *Command) {
		for _, sub := range cmd.Commands {
			walkFlags(sub)
		}
		cmd.Flag.VisitAll(func(f *flag.Flag) {
			knownFlag[f.Name] = true
		})
	}
	walkFlags(Go)

	// Ignore bad flag in go env and go bug, because
	// they are what people reach for when debugging
	// a problem, and maybe they're debugging GOFLAGS.
	// (Both will show the GOFLAGS setting if let succeed.)
	hideErrors := cfg.CmdName == "env" || cfg.CmdName == "bug"

	goflags = strings.Fields(cfg.Getenv("GOFLAGS"))
	if goflags == nil {
		goflags = []string{} // avoid work on later InitGOFLAGS call
	}

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
		if !knownFlag[name] {
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
		if i := strings.Index(goflag, "="); i >= 0 {
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
