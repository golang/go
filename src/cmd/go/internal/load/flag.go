// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"cmd/go/internal/base"
	"cmd/go/internal/modload"
	"cmd/internal/quoted"
	"fmt"
	"strings"
)

var (
	BuildAsmflags   PerPackageFlag // -asmflags
	BuildGcflags    PerPackageFlag // -gcflags
	BuildLdflags    PerPackageFlag // -ldflags
	BuildGccgoflags PerPackageFlag // -gccgoflags
)

// A PerPackageFlag is a command-line flag implementation (a flag.Value)
// that allows specifying different effective flags for different packages.
// See 'go help build' for more details about per-package flags.
type PerPackageFlag struct {
	raw     string
	present bool
	values  []ppfValue
}

// A ppfValue is a single <pattern>=<flags> per-package flag value.
type ppfValue struct {
	match func(*modload.State, *Package) bool // compiled pattern
	flags []string
}

// Set is called each time the flag is encountered on the command line.
func (f *PerPackageFlag) Set(v string) error {
	return f.set(v, base.Cwd())
}

// set is the implementation of Set, taking a cwd (current working directory) for easier testing.
func (f *PerPackageFlag) set(v, cwd string) error {
	f.raw = v
	f.present = true
	match := func(_ *modload.State, p *Package) bool { return p.Internal.CmdlinePkg || p.Internal.CmdlineFiles } // default predicate with no pattern
	// For backwards compatibility with earlier flag splitting, ignore spaces around flags.
	v = strings.TrimSpace(v)
	if v == "" {
		// Special case: -gcflags="" means no flags for command-line arguments
		// (overrides previous -gcflags="-whatever").
		f.values = append(f.values, ppfValue{match, []string{}})
		return nil
	}
	if !strings.HasPrefix(v, "-") {
		i := strings.Index(v, "=")
		if i < 0 {
			return fmt.Errorf("missing =<value> in <pattern>=<value>")
		}
		if i == 0 {
			return fmt.Errorf("missing <pattern> in <pattern>=<value>")
		}
		if v[0] == '\'' || v[0] == '"' {
			return fmt.Errorf("parameter may not start with quote character %c", v[0])
		}
		pattern := strings.TrimSpace(v[:i])
		match = MatchPackage(pattern, cwd)
		v = v[i+1:]
	}
	flags, err := quoted.Split(v)
	if err != nil {
		return err
	}
	if flags == nil {
		flags = []string{}
	}
	f.values = append(f.values, ppfValue{match, flags})
	return nil
}

func (f *PerPackageFlag) String() string { return f.raw }

// Present reports whether the flag appeared on the command line.
func (f *PerPackageFlag) Present() bool {
	return f.present
}

// For returns the flags to use for the given package.
//
// The module loader state is used by the matcher to know if certain
// patterns match packages within the state's MainModules.
func (f *PerPackageFlag) For(s *modload.State, p *Package) []string {
	flags := []string{}
	for _, v := range f.values {
		if v.match(s, p) {
			flags = v.flags
		}
	}
	return flags
}
