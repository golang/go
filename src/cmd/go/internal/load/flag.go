// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"cmd/go/internal/base"
	"cmd/go/internal/str"
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
	present bool
	values  []ppfValue
}

// A ppfValue is a single <pattern>=<flags> per-package flag value.
type ppfValue struct {
	match func(*Package) bool // compiled pattern
	flags []string
}

// Set is called each time the flag is encountered on the command line.
func (f *PerPackageFlag) Set(v string) error {
	return f.set(v, base.Cwd)
}

// set is the implementation of Set, taking a cwd (current working directory) for easier testing.
func (f *PerPackageFlag) set(v, cwd string) error {
	f.present = true
	match := func(p *Package) bool { return p.Internal.CmdlinePkg || p.Internal.CmdlineFiles } // default predicate with no pattern
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
		pattern := strings.TrimSpace(v[:i])
		match = MatchPackage(pattern, cwd)
		v = v[i+1:]
	}
	flags, err := str.SplitQuotedFields(v)
	if err != nil {
		return err
	}
	if flags == nil {
		flags = []string{}
	}
	f.values = append(f.values, ppfValue{match, flags})
	return nil
}

// String is required to implement flag.Value.
// It is not used, because cmd/go never calls flag.PrintDefaults.
func (f *PerPackageFlag) String() string { return "<PerPackageFlag>" }

// Present reports whether the flag appeared on the command line.
func (f *PerPackageFlag) Present() bool {
	return f.present
}

// For returns the flags to use for the given package.
func (f *PerPackageFlag) For(p *Package) []string {
	flags := []string{}
	for _, v := range f.values {
		if v.match(p) {
			flags = v.flags
		}
	}
	return flags
}
