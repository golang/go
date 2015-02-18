// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines the BuilderMode type and its command-line flag.

import (
	"bytes"
	"flag"
	"fmt"
)

// BuilderMode is a bitmask of options for diagnostics and checking.
type BuilderMode uint

const (
	PrintPackages        BuilderMode = 1 << iota // Print package inventory to stdout
	PrintFunctions                               // Print function SSA code to stdout
	LogSource                                    // Log source locations as SSA builder progresses
	SanityCheckFunctions                         // Perform sanity checking of function bodies
	NaiveForm                                    // Build naïve SSA form: don't replace local loads/stores with registers
	BuildSerially                                // Build packages serially, not in parallel.
	GlobalDebug                                  // Enable debug info for all packages
	BareInits                                    // Build init functions without guards or calls to dependent inits
)

const modeFlagUsage = `Options controlling the SSA builder.
The value is a sequence of zero or more of these letters:
C	perform sanity [C]hecking of the SSA form.
D	include [D]ebug info for every function.
P	print [P]ackage inventory.
F	print [F]unction SSA code.
S	log [S]ource locations as SSA builder progresses.
L	build distinct packages seria[L]ly instead of in parallel.
N	build [N]aive SSA form: don't replace local loads/stores with registers.
I	build bare [I]nit functions: no init guards or calls to dependent inits.
`

// BuilderModeFlag creates a new command line flag of type BuilderMode,
// adds it to the specified flag set, and returns it.
//
// Example:
// 	var ssabuild = BuilderModeFlag(flag.CommandLine, "ssabuild", 0)
//
func BuilderModeFlag(set *flag.FlagSet, name string, value BuilderMode) *BuilderMode {
	set.Var((*builderModeValue)(&value), name, modeFlagUsage)
	return &value
}

type builderModeValue BuilderMode // satisfies flag.Value and flag.Getter.

func (v *builderModeValue) Set(s string) error {
	var mode BuilderMode
	for _, c := range s {
		switch c {
		case 'D':
			mode |= GlobalDebug
		case 'P':
			mode |= PrintPackages
		case 'F':
			mode |= PrintFunctions
		case 'S':
			mode |= LogSource | BuildSerially
		case 'C':
			mode |= SanityCheckFunctions
		case 'N':
			mode |= NaiveForm
		case 'L':
			mode |= BuildSerially
		default:
			return fmt.Errorf("unknown BuilderMode option: %q", c)
		}
	}
	*v = builderModeValue(mode)
	return nil
}

func (v *builderModeValue) Get() interface{} { return BuilderMode(*v) }

func (v *builderModeValue) String() string {
	mode := BuilderMode(*v)
	var buf bytes.Buffer
	if mode&GlobalDebug != 0 {
		buf.WriteByte('D')
	}
	if mode&PrintPackages != 0 {
		buf.WriteByte('P')
	}
	if mode&PrintFunctions != 0 {
		buf.WriteByte('F')
	}
	if mode&LogSource != 0 {
		buf.WriteByte('S')
	}
	if mode&SanityCheckFunctions != 0 {
		buf.WriteByte('C')
	}
	if mode&NaiveForm != 0 {
		buf.WriteByte('N')
	}
	if mode&BuildSerially != 0 {
		buf.WriteByte('L')
	}
	return buf.String()
}
