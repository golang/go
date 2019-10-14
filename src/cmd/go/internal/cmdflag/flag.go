// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cmdflag handles flag processing common to several go tools.
package cmdflag

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"cmd/go/internal/base"
)

// The flag handling part of go commands such as test is large and distracting.
// We can't use the standard flag package because some of the flags from
// our command line are for us, and some are for the binary we're running,
// and some are for both.

// Defn defines a flag we know about.
type Defn struct {
	Name       string     // Name on command line.
	BoolVar    *bool      // If it's a boolean flag, this points to it.
	Value      flag.Value // The flag.Value represented.
	PassToTest bool       // Pass to the test binary? Used only by go test.
	Present    bool       // Flag has been seen.
}

// IsBool reports whether v is a bool flag.
func IsBool(v flag.Value) bool {
	vv, ok := v.(interface {
		IsBoolFlag() bool
	})
	if ok {
		return vv.IsBoolFlag()
	}
	return false
}

// SetBool sets the addressed boolean to the value.
func SetBool(cmd string, flag *bool, value string) {
	x, err := strconv.ParseBool(value)
	if err != nil {
		SyntaxError(cmd, "illegal bool flag value "+value)
	}
	*flag = x
}

// SetInt sets the addressed integer to the value.
func SetInt(cmd string, flag *int, value string) {
	x, err := strconv.Atoi(value)
	if err != nil {
		SyntaxError(cmd, "illegal int flag value "+value)
	}
	*flag = x
}

// SyntaxError reports an argument syntax error and exits the program.
func SyntaxError(cmd, msg string) {
	fmt.Fprintf(os.Stderr, "go %s: %s\n", cmd, msg)
	if cmd == "test" {
		fmt.Fprintf(os.Stderr, `run "go help %s" or "go help testflag" for more information`+"\n", cmd)
	} else {
		fmt.Fprintf(os.Stderr, `run "go help %s" for more information`+"\n", cmd)
	}
	base.SetExitStatus(2)
	base.Exit()
}

// AddKnownFlags registers the flags in defns with base.AddKnownFlag.
func AddKnownFlags(cmd string, defns []*Defn) {
	for _, f := range defns {
		base.AddKnownFlag(f.Name)
		base.AddKnownFlag(cmd + "." + f.Name)
	}
}

// Parse sees if argument i is present in the definitions and if so,
// returns its definition, value, and whether it consumed an extra word.
// If the flag begins (cmd.Name()+".") it is ignored for the purpose of this function.
func Parse(cmd string, usage func(), defns []*Defn, args []string, i int) (f *Defn, value string, extra bool) {
	arg := args[i]
	if strings.HasPrefix(arg, "--") { // reduce two minuses to one
		arg = arg[1:]
	}
	switch arg {
	case "-?", "-h", "-help":
		usage()
	}
	if arg == "" || arg[0] != '-' {
		return
	}
	name := arg[1:]
	// If there's already a prefix such as "test.", drop it for now.
	name = strings.TrimPrefix(name, cmd+".")
	equals := strings.Index(name, "=")
	if equals >= 0 {
		value = name[equals+1:]
		name = name[:equals]
	}
	for _, f = range defns {
		if name == f.Name {
			// Booleans are special because they have modes -x, -x=true, -x=false.
			if f.BoolVar != nil || IsBool(f.Value) {
				if equals < 0 { // Otherwise, it's been set and will be verified in SetBool.
					value = "true"
				} else {
					// verify it parses
					SetBool(cmd, new(bool), value)
				}
			} else { // Non-booleans must have a value.
				extra = equals < 0
				if extra {
					if i+1 >= len(args) {
						SyntaxError(cmd, "missing argument for flag "+f.Name)
					}
					value = args[i+1]
				}
			}
			if f.Present {
				SyntaxError(cmd, f.Name+" flag may be set only once")
			}
			f.Present = true
			return
		}
	}
	f = nil
	return
}

// FindGOFLAGS extracts and returns the flags matching defns from GOFLAGS.
// Ideally the caller would mention that the flags were from GOFLAGS
// when reporting errors, but that's too hard for now.
func FindGOFLAGS(defns []*Defn) []string {
	var flags []string
	for _, flag := range base.GOFLAGS() {
		// Flags returned by base.GOFLAGS are well-formed, one of:
		//	-x
		//	--x
		//	-x=value
		//	--x=value
		if strings.HasPrefix(flag, "--") {
			flag = flag[1:]
		}
		name := flag[1:]
		if i := strings.Index(name, "="); i >= 0 {
			name = name[:i]
		}
		for _, f := range defns {
			if name == f.Name {
				flags = append(flags, flag)
				break
			}
		}
	}
	return flags
}
