// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cmdflag handles flag processing common to several go tools.
package cmdflag

import (
	"errors"
	"flag"
	"fmt"
	"strings"
)

// The flag handling part of go commands such as test is large and distracting.
// We can't use the standard flag package because some of the flags from
// our command line are for us, and some are for the binary we're running,
// and some are for both.

// ErrFlagTerminator indicates the distinguished token "--", which causes the
// flag package to treat all subsequent arguments as non-flags.
var ErrFlagTerminator = errors.New("flag terminator")

// A FlagNotDefinedError indicates a flag-like argument that does not correspond
// to any registered flag in a FlagSet.
type FlagNotDefinedError struct {
	RawArg   string // the original argument, like --foo or -foo=value
	Name     string
	HasValue bool   // is this the -foo=value or --foo=value form?
	Value    string // only provided if HasValue is true
}

func (e FlagNotDefinedError) Error() string {
	return fmt.Sprintf("flag provided but not defined: -%s", e.Name)
}

// A NonFlagError indicates an argument that is not a syntactically-valid flag.
type NonFlagError struct {
	RawArg string
}

func (e NonFlagError) Error() string {
	return fmt.Sprintf("not a flag: %q", e.RawArg)
}

// ParseOne sees if args[0] is present in the given flag set and if so,
// sets its value and returns the flag along with the remaining (unused) arguments.
//
// ParseOne always returns either a non-nil Flag or a non-nil error,
// and always consumes at least one argument (even on error).
//
// Unlike (*flag.FlagSet).Parse, ParseOne does not log its own errors.
func ParseOne(fs *flag.FlagSet, args []string) (f *flag.Flag, remainingArgs []string, err error) {
	// This function is loosely derived from (*flag.FlagSet).parseOne.

	raw, args := args[0], args[1:]
	arg := raw
	if strings.HasPrefix(arg, "--") {
		if arg == "--" {
			return nil, args, ErrFlagTerminator
		}
		arg = arg[1:] // reduce two minuses to one
	}

	switch arg {
	case "-?", "-h", "-help":
		return nil, args, flag.ErrHelp
	}
	if len(arg) < 2 || arg[0] != '-' || arg[1] == '-' || arg[1] == '=' {
		return nil, args, NonFlagError{RawArg: raw}
	}

	name, value, hasValue := strings.Cut(arg[1:], "=")

	f = fs.Lookup(name)
	if f == nil {
		return nil, args, FlagNotDefinedError{
			RawArg:   raw,
			Name:     name,
			HasValue: hasValue,
			Value:    value,
		}
	}

	// Use fs.Set instead of f.Value.Set below so that any subsequent call to
	// fs.Visit will correctly visit the flags that have been set.

	failf := func(format string, a ...any) (*flag.Flag, []string, error) {
		return f, args, fmt.Errorf(format, a...)
	}

	if fv, ok := f.Value.(boolFlag); ok && fv.IsBoolFlag() { // special case: doesn't need an arg
		if hasValue {
			if err := fs.Set(name, value); err != nil {
				return failf("invalid boolean value %q for -%s: %v", value, name, err)
			}
		} else {
			if err := fs.Set(name, "true"); err != nil {
				return failf("invalid boolean flag %s: %v", name, err)
			}
		}
	} else {
		// It must have a value, which might be the next argument.
		if !hasValue && len(args) > 0 {
			// value is the next arg
			hasValue = true
			value, args = args[0], args[1:]
		}
		if !hasValue {
			return failf("flag needs an argument: -%s", name)
		}
		if err := fs.Set(name, value); err != nil {
			return failf("invalid value %q for flag -%s: %v", value, name, err)
		}
	}

	return f, args, nil
}

type boolFlag interface {
	IsBoolFlag() bool
}
