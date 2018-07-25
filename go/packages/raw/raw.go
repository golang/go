// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package raw is the experimental API to raw package information.

NOTE: THIS PACKAGE IS NOT YET READY FOR WIDESPREAD USE:
 - The interface is not yet stable.
 - We reserve the right to add and remove fields and change the command line args.
 - This may remain unstable even after the higher level x/tools/go/packages API is stable.

This package is used by x/tools/go/packages to provide the low level raw
information about file layout.
It should not be needed unless you are attempting to implement a new source of
data for the packages API, all tools should interact only with the packages API.
*/
package raw

import (
	"flag"
)

// Results describes the results of a load operation.
type Results struct {
	// Roots is the set of package identifiers that directly matched the patterns.
	Roots []string
	// Error is an error message if the query failed for some reason.
	Error string `json:",omitempty"`
}

// Package is the raw serialized form of a packages.Package
type Package struct {
	// ID is a unique identifier for a package,
	// in a syntax provided by the underlying build system.
	//
	// Because the syntax varies based on the build system,
	// clients should treat IDs as opaque and not attempt to
	// interpret them.
	ID string

	// Name is the package name as it appears in the package source code.
	Name string `json:",omitempty"`

	// This is the package path as used by the types package.
	// This is used to map entries in the export data back to the package they
	// come from.
	PkgPath string `json:",omitempty"`

	// Imports maps import paths appearing in the package's Go source files
	// to corresponding package identifiers.
	Imports map[string]string `json:",omitempty"`

	// Export is the absolute path to a file containing the export data for the
	// package.
	Export string `json:",omitempty"`

	// GoFiles lists the absolute file paths of the package's Go source files.
	GoFiles []string `json:",omitempty"`

	// CompiledGoFiles lists the absolute file paths of the package's source
	// files that were handed to the compiler.
	// This is allowed to be different to GoFiles in the presence of files that
	// were automatically modified or processed before compilation.
	CompiledGoFiles []string `json:",omitempty"`

	// OtherFiles lists the absolute file paths of the package's non-Go source
	// files, including assembly, C, C++, Fortran, Objective-C, SWIG, and so on.
	OtherFiles []string `json:",omitempty"`
}

// Config specifies details about what raw package information is needed
// and how the underlying build tool should load package data.
type Config struct {
	// Dir is the directory in which to run the build system tool
	// that provides information about the packages.
	// If Dir is empty, the tool is run in the current directory.
	Dir string

	// Env is the environment to use when invoking the build system tool.
	// If Env is nil, the current environment is used.
	// Like in os/exec's Cmd, only the last value in the slice for
	// each environment key is used. To specify the setting of only
	// a few variables, append to the current environment, as in:
	//
	//	opt.Env = append(os.Environ(), "GOOS=plan9", "GOARCH=386")
	//
	Env []string

	// Flags is a list of command-line flags to be passed through to
	// the underlying query tool.
	Flags []string

	// Export controls whether the raw packages must contain the export
	// data file.
	Export bool

	// If Tests is set, the loader includes not just the packages
	// matching a particular pattern but also any related test packages,
	// including test-only variants of the package and the test executable.
	//
	// For example, when using the go command, loading "fmt" with Tests=true
	// returns four packages, with IDs "fmt" (the standard package),
	// "fmt [fmt.test]" (the package as compiled for the test),
	// "fmt_test" (the test functions from source files in package fmt_test),
	// and "fmt.test" (the test binary).
	//
	// In build systems with explicit names for tests,
	// setting Tests may have no effect.
	Tests bool

	// If Deps is set, the loader will include the full dependency graph.
	// Packages that are only in the results because of Deps will have DepOnly
	// set on them.
	Deps bool
}

// AddFlags adds the standard flags used to set a Config to the supplied flag set.
// This is used by implementations of the external raw package binary to correctly
// interpret the flags passed from the config.
func (cfg *Config) AddFlags(flags *flag.FlagSet) {
	flags.BoolVar(&cfg.Deps, "deps", false, "include all dependencies")
	flags.BoolVar(&cfg.Tests, "test", false, "include all test packages")
	flags.BoolVar(&cfg.Export, "export", false, "include export data files")
	flags.Var(extraFlags{cfg}, "flags", "extra flags to pass to the underlying command")
}

// extraFlags collects all occurrences of --flags into a single array
// We do this because it's much easier than escaping joining and splitting
// the extra flags that must be passed across the boundary unmodified
type extraFlags struct {
	cfg *Config
}

func (e extraFlags) String() string {
	return ""
}

func (e extraFlags) Set(value string) error {
	e.cfg.Flags = append(e.cfg.Flags, value)
	return nil
}
