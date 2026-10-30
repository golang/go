// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gotestchecker is the driver for analysis in go test.
//
// The go command synthesizes a main package, then builds an
// executable called the "analysis tool" or "tool". It is exempt from
// the usual rules on importing internal packages.
//
// The analysis tool's command-line interface, described below, is
// tightly coupled to the go command; it is unstable and may change at
// any time.
//
// The generated main file calls [Main] with the values of the
// variables referenced by its analysis configuration (analyzers and
// suites, such as the vet suite of [internal/analysis/suite/vet]),
// then executes the tool in one of two modes:
//
//	tool -describe=FILE   # write analyzer metadata (JSON) to FILE
//	tool REQUEST          # analyze the package described by REQUEST (JSON)
//
// The result of the -describe query lets the go command plan the
// analysis part of the build: from the metadata it learns the
// Requires graph of the analyzers, and which of them use facts, so
// that it materializes facts only for the analyzers that need them,
// and only once per (package, analyzer), regardless of how many root
// analyzers require that analyzer.
//
// Within the tool, analyzers are identified by pointer. In the
// protocol between go command and the tool (analyze requests,
// metadata and pass files), they are identified by their index in the
// metadata (see [protocol.Metadata]), which is deterministic for a
// given tool.
//
// The sequence of events is:
//
//  1. The tool is linked with the variables of the references that
//     survive the go command's static stage: positive references,
//     after textual negation of whole variables, and negations by
//     reference (whose variables are needed to negate by pointer).
//
//  2. The -describe query expands each suite into its members,
//     giving the index of each member and of each analyzer in its
//     Requires closure. The go command then performs the dynamic
//     stage: it selects suite members, applies negations by pointer
//     (=index) and by Analyzer.Name, and plans which analyzers each
//     package needs. So a REQUEST never mentions references, suites,
//     or negations: it names analyzers only by index, in
//     Outputs (the analyzers whose results the go command lacks),
//     Report (those whose diagnostics are reported), and
//     Facts (the fact-using analyzers in the Requires closure of Outputs).
//
//  3. The go command executes the tool for each package, like a build.
//     The tool runs the Requires graph of all needed analyzers
//     and writes a "pass" file for needed result specified by Outputs.
//
// The protocol message types (analyze requests, metadata, and pass
// files) are defined by package
// [internal/analysis/gotestchecker/protocol], which the go command
// also imports.
package gotestchecker

import (
	"flag"
	"fmt"
	"log"
	"os"
)

// A Var is an analyzer or suite variable linked into the tool: its
// symbol, the qualified name "pkgpath.Var" (never with a member list),
// and its value, which should be an analyzer (*analysis.Analyzer) or
// a suite ([]*analysis.Analyzer). Values of other types are reported
// as errors by the metadata query.
type Var struct {
	Symbol string
	Value  any
}

// Main is the main function of a go test analysis tool.
// The go command passes the variables in order of Symbol, with no
// duplicates.
func Main(vars []Var) {
	log.SetFlags(0)
	log.SetPrefix("gotestchecker: ")

	// We use a private flag set, so that the flags
	// of analyzer packages cannot interfere.
	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	describe := flags.String("describe", "", "write analyzer metadata (JSON) to the named file")
	flags.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %[1]s -describe=FILE | %[1]s REQUEST\n", os.Args[0])
		os.Exit(2)
	}
	flags.Parse(os.Args[1:])

	if *describe != "" {
		if flags.NArg() != 0 {
			flags.Usage()
		}
		if err := writeMetadata(*describe, vars); err != nil {
			log.Fatal(err)
		}
		return
	}

	if flags.NArg() != 1 {
		flags.Usage()
	}
	if err := run(flags.Arg(0), vars); err != nil {
		log.Fatal(err)
	}
}
