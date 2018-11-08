// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package multichecker defines the main function for an analysis driver
// with several analyzers. This package makes it easy for anyone to build
// an analysis tool containing just the analyzers they need.
package multichecker

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/analysisflags"
	"golang.org/x/tools/go/analysis/internal/checker"
)

// TODO(adonovan): document (and verify) the exit codes:
// "Vet's exit code is 2 for erroneous invocation of the tool, 1 if a
// problem was reported, and 0 otherwise. Note that the tool does not
// check every possible problem and depends on unreliable heuristics
// so it should be used as guidance only, not as a firm indicator of
// program correctness."

func Main(analyzers ...*analysis.Analyzer) {
	progname := filepath.Base(os.Args[0])
	log.SetFlags(0)
	log.SetPrefix(progname + ": ") // e.g. "vet: "

	if err := analysis.Validate(analyzers); err != nil {
		log.Fatal(err)
	}

	checker.RegisterFlags()

	analyzers = analysisflags.Parse(analyzers, true)

	args := flag.Args()
	if len(args) == 0 {
		analysisflags.PrintUsage(os.Stderr)
		fmt.Fprintf(os.Stderr, "Run '%[1]s help' for more detail,\n"+
			" or '%[1]s help name' for details and flags of a specific analyzer.\n",
			progname)
		os.Exit(1)
	}

	if args[0] == "help" {
		analysisflags.Help(progname, analyzers, args[1:])
		os.Exit(0)
	}

	if err := checker.Run(args, analyzers); err != nil {
		log.Fatal(err)
	}
}
