// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package singlechecker defines the main function for an analysis
// driver with only a single analysis.
// This package makes it easy for a provider of an analysis package to
// also provide a standalone tool that runs just that analysis.
//
// For example, if example.org/findbadness is an analysis package,
// all that is needed to define a standalone tool is a file,
// example.org/findbadness/cmd/findbadness/main.go, containing:
//
//      // The findbadness command runs an analysis.
// 	package main
//
// 	import (
// 		"example.org/findbadness"
// 		"golang.org/x/tools/go/analysis/singlechecker"
// 	)
//
// 	func main() { singlechecker.Main(findbadness.Analyzer) }
//
package singlechecker

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/analysisflags"
	"golang.org/x/tools/go/analysis/internal/checker"
)

// Main is the main function for a checker command for a single analysis.
func Main(a *analysis.Analyzer) {
	log.SetFlags(0)
	log.SetPrefix(a.Name + ": ")

	analyzers := []*analysis.Analyzer{a}

	if err := analysis.Validate(analyzers); err != nil {
		log.Fatal(err)
	}

	checker.RegisterFlags()

	flag.Usage = func() {
		paras := strings.Split(a.Doc, "\n\n")
		fmt.Fprintf(os.Stderr, "%s: %s\n\n", a.Name, paras[0])
		fmt.Printf("Usage: %s [-flag] [package]\n\n", a.Name)
		if len(paras) > 1 {
			fmt.Println(strings.Join(paras[1:], "\n\n"))
		}
		fmt.Println("\nFlags:")
		flag.PrintDefaults()
	}

	analyzers = analysisflags.Parse(analyzers, false)

	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	if err := checker.Run(args, analyzers); err != nil {
		log.Fatal(err)
	}
}
