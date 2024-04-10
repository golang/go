// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Preprofile creates an intermediate representation of a pprof profile for use
// during PGO in the compiler. This transformation depends only on the profile
// itself and is thus wasteful to perform in every invocation of the compiler.
//
// Usage:
//
//	go tool preprofile [-v] [-o output] -i input
//
//

package main

import (
	"bufio"
	"cmd/internal/pgo"
	"flag"
	"fmt"
	"log"
	"os"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go tool preprofile [-v] [-o output] -i input\n\n")
	flag.PrintDefaults()
	os.Exit(2)
}

var (
	output  = flag.String("o", "", "output file path")
	input   = flag.String("i", "", "input pprof file path")
)

func preprocess(profileFile string, outputFile string) error {
	f, err := os.Open(profileFile)
	if err != nil {
		return fmt.Errorf("error opening profile: %w", err)
	}
	defer f.Close()

	r := bufio.NewReader(f)
	d, err := pgo.FromPProf(r)
	if err != nil {
		return fmt.Errorf("error parsing profile: %w", err)
	}

	var out *os.File
	if outputFile == "" {
		out = os.Stdout
	} else {
		out, err = os.Create(outputFile)
		if err != nil {
			return fmt.Errorf("error creating output file: %w", err)
		}
		defer out.Close()
	}

	w := bufio.NewWriter(out)
	if _, err := d.WriteTo(w); err != nil {
		return fmt.Errorf("error writing output file: %w", err)
	}

	return nil
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("preprofile: ")

	flag.Usage = usage
	flag.Parse()
	if *input == "" {
		log.Print("Input pprof path required (-i)")
		usage()
	}

	if err := preprocess(*input, *output); err != nil {
		log.Fatal(err)
	}
}
