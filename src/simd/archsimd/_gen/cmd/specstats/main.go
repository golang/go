// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Specstats reports progress on the internal/spec transition described in
// SPEC-TRANSITION.md. The checked-in output is archsimd/_gen/SPEC-STATS.txt:
//
//	go run ./cmd/specstats -w
//
// Every figure it prints is a fraction heading for 100% or a defect count
// heading for zero, tied to the tasks that move it. Anything that is merely
// interesting, rather than a measure of how far along we are, does not belong
// here.
//
// # How it is put together
//
// Almost every figure in the report is the same operation: count the distinct
// things at some granularity, and count how many of them are done. So the
// program builds one fact table -- [fact], a declaration as seen in one build
// configuration -- and everything downstream is a pivot over it: pick a grain
// ([fact.declKey], [fact.triple], [fact.pair], name), pick a predicate, and
// optionally group by a dimension. [tally] does the counting and [pivot] does
// the grouping, so a new figure is a few lines rather than a new loop.
//
// Like Fill, this parses rather than type-checks, so it sees every target's
// declarations regardless of build tags (SPEC-TRANSITION.md §2.4). It is a
// measurement tool and not a checker: it reports divergence and does not judge
// it. fill builds the checker, and should absorb the comparison half of this
// program rather than leave two implementations to drift.
package main

import (
	"flag"
	"log"
	"path/filepath"

	"simd/archsimd/_gen/gentools"
	"simd/archsimd/_gen/specgen"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("specstats: ")
	genOpts := gentools.RegisterFlags(nil)
	flag.Parse()

	specDir := specgen.MustFindSpecDir(genOpts.GOROOT)
	simdDir := filepath.Clean(filepath.Join(specDir, "..", ".."))
	genDir := filepath.Join(simdDir, "archsimd", "_gen")

	spec, err := specgen.Load(specDir, nil)
	if err != nil {
		log.Fatalf("loading spec: %s", err)
	}

	var files gentools.Files
	defer files.FlushOrExit()

	buf := files.NewRawFile("simd/archsimd/_gen/SPEC-STATS.txt")
	measure(loadAPI(genOpts, simdDir), spec, genDir).report(buf)
}
