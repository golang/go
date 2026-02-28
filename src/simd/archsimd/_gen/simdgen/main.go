// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// simdgen is an experiment in generating Go <-> asm SIMD mappings.
//
// Usage: simdgen [-xedPath=path] [-q=query] input.yaml...
//
// If -xedPath is provided, one of the inputs is a sum of op-code definitions
// generated from the Intel XED data at path.
//
// If input YAML files are provided, each file is read as an input value. See
// [unify.Closure.UnmarshalYAML] or "go doc unify.Closure.UnmarshalYAML" for the
// format of these files.
//
// TODO: Example definitions and values.
//
// The command unifies across all of the inputs and prints all possible results
// of this unification.
//
// If the -q flag is provided, its string value is parsed as a value and treated
// as another input to unification. This is intended as a way to "query" the
// result, typically by narrowing it down to a small subset of results.
//
// Typical usage:
//
//	go run . -xedPath $XEDPATH *.yaml
//
// To see just the definitions generated from XED, run:
//
//	go run . -xedPath $XEDPATH
//
// (This works because if there's only one input, there's nothing to unify it
// with, so the result is simply itself.)
//
// To see just the definitions for VPADDQ:
//
//	go run . -xedPath $XEDPATH -q '{asm: VPADDQ}'
//
// simdgen can also generate Go definitions of SIMD mappings:
// To generate go files to the go root, run:
//
//	go run . -xedPath $XEDPATH -o godefs -goroot $PATH/TO/go go.yaml categories.yaml types.yaml
//
// types.yaml is already written, it specifies the shapes of vectors.
// categories.yaml and go.yaml contains definitions that unifies with types.yaml and XED
// data, you can find an example in ops/AddSub/.
//
// When generating Go definitions, simdgen do 3 "magic"s:
// - It splits masked operations(with op's [Masked] field set) to const and non const:
//   - One is a normal masked operation, the original
//   - The other has its mask operand's [Const] fields set to "K0".
//   - This way the user does not need to provide a separate "K0"-masked operation def.
//
// - It deduplicates intrinsic names that have duplicates:
//   - If there are two operations that shares the same signature, one is AVX512 the other
//     is before AVX512, the other will be selected.
//   - This happens often when some operations are defined both before AVX512 and after.
//     This way the user does not need to provide a separate "K0" operation for the
//     AVX512 counterpart.
//
// - It copies the op's [ConstImm] field to its immediate operand's [Const] field.
//   - This way the user does not need to provide verbose op definition while only
//     the const immediate field is different. This is useful to reduce verbosity of
//     compares with imm control predicates.
//
// These 3 magics could be disabled by enabling -nosplitmask, -nodedup or
// -noconstimmporting flags.
//
// simdgen right now only supports amd64, -arch=$OTHERARCH will trigger a fatal error.
package main

// Big TODOs:
//
// - This can produce duplicates, which can also lead to less efficient
// environment merging. Add hashing and use it for deduplication. Be careful
// about how this shows up in debug traces, since it could make things
// confusing if we don't show it happening.
//
// - Do I need Closure, Value, and Domain? It feels like I should only need two
// types.

import (
	"cmp"
	"flag"
	"fmt"
	"log"
	"maps"
	"os"
	"path/filepath"
	"runtime/pprof"
	"slices"
	"strings"

	"simd/archsimd/_gen/unify"

	"gopkg.in/yaml.v3"
)

var (
	xedPath               = flag.String("xedPath", "", "load XED datafiles from `path`")
	flagQ                 = flag.String("q", "", "query: read `def` as another input (skips final validation)")
	flagO                 = flag.String("o", "yaml", "output type: yaml, godefs (generate definitions into a Go source tree")
	flagGoDefRoot         = flag.String("goroot", ".", "the path to the Go dev directory that will receive the generated files")
	FlagNoDedup           = flag.Bool("nodedup", false, "disable deduplicating godefs of 2 qualifying operations from different extensions")
	FlagNoConstImmPorting = flag.Bool("noconstimmporting", false, "disable const immediate porting from op to imm operand")
	FlagArch              = flag.String("arch", "amd64", "the target architecture")

	Verbose = flag.Bool("v", false, "verbose")

	flagDebugXED   = flag.Bool("debug-xed", false, "show XED instructions")
	flagDebugUnify = flag.Bool("debug-unify", false, "print unification trace")
	flagDebugHTML  = flag.String("debug-html", "", "write unification trace to `file.html`")
	FlagReportDup  = flag.Bool("reportdup", false, "report the duplicate godefs")

	flagCPUProfile = flag.String("cpuprofile", "", "write CPU profile to `file`")
	flagMemProfile = flag.String("memprofile", "", "write memory profile to `file`")
)

const simdPackage = "simd/archsimd"

func main() {
	flag.Parse()

	if *flagCPUProfile != "" {
		f, err := os.Create(*flagCPUProfile)
		if err != nil {
			log.Fatalf("-cpuprofile: %s", err)
		}
		defer f.Close()
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	if *flagMemProfile != "" {
		f, err := os.Create(*flagMemProfile)
		if err != nil {
			log.Fatalf("-memprofile: %s", err)
		}
		defer func() {
			pprof.WriteHeapProfile(f)
			f.Close()
		}()
	}

	var inputs []unify.Closure

	if *FlagArch != "amd64" {
		log.Fatalf("simdgen only supports amd64")
	}

	// Load XED into a defs set.
	if *xedPath != "" {
		xedDefs := loadXED(*xedPath)
		inputs = append(inputs, unify.NewSum(xedDefs...))
	}

	// Load query.
	if *flagQ != "" {
		r := strings.NewReader(*flagQ)
		def, err := unify.Read(r, "<query>", unify.ReadOpts{})
		if err != nil {
			log.Fatalf("parsing -q: %s", err)
		}
		inputs = append(inputs, def)
	}

	// Load defs files.
	must := make(map[*unify.Value]struct{})
	for _, path := range flag.Args() {
		defs, err := unify.ReadFile(path, unify.ReadOpts{})
		if err != nil {
			log.Fatal(err)
		}
		inputs = append(inputs, defs)

		if filepath.Base(path) == "go.yaml" {
			// These must all be used in the final result
			for def := range defs.Summands() {
				must[def] = struct{}{}
			}
		}
	}

	// Prepare for unification
	if *flagDebugUnify {
		unify.Debug.UnifyLog = os.Stderr
	}
	if *flagDebugHTML != "" {
		f, err := os.Create(*flagDebugHTML)
		if err != nil {
			log.Fatal(err)
		}
		unify.Debug.HTML = f
		defer f.Close()
	}

	// Unify!
	unified, err := unify.Unify(inputs...)
	if err != nil {
		log.Fatal(err)
	}

	// Validate results.
	//
	// Don't validate if this is a command-line query because that tends to
	// eliminate lots of required defs and is used in cases where maybe defs
	// aren't enumerable anyway.
	if *flagQ == "" && len(must) > 0 {
		validate(unified, must)
	}

	// Print results.
	switch *flagO {
	case "yaml":
		// Produce a result that looks like encoding a slice, but stream it.
		fmt.Println("!sum")
		var val1 [1]*unify.Value
		for val := range unified.All() {
			val1[0] = val
			// We have to make a new encoder each time or it'll print a document
			// separator between each object.
			enc := yaml.NewEncoder(os.Stdout)
			if err := enc.Encode(val1); err != nil {
				log.Fatal(err)
			}
			enc.Close()
		}
	case "godefs":
		if err := writeGoDefs(*flagGoDefRoot, unified); err != nil {
			log.Fatalf("Failed writing godefs: %+v", err)
		}
	}

	if !*Verbose && *xedPath != "" {
		if operandRemarks == 0 {
			fmt.Fprintf(os.Stderr, "XED decoding generated no errors, which is unusual.\n")
		} else {
			fmt.Fprintf(os.Stderr, "XED decoding generated %d \"errors\" which is not cause for alarm, use -v for details.\n", operandRemarks)
		}
	}
}

func validate(cl unify.Closure, required map[*unify.Value]struct{}) {
	// Validate that:
	// 1. All final defs are exact
	// 2. All required defs are used
	for def := range cl.All() {
		if _, ok := def.Domain.(unify.Def); !ok {
			fmt.Fprintf(os.Stderr, "%s: expected Def, got %T\n", def.PosString(), def.Domain)
			continue
		}

		if !def.Exact() {
			fmt.Fprintf(os.Stderr, "%s: def not reduced to an exact value, why is %s:\n", def.PosString(), def.WhyNotExact())
			fmt.Fprintf(os.Stderr, "\t%s\n", strings.ReplaceAll(def.String(), "\n", "\n\t"))
		}

		for root := range def.Provenance() {
			delete(required, root)
		}
	}
	// Report unused defs
	unused := slices.SortedFunc(maps.Keys(required),
		func(a, b *unify.Value) int {
			return cmp.Or(
				cmp.Compare(a.Pos().Path, b.Pos().Path),
				cmp.Compare(a.Pos().Line, b.Pos().Line),
			)
		})
	for _, def := range unused {
		// TODO: Can we say anything more actionable? This is always a problem
		// with unification: if it fails, it's very hard to point a finger at
		// any particular reason. We could go back and try unifying this again
		// with each subset of the inputs (starting with individual inputs) to
		// at least say "it doesn't unify with anything in x.yaml". That's a lot
		// of work, but if we have trouble debugging unification failure it may
		// be worth it.
		fmt.Fprintf(os.Stderr, "%s: def required, but did not unify (%v)\n",
			def.PosString(), def)
	}
}
