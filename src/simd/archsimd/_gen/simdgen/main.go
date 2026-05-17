// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// simdgen is an experiment in generating Go <-> asm SIMD mappings.
//
// Usage: simdgen [-xedPath=path | -arm64Path=path] [-q=query] input.yaml...
//
// Only one of -xedPath or -arm64Path may be specified.
//
// If -xedPath is provided, one of the inputs is a sum of op-code definitions
// generated from the Intel XED data at path.
//
// If -arm64Path is provided, one of the inputs is a set of instruction
// definitions parsed from ARM64 ISA XML files at path (obtained from
// https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-A64-ISA/ISA_A64/ISA_A64_xml_A_profile-2025-12.tar.gz).
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
// For VADD.S4 on ARM64:
//
//	go run . -arm64Path $ARM64_ISA_PATH -o yaml -q '{asm: VADD, arrangement: "4S"}'
//
// simdgen can also generate Go definitions of SIMD mappings:
// To generate go files to the go root, run:
//
//	go run . -xedPath $XEDPATH -o godefs -goroot $PATH/TO/go go_amd64.yaml categories.yaml types.yaml
//
// For ARM64:
//
//	go run . -arm64Path $ARM64_ISA_PATH -o godefs -goroot $PATH/TO/go go_arm64.yaml categories.yaml types.yaml
//
// types.yaml is already written, it specifies the shapes of vectors.
// categories.yaml and go_<arch>.yaml contain definitions that unify with types.yaml and
// XED/ARM64 ISA data, you can find an example in ops/AddSub/.
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
// simdgen supports amd64 and arm64 architectures.
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

	"_gen/simdgen/arm64"
	"_gen/unify"

	"gopkg.in/yaml.v3"
)

var (
	xedPath               = flag.String("xedPath", "", "load XED datafiles from `path`")
	arm64Path             = flag.String("arm64Path", "", "load ARM64 instruction xml definitions from `path`")
	flagQ                 = flag.String("q", "", "query: read `def` as another input (skips final validation)")
	flagO                 = flag.String("o", "yaml", "output type: yaml, godefs (generate definitions into a Go source tree")
	flagGoDefRoot         = flag.String("goroot", ".", "the path to the Go dev directory that will receive the generated files")
	FlagNoDedup           = flag.Bool("nodedup", false, "disable deduplicating godefs of 2 qualifying operations from different extensions")
	FlagNoConstImmPorting = flag.Bool("noconstimmporting", false, "disable const immediate porting from op to imm operand")

	// FlagArch must be pre-initialized to a bogus value because there have been initializations that depended on it
	FlagArch = flag.String("arch", "must be specified, amd64 or arm64", "the target architecture")

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

	// Default -arch to arm64 when -arm64Path is specified.
	if *arm64Path != "" && *FlagArch != "arm64" {
		if *xedPath != "" {
			log.Fatalf("both -xedPath and -arm64Path specified")
		}
		// *FlagArch = "arm64"
	}

	// Load instructions into the architecture-specific defs set.
	var defs []*unify.Value
	switch *FlagArch {
	case "amd64":
		if *xedPath != "" {
			defs = loadXED(*xedPath)
		}
	case "arm64":
		if *arm64Path != "" {
			var err error
			defs, err = arm64.Load(*arm64Path)
			if err != nil {
				log.Fatalf("loading ARM64 instructions: %s", err)
			}
		}
	default:
		log.Fatalf("simdgen only supports amd64 and arm64")
	}

	var inputs []unify.Closure
	inputs = append(inputs, unify.NewSum(defs...))

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

		base := filepath.Base(path)
		if base == "go_amd64.yaml" || base == "go_arm64.yaml" {
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

	ok := true

	// Validate results.
	//
	// Don't validate if this is a command-line query because that tends to
	// eliminate lots of required defs and is used in cases where maybe defs
	// aren't enumerable anyway.
	if *flagQ == "" && len(must) > 0 {
		ok = validate(unified, must)
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
	if !ok {
		os.Exit(1)
	}
}

func validate(cl unify.Closure, required map[*unify.Value]struct{}) bool {
	ok := true
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
		ok = false
	}
	return ok
}
