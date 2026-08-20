// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// simdgen is an experiment in generating Go <-> asm SIMD mappings.
//
// Usage: simdgen [-arch=amd64|arm64] [-xedPath=path] [-arm64Path=path] [-svePath=path] [-q=query] input.yaml...
//
// The external data inputs (XED data for amd64 or ARM64 ISA XML specs for arm64 and sve)
// are resolved automatically via search paths including standard environment variables
// ($XEDPATH or $ARM64_ISA_PATH), local directories populated by fetch-xed.sh /
// fetch-arm64.sh (e.g. ../extern/), or default $HOME locations. Explicit -xedPath
// or -arm64Path flags may be supplied to override search paths.
//
// If arch is amd64, one of the inputs is a sum of op-code definitions generated
// from the Intel XED data (which can be downloaded via fetch-xed.sh).
//
// If arch is arm64, one of the inputs is a set of NEON (advsimd) instruction
// definitions parsed from ARM64 ISA XML files (which can be downloaded via
// fetch-arm64.sh).
//
// Likewise, if arch is sve, one of the inputs is the set of SVE / SVE2
// instruction definitions parsed from the ARM64 ISA XML files.
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
//	go run . *.yaml
//
// To see just the definitions generated from XED, run:
//
//	go run . -arch=amd64
//
// (This works because if there's only one input, there's nothing to unify it
// with, so the result is simply itself.)
//
// To see just the definitions for VPADDQ on AMD64:
//
//	go run . -arch=amd64 -q '{asm: VPADDQ}'
//
// For VADD.S4 on ARM64:
//
//	go run . -arch arm64 -q '{asm: VADD, arrangement: "4S"}'
//
// simdgen can also generate Go definitions of SIMD mappings.
// To generate go files to the go root, run:
//
//	go run . -arch amd64 -o godefs -goroot $PATH/TO/go go_amd64.yaml categories.yaml types.yaml
//
// For ARM64:
//
//	go run . -arch arm64 -o godefs -goroot $PATH/TO/go go_arm64.yaml categories.yaml types.yaml
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
	"path"
	"path/filepath"
	"runtime/pprof"
	"slices"
	"strconv"
	"strings"
	"text/template"

	"simd/archsimd/_gen/sgutil"
	"simd/archsimd/_gen/simdgen/arm64"
	"simd/archsimd/_gen/simdgen/sve"
	"simd/archsimd/_gen/unify"

	"gopkg.in/yaml.v3"
)

var (
	flagXedPath           = sgutil.FlagXEDPath("..")
	flagArm64Path         = sgutil.FlagARM64Path("..")
	flagQ                 = flag.String("q", "", "query: read `def` as another input (skips final validation)")
	flagO                 = flag.String("o", "yaml", "output type: yaml, godefs (generate definitions into a Go source tree")
	flagGoDefRoot         = flag.String("goroot", ".", "the path to the Go dev directory that will receive the generated files")
	FlagNoDedup           = flag.Bool("nodedup", false, "disable deduplicating godefs of 2 qualifying operations from different extensions")
	FlagNoConstImmPorting = flag.Bool("noconstimmporting", false, "disable const immediate porting from op to imm operand")

	FlagArch = flag.String("arch", "", "unify with architecture definitions for `arch`\n\tif amd64, loads from -xedPath\n\tif arm64 or sve, loads from -arm64Path")

	Verbose = flag.Bool("v", false, "verbose")

	flagDebugXED   = flag.Bool("debug-xed", false, "show XED instructions")
	flagDebugUnify = flag.Bool("debug-unify", false, "print unification trace")
	flagDebugHTML  = flag.String("debug-html", "", "write unification trace to `file.html`")
	FlagReportDup  = flag.Bool("reportdup", false, "report the duplicate godefs")

	flagCPUProfile = flag.String("cpuprofile", "", "write CPU profile to `file`")
	flagMemProfile = flag.String("memprofile", "", "write memory profile to `file`")
)

const simdPackage = "simd/archsimd"

var splitPhase = phase6Rewrites

var (
	title = identity

	splitOpPkg = "cmd/compile/internal/ssa"

	splitCorePath   = "cmd/compile/internal/ssa"
	splitCorePkg    = "ssa"
	splitCorePrefix = ""

	splitConvPrefix = ""
)

var splitFuncs = template.FuncMap{
	"OpImport":   func() string { return strconv.Quote(splitOpPkg) },
	"OpPkg":      func() string { return path.Base(splitOpPkg) },
	"CoreImport": func() string { return strconv.Quote(splitCorePath) },
	"CorePkg":    func() string { return splitCorePkg },
	"ConvName":   func(name string) string { return splitConvPrefix + title(name) },
}

func identity(s string) string { return s }

func simpleTitle(s string) string { return strings.ToUpper(s[:1]) + s[1:] }

const (
	phase0Start = iota
	phase0Export
	phase1Op
	phase2Core
	phase3Compile
	phase4CoreRename
	phase5Conv
	phase6Rewrites
)

func init() {
	if splitPhase >= phase0Export {
		title = simpleTitle
	}
	if splitPhase >= phase1Op {
		splitOpPkg = "cmd/compile/internal/ssa/ssaop"
	}
	if splitPhase >= phase2Core {
		splitCorePath = "cmd/compile/internal/ssa/ssacore"
		splitCorePkg = "ssacore"
		splitCorePrefix = "ssacore."
	}
	if splitPhase >= phase4CoreRename {
		splitCorePath = "cmd/compile/internal/ssa"
		splitCorePkg = "ssa"
		splitCorePrefix = "ssa."
	}
	if splitPhase >= phase5Conv {
		splitConvPrefix = "ssa."
	}
}

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

	// Load instructions into the architecture-specific defs set.
	var defs []*unify.Value
	switch *FlagArch {
	case "":
		// No input from architecture definitions. That's fine if we're just
		// emitting generic yaml, but if we're emitting godefs we need to know
		// the arch.
		if *flagO == "godefs" {
			log.Fatalf("-o godefs requires -arch")
		}
	case "amd64":
		xedPath, err := sgutil.ResolveXEDPath(flagXedPath)
		if err != nil {
			log.Fatal(err)
		}
		defs = loadXED(xedPath)
	case "arm64":
		arm64Path, err := sgutil.ResolveARM64Path(flagArm64Path)
		if err != nil {
			log.Fatal(err)
		}
		defs, err = arm64.Load(arm64Path)
		if err != nil {
			log.Fatalf("loading ARM64 instructions: %s", err)
		}
	case "sve":
		arm64Path, err := sgutil.ResolveARM64Path(flagArm64Path)
		if err != nil {
			log.Fatal(err)
		}
		defs, err = sve.Load(arm64Path)
		if err != nil {
			log.Fatalf("loading ARM64 SVE instructions: %s", err)
		}
	default:
		log.Fatalf("-arch must be one of: amd64, arm64, sve")
	}

	var inputs []unify.Closure
	if defs != nil {
		inputs = append(inputs, unify.NewSum(defs...))
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

		base := filepath.Base(path)
		if base == "go_amd64.yaml" || base == "go_arm64.yaml" || base == "go_sve.yaml" {
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

	if !*Verbose && *FlagArch == "amd64" {
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
