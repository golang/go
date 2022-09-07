// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/src"
	"fmt"
	"hash/crc32"
	"internal/buildcfg"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

// Compile is the main entry point for this package.
// Compile modifies f so that on return:
//   - all Values in f map to 0 or 1 assembly instructions of the target architecture
//   - the order of f.Blocks is the order to emit the Blocks
//   - the order of b.Values is the order to emit the Values in each Block
//   - f has a non-nil regAlloc field
func Compile(f *Func) {
	// TODO: debugging - set flags to control verbosity of compiler,
	// which phases to dump IR before/after, etc.
	if f.Log() {
		f.Logf("compiling %s\n", f.Name)
	}

	var rnd *rand.Rand
	if checkEnabled {
		seed := int64(crc32.ChecksumIEEE(([]byte)(f.Name))) ^ int64(checkRandSeed)
		rnd = rand.New(rand.NewSource(seed))
	}

	// hook to print function & phase if panic happens
	phaseName := "init"
	defer func() {
		if phaseName != "" {
			err := recover()
			stack := make([]byte, 16384)
			n := runtime.Stack(stack, false)
			stack = stack[:n]
			if f.HTMLWriter != nil {
				f.HTMLWriter.flushPhases()
			}
			f.Fatalf("panic during %s while compiling %s:\n\n%v\n\n%s\n", phaseName, f.Name, err, stack)
		}
	}()

	// Run all the passes
	if f.Log() {
		printFunc(f)
	}
	f.HTMLWriter.WritePhase("start", "start")
	if BuildDump[f.Name] {
		f.dumpFile("build")
	}
	if checkEnabled {
		checkFunc(f)
	}
	const logMemStats = false
	for _, p := range passes {
		if !f.Config.optimize && !p.required || p.disabled {
			continue
		}
		f.pass = &p
		phaseName = p.name
		if f.Log() {
			f.Logf("  pass %s begin\n", p.name)
		}
		// TODO: capture logging during this pass, add it to the HTML
		var mStart runtime.MemStats
		if logMemStats || p.mem {
			runtime.ReadMemStats(&mStart)
		}

		if checkEnabled && !f.scheduled {
			// Test that we don't depend on the value order, by randomizing
			// the order of values in each block. See issue 18169.
			for _, b := range f.Blocks {
				for i := 0; i < len(b.Values)-1; i++ {
					j := i + rnd.Intn(len(b.Values)-i)
					b.Values[i], b.Values[j] = b.Values[j], b.Values[i]
				}
			}
		}

		tStart := time.Now()
		p.fn(f)
		tEnd := time.Now()

		// Need something less crude than "Log the whole intermediate result".
		if f.Log() || f.HTMLWriter != nil {
			time := tEnd.Sub(tStart).Nanoseconds()
			var stats string
			if logMemStats {
				var mEnd runtime.MemStats
				runtime.ReadMemStats(&mEnd)
				nBytes := mEnd.TotalAlloc - mStart.TotalAlloc
				nAllocs := mEnd.Mallocs - mStart.Mallocs
				stats = fmt.Sprintf("[%d ns %d allocs %d bytes]", time, nAllocs, nBytes)
			} else {
				stats = fmt.Sprintf("[%d ns]", time)
			}

			if f.Log() {
				f.Logf("  pass %s end %s\n", p.name, stats)
				printFunc(f)
			}
			f.HTMLWriter.WritePhase(phaseName, fmt.Sprintf("%s <span class=\"stats\">%s</span>", phaseName, stats))
		}
		if p.time || p.mem {
			// Surround timing information w/ enough context to allow comparisons.
			time := tEnd.Sub(tStart).Nanoseconds()
			if p.time {
				f.LogStat("TIME(ns)", time)
			}
			if p.mem {
				var mEnd runtime.MemStats
				runtime.ReadMemStats(&mEnd)
				nBytes := mEnd.TotalAlloc - mStart.TotalAlloc
				nAllocs := mEnd.Mallocs - mStart.Mallocs
				f.LogStat("TIME(ns):BYTES:ALLOCS", time, nBytes, nAllocs)
			}
		}
		if p.dump != nil && p.dump[f.Name] {
			// Dump function to appropriately named file
			f.dumpFile(phaseName)
		}
		if checkEnabled {
			checkFunc(f)
		}
	}

	if f.HTMLWriter != nil {
		// Ensure we write any pending phases to the html
		f.HTMLWriter.flushPhases()
	}

	if f.ruleMatches != nil {
		var keys []string
		for key := range f.ruleMatches {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		buf := new(strings.Builder)
		fmt.Fprintf(buf, "%s: ", f.Name)
		for _, key := range keys {
			fmt.Fprintf(buf, "%s=%d ", key, f.ruleMatches[key])
		}
		fmt.Fprint(buf, "\n")
		fmt.Print(buf.String())
	}

	// Squash error printing defer
	phaseName = ""
}

// DumpFileForPhase creates a file from the function name and phase name,
// warning and returning nil if this is not possible.
func (f *Func) DumpFileForPhase(phaseName string) io.WriteCloser {
	f.dumpFileSeq++
	fname := fmt.Sprintf("%s_%02d__%s.dump", f.Name, int(f.dumpFileSeq), phaseName)
	fname = strings.Replace(fname, " ", "_", -1)
	fname = strings.Replace(fname, "/", "_", -1)
	fname = strings.Replace(fname, ":", "_", -1)

	if ssaDir := os.Getenv("GOSSADIR"); ssaDir != "" {
		fname = filepath.Join(ssaDir, fname)
	}

	fi, err := os.Create(fname)
	if err != nil {
		f.Warnl(src.NoXPos, "Unable to create after-phase dump file %s", fname)
		return nil
	}
	return fi
}

// dumpFile creates a file from the phase name and function name
// Dumping is done to files to avoid buffering huge strings before
// output.
func (f *Func) dumpFile(phaseName string) {
	fi := f.DumpFileForPhase(phaseName)
	if fi != nil {
		p := stringFuncPrinter{w: fi}
		fprintFunc(p, f)
		fi.Close()
	}
}

type pass struct {
	name     string
	fn       func(*Func)
	required bool
	disabled bool
	time     bool            // report time to run pass
	mem      bool            // report mem stats to run pass
	stats    int             // pass reports own "stats" (e.g., branches removed)
	debug    int             // pass performs some debugging. =1 should be in error-testing-friendly Warnl format.
	test     int             // pass-specific ad-hoc option, perhaps useful in development
	dump     map[string]bool // dump if function name matches
}

func (p *pass) addDump(s string) {
	if p.dump == nil {
		p.dump = make(map[string]bool)
	}
	p.dump[s] = true
}

func (p *pass) String() string {
	if p == nil {
		return "nil pass"
	}
	return p.name
}

// Run consistency checker between each phase
var (
	checkEnabled  = false
	checkRandSeed = 0
)

// Debug output
var IntrinsicsDebug int
var IntrinsicsDisable bool

var BuildDebug int
var BuildTest int
var BuildStats int
var BuildDump map[string]bool = make(map[string]bool) // names of functions to dump after initial build of ssa

var GenssaDump map[string]bool = make(map[string]bool) // names of functions to dump after ssa has been converted to asm

// PhaseOption sets the specified flag in the specified ssa phase,
// returning empty string if this was successful or a string explaining
// the error if it was not.
// A version of the phase name with "_" replaced by " " is also checked for a match.
// If the phase name begins a '~' then the rest of the underscores-replaced-with-blanks
// version is used as a regular expression to match the phase name(s).
//
// Special cases that have turned out to be useful:
//   - ssa/check/on enables checking after each phase
//   - ssa/all/time enables time reporting for all phases
//
// See gc/lex.go for dissection of the option string.
// Example uses:
//
// GO_GCFLAGS=-d=ssa/generic_cse/time,ssa/generic_cse/stats,ssa/generic_cse/debug=3 ./make.bash
//
// BOOT_GO_GCFLAGS=-d='ssa/~^.*scc$/off' GO_GCFLAGS='-d=ssa/~^.*scc$/off' ./make.bash
func PhaseOption(phase, flag string, val int, valString string) string {
	switch phase {
	case "", "help":
		lastcr := 0
		phasenames := "    check, all, build, intrinsics, genssa"
		for _, p := range passes {
			pn := strings.Replace(p.name, " ", "_", -1)
			if len(pn)+len(phasenames)-lastcr > 70 {
				phasenames += "\n    "
				lastcr = len(phasenames)
				phasenames += pn
			} else {
				phasenames += ", " + pn
			}
		}
		return `PhaseOptions usage:

    go tool compile -d=ssa/<phase>/<flag>[=<value>|<function_name>]

where:

- <phase> is one of:
` + phasenames + `

- <flag> is one of:
    on, off, debug, mem, time, test, stats, dump, seed

- <value> defaults to 1

- <function_name> is required for the "dump" flag, and specifies the
  name of function to dump after <phase>

Phase "all" supports flags "time", "mem", and "dump".
Phase "intrinsics" supports flags "on", "off", and "debug".
Phase "genssa" (assembly generation) supports the flag "dump".

If the "dump" flag is specified, the output is written on a file named
<phase>__<function_name>_<seq>.dump; otherwise it is directed to stdout.

Examples:

    -d=ssa/check/on
enables checking after each phase

	-d=ssa/check/seed=1234
enables checking after each phase, using 1234 to seed the PRNG
used for value order randomization

    -d=ssa/all/time
enables time reporting for all phases

    -d=ssa/prove/debug=2
sets debugging level to 2 in the prove pass

Be aware that when "/debug=X" is applied to a pass, some passes
will emit debug output for all functions, and other passes will
only emit debug output for functions that match the current
GOSSAFUNC value.

Multiple flags can be passed at once, by separating them with
commas. For example:

    -d=ssa/check/on,ssa/all/time
`
	}

	if phase == "check" {
		switch flag {
		case "on":
			checkEnabled = val != 0
			debugPoset = checkEnabled // also turn on advanced self-checking in prove's datastructure
			return ""
		case "off":
			checkEnabled = val == 0
			debugPoset = checkEnabled
			return ""
		case "seed":
			checkEnabled = true
			checkRandSeed = val
			debugPoset = checkEnabled
			return ""
		}
	}

	alltime := false
	allmem := false
	alldump := false
	if phase == "all" {
		switch flag {
		case "time":
			alltime = val != 0
		case "mem":
			allmem = val != 0
		case "dump":
			alldump = val != 0
			if alldump {
				BuildDump[valString] = true
				GenssaDump[valString] = true
			}
		default:
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option (expected ssa/all/{time,mem,dump=function_name})", flag, phase)
		}
	}

	if phase == "intrinsics" {
		switch flag {
		case "on":
			IntrinsicsDisable = val == 0
		case "off":
			IntrinsicsDisable = val != 0
		case "debug":
			IntrinsicsDebug = val
		default:
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option (expected ssa/intrinsics/{on,off,debug})", flag, phase)
		}
		return ""
	}
	if phase == "build" {
		switch flag {
		case "debug":
			BuildDebug = val
		case "test":
			BuildTest = val
		case "stats":
			BuildStats = val
		case "dump":
			BuildDump[valString] = true
		default:
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option (expected ssa/build/{debug,test,stats,dump=function_name})", flag, phase)
		}
		return ""
	}
	if phase == "genssa" {
		switch flag {
		case "dump":
			GenssaDump[valString] = true
		default:
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option (expected ssa/genssa/dump=function_name)", flag, phase)
		}
		return ""
	}

	underphase := strings.Replace(phase, "_", " ", -1)
	var re *regexp.Regexp
	if phase[0] == '~' {
		r, ok := regexp.Compile(underphase[1:])
		if ok != nil {
			return fmt.Sprintf("Error %s in regexp for phase %s, flag %s", ok.Error(), phase, flag)
		}
		re = r
	}
	matchedOne := false
	for i, p := range passes {
		if phase == "all" {
			p.time = alltime
			p.mem = allmem
			if alldump {
				p.addDump(valString)
			}
			passes[i] = p
			matchedOne = true
		} else if p.name == phase || p.name == underphase || re != nil && re.MatchString(p.name) {
			switch flag {
			case "on":
				p.disabled = val == 0
			case "off":
				p.disabled = val != 0
			case "time":
				p.time = val != 0
			case "mem":
				p.mem = val != 0
			case "debug":
				p.debug = val
			case "stats":
				p.stats = val
			case "test":
				p.test = val
			case "dump":
				p.addDump(valString)
			default:
				return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option", flag, phase)
			}
			if p.disabled && p.required {
				return fmt.Sprintf("Cannot disable required SSA phase %s using -d=ssa/%s debug option", phase, phase)
			}
			passes[i] = p
			matchedOne = true
		}
	}
	if matchedOne {
		return ""
	}
	return fmt.Sprintf("Did not find a phase matching %s in -d=ssa/... debug option", phase)
}

// list of passes for the compiler
var passes = [...]pass{
	// TODO: combine phielim and copyelim into a single pass?
	{name: "number lines", fn: numberLines, required: true},
	{name: "early phielim", fn: phielim},
	{name: "early copyelim", fn: copyelim},
	{name: "early deadcode", fn: deadcode}, // remove generated dead code to avoid doing pointless work during opt
	{name: "short circuit", fn: shortcircuit},
	{name: "decompose user", fn: decomposeUser, required: true},
	{name: "pre-opt deadcode", fn: deadcode},
	{name: "opt", fn: opt, required: true},               // NB: some generic rules know the name of the opt pass. TODO: split required rules and optimizing rules
	{name: "zero arg cse", fn: zcse, required: true},     // required to merge OpSB values
	{name: "opt deadcode", fn: deadcode, required: true}, // remove any blocks orphaned during opt
	{name: "generic cse", fn: cse},
	{name: "phiopt", fn: phiopt},
	{name: "gcse deadcode", fn: deadcode, required: true}, // clean out after cse and phiopt
	{name: "nilcheckelim", fn: nilcheckelim},
	{name: "prove", fn: prove},
	{name: "early fuse", fn: fuseEarly},
	{name: "decompose builtin", fn: decomposeBuiltIn, required: true},
	{name: "expand calls", fn: expandCalls, required: true},
	{name: "softfloat", fn: softfloat, required: true},
	{name: "late opt", fn: opt, required: true}, // TODO: split required rules and optimizing rules
	{name: "dead auto elim", fn: elimDeadAutosGeneric},
	{name: "generic deadcode", fn: deadcode, required: true}, // remove dead stores, which otherwise mess up store chain
	{name: "check bce", fn: checkbce},
	{name: "branchelim", fn: branchelim},
	{name: "late fuse", fn: fuseLate},
	{name: "dse", fn: dse},
	{name: "writebarrier", fn: writebarrier, required: true}, // expand write barrier ops
	{name: "insert resched checks", fn: insertLoopReschedChecks,
		disabled: !buildcfg.Experiment.PreemptibleLoops}, // insert resched checks in loops.
	{name: "lower", fn: lower, required: true},
	{name: "addressing modes", fn: addressingModes, required: false},
	{name: "lowered deadcode for cse", fn: deadcode}, // deadcode immediately before CSE avoids CSE making dead values live again
	{name: "lowered cse", fn: cse},
	{name: "elim unread autos", fn: elimUnreadAutos},
	{name: "tighten tuple selectors", fn: tightenTupleSelectors, required: true},
	{name: "lowered deadcode", fn: deadcode, required: true},
	{name: "checkLower", fn: checkLower, required: true},
	{name: "late phielim", fn: phielim},
	{name: "late copyelim", fn: copyelim},
	{name: "tighten", fn: tighten}, // move values closer to their uses
	{name: "late deadcode", fn: deadcode},
	{name: "critical", fn: critical, required: true}, // remove critical edges
	{name: "phi tighten", fn: phiTighten},            // place rematerializable phi args near uses to reduce value lifetimes
	{name: "likelyadjust", fn: likelyadjust},
	{name: "layout", fn: layout, required: true},     // schedule blocks
	{name: "schedule", fn: schedule, required: true}, // schedule values
	{name: "late nilcheck", fn: nilcheckelim2},
	{name: "flagalloc", fn: flagalloc, required: true}, // allocate flags register
	{name: "regalloc", fn: regalloc, required: true},   // allocate int & float registers + stack slots
	{name: "loop rotate", fn: loopRotate},
	{name: "stackframe", fn: stackframe, required: true},
	{name: "trim", fn: trim}, // remove empty blocks
}

// Double-check phase ordering constraints.
// This code is intended to document the ordering requirements
// between different phases. It does not override the passes
// list above.
type constraint struct {
	a, b string // a must come before b
}

var passOrder = [...]constraint{
	// "insert resched checks" uses mem, better to clean out stores first.
	{"dse", "insert resched checks"},
	// insert resched checks adds new blocks containing generic instructions
	{"insert resched checks", "lower"},
	{"insert resched checks", "tighten"},

	// prove relies on common-subexpression elimination for maximum benefits.
	{"generic cse", "prove"},
	// deadcode after prove to eliminate all new dead blocks.
	{"prove", "generic deadcode"},
	// common-subexpression before dead-store elim, so that we recognize
	// when two address expressions are the same.
	{"generic cse", "dse"},
	// cse substantially improves nilcheckelim efficacy
	{"generic cse", "nilcheckelim"},
	// allow deadcode to clean up after nilcheckelim
	{"nilcheckelim", "generic deadcode"},
	// nilcheckelim generates sequences of plain basic blocks
	{"nilcheckelim", "late fuse"},
	// nilcheckelim relies on opt to rewrite user nil checks
	{"opt", "nilcheckelim"},
	// tighten will be most effective when as many values have been removed as possible
	{"generic deadcode", "tighten"},
	{"generic cse", "tighten"},
	// checkbce needs the values removed
	{"generic deadcode", "check bce"},
	// don't run optimization pass until we've decomposed builtin objects
	{"decompose builtin", "late opt"},
	// decompose builtin is the last pass that may introduce new float ops, so run softfloat after it
	{"decompose builtin", "softfloat"},
	// tuple selectors must be tightened to generators and de-duplicated before scheduling
	{"tighten tuple selectors", "schedule"},
	// remove critical edges before phi tighten, so that phi args get better placement
	{"critical", "phi tighten"},
	// don't layout blocks until critical edges have been removed
	{"critical", "layout"},
	// regalloc requires the removal of all critical edges
	{"critical", "regalloc"},
	// regalloc requires all the values in a block to be scheduled
	{"schedule", "regalloc"},
	// checkLower must run after lowering & subsequent dead code elim
	{"lower", "checkLower"},
	{"lowered deadcode", "checkLower"},
	// late nilcheck needs instructions to be scheduled.
	{"schedule", "late nilcheck"},
	// flagalloc needs instructions to be scheduled.
	{"schedule", "flagalloc"},
	// regalloc needs flags to be allocated first.
	{"flagalloc", "regalloc"},
	// loopRotate will confuse regalloc.
	{"regalloc", "loop rotate"},
	// stackframe needs to know about spilled registers.
	{"regalloc", "stackframe"},
	// trim needs regalloc to be done first.
	{"regalloc", "trim"},
}

func init() {
	for _, c := range passOrder {
		a, b := c.a, c.b
		i := -1
		j := -1
		for k, p := range passes {
			if p.name == a {
				i = k
			}
			if p.name == b {
				j = k
			}
		}
		if i < 0 {
			log.Panicf("pass %s not found", a)
		}
		if j < 0 {
			log.Panicf("pass %s not found", b)
		}
		if i >= j {
			log.Panicf("passes %s and %s out of order", a, b)
		}
	}
}
