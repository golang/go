// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/objabi"
	"cmd/internal/src"
	"fmt"
	"log"
	"os"
	"regexp"
	"runtime"
	"strings"
	"time"
)

// Compile is the main entry point for this package.
// Compile modifies f so that on return:
//   · all Values in f map to 0 or 1 assembly instructions of the target architecture
//   · the order of f.Blocks is the order to emit the Blocks
//   · the order of b.Values is the order to emit the Values in each Block
//   · f has a non-nil regAlloc field
func Compile(f *Func) {
	// TODO: debugging - set flags to control verbosity of compiler,
	// which phases to dump IR before/after, etc.
	if f.Log() {
		f.Logf("compiling %s\n", f.Name)
	}

	// hook to print function & phase if panic happens
	phaseName := "init"
	defer func() {
		if phaseName != "" {
			err := recover()
			stack := make([]byte, 16384)
			n := runtime.Stack(stack, false)
			stack = stack[:n]
			f.Fatalf("panic during %s while compiling %s:\n\n%v\n\n%s\n", phaseName, f.Name, err, stack)
		}
	}()

	// Run all the passes
	printFunc(f)
	f.HTMLWriter.WriteFunc("start", "start", f)
	if BuildDump != "" && BuildDump == f.Name {
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

			f.Logf("  pass %s end %s\n", p.name, stats)
			printFunc(f)
			f.HTMLWriter.WriteFunc(phaseName, fmt.Sprintf("%s <span class=\"stats\">%s</span>", phaseName, stats), f)
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

	// Squash error printing defer
	phaseName = ""
}

// TODO: should be a config field
var dumpFileSeq int

// dumpFile creates a file from the phase name and function name
// Dumping is done to files to avoid buffering huge strings before
// output.
func (f *Func) dumpFile(phaseName string) {
	dumpFileSeq++
	fname := fmt.Sprintf("%s_%02d__%s.dump", f.Name, dumpFileSeq, phaseName)
	fname = strings.Replace(fname, " ", "_", -1)
	fname = strings.Replace(fname, "/", "_", -1)
	fname = strings.Replace(fname, ":", "_", -1)

	fi, err := os.Create(fname)
	if err != nil {
		f.Warnl(src.NoXPos, "Unable to create after-phase dump file %s", fname)
		return
	}

	p := stringFuncPrinter{w: fi}
	fprintFunc(p, f)
	fi.Close()
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

// Run consistency checker between each phase
var checkEnabled = false

// Debug output
var IntrinsicsDebug int
var IntrinsicsDisable bool

var BuildDebug int
var BuildTest int
var BuildStats int
var BuildDump string // name of function to dump after initial build of ssa

// PhaseOption sets the specified flag in the specified ssa phase,
// returning empty string if this was successful or a string explaining
// the error if it was not.
// A version of the phase name with "_" replaced by " " is also checked for a match.
// If the phase name begins a '~' then the rest of the underscores-replaced-with-blanks
// version is used as a regular expression to match the phase name(s).
//
// Special cases that have turned out to be useful:
//  ssa/check/on enables checking after each phase
//  ssa/all/time enables time reporting for all phases
//
// See gc/lex.go for dissection of the option string.
// Example uses:
//
// GO_GCFLAGS=-d=ssa/generic_cse/time,ssa/generic_cse/stats,ssa/generic_cse/debug=3 ./make.bash
//
// BOOT_GO_GCFLAGS=-d='ssa/~^.*scc$/off' GO_GCFLAGS='-d=ssa/~^.*scc$/off' ./make.bash
//
func PhaseOption(phase, flag string, val int, valString string) string {
	if phase == "help" {
		lastcr := 0
		phasenames := "    check, all, build, intrinsics"
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
    on, off, debug, mem, time, test, stats, dump

- <value> defaults to 1

- <function_name> is required for the "dump" flag, and specifies the
  name of function to dump after <phase>

Phase "all" supports flags "time", "mem", and "dump".
Phase "intrinsics" supports flags "on", "off", and "debug".

If the "dump" flag is specified, the output is written on a file named
<phase>__<function_name>_<seq>.dump; otherwise it is directed to stdout.

Examples:

    -d=ssa/check/on
enables checking after each phase

    -d=ssa/all/time
enables time reporting for all phases

    -d=ssa/prove/debug=2
sets debugging level to 2 in the prove pass

Multiple flags can be passed at once, by separating them with
commas. For example:

    -d=ssa/check/on,ssa/all/time
`
	}

	if phase == "check" && flag == "on" {
		checkEnabled = val != 0
		return ""
	}
	if phase == "check" && flag == "off" {
		checkEnabled = val == 0
		return ""
	}

	alltime := false
	allmem := false
	alldump := false
	if phase == "all" {
		if flag == "time" {
			alltime = val != 0
		} else if flag == "mem" {
			allmem = val != 0
		} else if flag == "dump" {
			alldump = val != 0
			if alldump {
				BuildDump = valString
			}
		} else {
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option", flag, phase)
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
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option", flag, phase)
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
			BuildDump = valString
		default:
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option", flag, phase)
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
	{name: "opt", fn: opt, required: true},               // TODO: split required rules and optimizing rules
	{name: "zero arg cse", fn: zcse, required: true},     // required to merge OpSB values
	{name: "opt deadcode", fn: deadcode, required: true}, // remove any blocks orphaned during opt
	{name: "generic cse", fn: cse},
	{name: "phiopt", fn: phiopt},
	{name: "nilcheckelim", fn: nilcheckelim},
	{name: "prove", fn: prove},
	{name: "decompose builtin", fn: decomposeBuiltIn, required: true},
	{name: "softfloat", fn: softfloat, required: true},
	{name: "late opt", fn: opt, required: true}, // TODO: split required rules and optimizing rules
	{name: "dead auto elim", fn: elimDeadAutosGeneric},
	{name: "generic deadcode", fn: deadcode},
	{name: "check bce", fn: checkbce},
	{name: "branchelim", fn: branchelim},
	{name: "fuse", fn: fuse},
	{name: "dse", fn: dse},
	{name: "writebarrier", fn: writebarrier, required: true}, // expand write barrier ops
	{name: "insert resched checks", fn: insertLoopReschedChecks,
		disabled: objabi.Preemptibleloops_enabled == 0}, // insert resched checks in loops.
	{name: "lower", fn: lower, required: true},
	{name: "lowered cse", fn: cse},
	{name: "elim unread autos", fn: elimUnreadAutos},
	{name: "lowered deadcode", fn: deadcode, required: true},
	{name: "checkLower", fn: checkLower, required: true},
	{name: "late phielim", fn: phielim},
	{name: "late copyelim", fn: copyelim},
	{name: "tighten", fn: tighten}, // move values closer to their uses
	{name: "phi tighten", fn: phiTighten},
	{name: "late deadcode", fn: deadcode},
	{name: "critical", fn: critical, required: true}, // remove critical edges
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
	{"nilcheckelim", "fuse"},
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
