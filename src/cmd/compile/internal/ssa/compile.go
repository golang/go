// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ssa/ssaconfig"
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
	"strconv"
	"strings"
	"sync"
	"time"
)

// Compile is the main entry point for this package.
// Compile modifies f so that on return:
//   - all Values in f map to 0 or 1 assembly instructions of the target architecture
//   - the order of f.Blocks is the order to emit the Blocks
//   - the order of b.Values is the order to emit the Values in each Block
//   - f has a non-nil regAlloc field
func Compile(f *Func, htmlWriter *HTMLWriter) {
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
			if htmlWriter != nil {
				htmlWriter.flushPhases()
			}
			f.Fatalf("panic during %s while compiling %s:\n\n%v\n\n%s\n", phaseName, f.Name, err, stack)
		}
	}()

	// Run all the passes
	if f.Log() {
		PrintFunc(f)
	}
	htmlWriter.WritePhase("start", "start")
	if ssaconfig.BuildDump[f.Name] {
		f.DumpFile("build")
	}
	if checkEnabled {
		checkFunc(f)
	}
	const logMemStats = false
	for _, p := range passes {
		if !f.Config.Optimize && !p.Required || p.Disabled {
			continue
		}
		f.Pass = &p
		phaseName = p.Name
		if f.Log() {
			f.Logf("  pass %s begin\n", p.Name)
		}
		// TODO: capture logging during this pass, add it to the HTML
		var mStart runtime.MemStats
		if logMemStats || p.Mem {
			runtime.ReadMemStats(&mStart)
		}

		if checkEnabled && !f.Scheduled {
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
		p.Fn(f)
		tEnd := time.Now()

		// Need something less crude than "Log the whole intermediate result".
		if f.Log() || htmlWriter != nil {
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
				f.Logf("  pass %s end %s\n", p.Name, stats)
				PrintFunc(f)
			}
			htmlWriter.WritePhase(phaseName, fmt.Sprintf("%s <span class=\"stats\">%s</span>", phaseName, stats))
		}
		if p.Time || p.Mem {
			// Surround timing information w/ enough context to allow comparisons.
			time := tEnd.Sub(tStart).Nanoseconds()
			if p.Time {
				f.LogStat("TIME(ns)", time)
			}
			if p.Mem {
				var mEnd runtime.MemStats
				runtime.ReadMemStats(&mEnd)
				nBytes := mEnd.TotalAlloc - mStart.TotalAlloc
				nAllocs := mEnd.Mallocs - mStart.Mallocs
				f.LogStat("TIME(ns):BYTES:ALLOCS", time, nBytes, nAllocs)
			}
		}
		if p.Dump != nil && p.Dump[f.Name] {
			// Dump function to appropriately named file
			f.DumpFile(phaseName)
		}
		if checkEnabled {
			checkFunc(f)
		}
	}

	if htmlWriter != nil {
		// Ensure we write any pending phases to the html
		htmlWriter.flushPhases()
	}

	if f.RuleMatches != nil {
		var keys []string
		for key := range f.RuleMatches {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		buf := new(strings.Builder)
		fmt.Fprintf(buf, "%s: ", f.Name)
		for _, key := range keys {
			fmt.Fprintf(buf, "%s=%d ", key, f.RuleMatches[key])
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
	fname = strings.ReplaceAll(fname, " ", "_")
	fname = strings.ReplaceAll(fname, "/", "_")
	fname = strings.ReplaceAll(fname, ":", "_")

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

// DumpFile creates a file from the phase name and function name
// Dumping is done to files to avoid buffering huge strings before
// output.
func (f *Func) DumpFile(phaseName string) {
	fi := f.DumpFileForPhase(phaseName)
	if fi != nil {
		p := StringFuncPrinter{w: fi}
		FprintFunc(p, f)
		fi.Close()
	}
}

type Pass struct {
	Name     string
	Fn       func(*Func)
	Required bool
	Disabled bool
	Time     bool             // report time to run pass
	Mem      bool             // report mem stats to run pass
	Stats    int              // pass reports own "stats" (e.g., branches removed)
	Debug    int              // pass performs some debugging. =1 should be in error-testing-friendly Warnl format.
	Test     int              // pass-specific ad-hoc option, perhaps useful in development
	Dump     map[string]bool  // dump if function name matches
	Keywords map[string]int64 // ad hoc parameters, typically for experiments/tuning
	UsedKW   map[string]bool  // if a keyword is supplied to a phase, note that it was used.
}

func (p *Pass) AddDump(s string) {
	if p.Dump == nil {
		p.Dump = make(map[string]bool)
	}
	p.Dump[s] = true
}

func (p *Pass) String() string {
	if p == nil {
		return "nil pass"
	}
	return p.Name
}

var kwMu sync.Mutex

func (p *Pass) Val(kw string, ifUnset int64) int64 {
	if p == nil || p.Keywords == nil {
		return ifUnset
	}
	if v, ok := p.Keywords[kw]; ok {
		kwMu.Lock()
		p.UsedKW[kw] = true
		kwMu.Unlock()
		return v
	}
	return ifUnset
}

// Run consistency checker between each phase
var (
	checkEnabled  = false
	checkRandSeed = 0
)

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
			pn := strings.ReplaceAll(p.Name, " ", "_")
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
    on, off, debug, mem, time, test, stats, dump, seed, @<keyword>

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
			DebugPoset = checkEnabled // also turn on advanced self-checking in prove's data structure
			return ""
		case "off":
			checkEnabled = val == 0
			DebugPoset = checkEnabled
			return ""
		case "seed":
			checkEnabled = true
			checkRandSeed = val
			DebugPoset = checkEnabled
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
				ssaconfig.BuildDump[valString] = true
				ssaconfig.GenssaDump[valString] = true
			}
		default:
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option (expected ssa/all/{time,mem,dump=function_name})", flag, phase)
		}
	}

	if phase == "intrinsics" {
		switch flag {
		case "on":
			ssaconfig.IntrinsicsDisable = val == 0
		case "off":
			ssaconfig.IntrinsicsDisable = val != 0
		case "debug":
			ssaconfig.IntrinsicsDebug = val
		default:
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option (expected ssa/intrinsics/{on,off,debug})", flag, phase)
		}
		return ""
	}
	if phase == "build" {
		switch flag {
		case "debug":
			ssaconfig.BuildDebug = val
		case "test":
			ssaconfig.BuildTest = val
		case "stats":
			ssaconfig.BuildStats = val
		case "dump":
			ssaconfig.BuildDump[valString] = true
		default:
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option (expected ssa/build/{debug,test,stats,dump=function_name})", flag, phase)
		}
		return ""
	}
	if phase == "genssa" {
		switch flag {
		case "dump":
			ssaconfig.GenssaDump[valString] = true
		default:
			return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option (expected ssa/genssa/dump=function_name)", flag, phase)
		}
		return ""
	}

	underphase := strings.ReplaceAll(phase, "_", " ")
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
			p.Time = alltime
			p.Mem = allmem
			if alldump {
				p.AddDump(valString)
			}
			passes[i] = p
			matchedOne = true
		} else if p.Name == phase || p.Name == underphase || re != nil && re.MatchString(p.Name) {
			switch flag {
			case "on":
				p.Disabled = val == 0
			case "off":
				p.Disabled = val != 0
			case "time":
				p.Time = val != 0
			case "mem":
				p.Mem = val != 0
			case "debug":
				p.Debug = val
			case "stats":
				p.Stats = val
			case "test":
				p.Test = val
			case "dump":
				p.AddDump(valString)
			default:
				if flag != "" && flag[0] == '@' {
					if p.Keywords == nil {
						p.Keywords = make(map[string]int64)
						p.UsedKW = make(map[string]bool)
					}
					val64, err := strconv.ParseInt(valString, 10, 64)
					if err != nil {
						return fmt.Sprintf("Failed to parse %s as integer value in -d=ssa/%s/%s=%s option", valString, phase, flag, valString)
					}
					p.Keywords[flag[1:]] = int64(val64)
				} else {
					return fmt.Sprintf("Did not find a flag matching %s in -d=ssa/%s debug option", flag, phase)
				}
			}
			if p.Disabled && p.Required {
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
var passes = [...]Pass{
	{Name: "number lines", Fn: numberLines, Required: true},
	{Name: "early phielim and copyelim", Fn: copyelim},
	{Name: "early deadcode", Fn: deadcode}, // remove generated dead code to avoid doing pointless work during opt
	{Name: "short circuit", Fn: shortcircuit},
	{Name: "decompose user", Fn: decomposeUser, Required: true},
	{Name: "pre-opt deadcode", Fn: deadcode},
	{Name: "opt", Fn: opt, Required: true},
	{Name: "zero arg cse", Fn: zcse, Required: true},     // required to merge OpSB values
	{Name: "opt deadcode", Fn: deadcode, Required: true}, // remove any blocks orphaned during opt
	{Name: "generic cse", Fn: cse},
	{Name: "phiopt", Fn: phiopt},
	{Name: "gcse deadcode", Fn: deadcode, Required: true}, // clean out after cse and phiopt
	{Name: "nilcheckelim", Fn: nilcheckelim},
	{Name: "prove", Fn: prove},
	{Name: "divisible", Fn: divisible, Required: true},
	{Name: "divmod", Fn: divmod, Required: true},
	{Name: "middle opt", Fn: opt, Required: true},
	{Name: "known bits", Fn: KnownBits},
	{Name: "early fuse", Fn: fuseEarly},
	{Name: "expand calls", Fn: expandCalls, Required: true},
	{Name: "decompose builtin", Fn: postExpandCallsDecompose, Required: true},
	{Name: "softfloat", Fn: softfloat, Required: true},
	{Name: "branchelim", Fn: branchelim},
	{Name: "late opt", Fn: opt, Required: true},
	{Name: "dead auto elim", Fn: elimDeadAutosGeneric},
	{Name: "sccp", Fn: sccp},
	{Name: "generic deadcode", Fn: deadcode, Required: true}, // remove dead stores, which otherwise mess up store chain
	{Name: "late fuse", Fn: fuseLate},
	{Name: "check bce", Fn: checkbce},
	{Name: "dse", Fn: dse},
	{Name: "memcombine", Fn: memcombine},
	{Name: "writebarrier", Fn: writebarrier, Required: true}, // expand write barrier ops
	{Name: "insert resched checks", Fn: insertLoopReschedChecks,
		Disabled: !buildcfg.Experiment.PreemptibleLoops}, // insert resched checks in loops.
	{Name: "cpufeatures", Fn: cpufeatures, Required: buildcfg.Experiment.SIMD, Disabled: !buildcfg.Experiment.SIMD},
	{Name: "rewrite tern", Fn: rewriteTern, Required: false, Disabled: !buildcfg.Experiment.SIMD},
	{Name: "lower", Fn: lower, Required: true},
	{Name: "addressing modes", Fn: addressingModes, Required: false},
	{Name: "late lower", Fn: lateLower, Required: true},
	{Name: "pair", Fn: pair},
	{Name: "lowered deadcode for cse", Fn: deadcode}, // deadcode immediately before CSE avoids CSE making dead values live again
	{Name: "lowered cse", Fn: cse},
	{Name: "elim unread autos", Fn: elimUnreadAutos},
	{Name: "tighten tuple selectors", Fn: tightenTupleSelectors, Required: true},
	{Name: "lowered deadcode", Fn: deadcode, Required: true},
	{Name: "checkLower", Fn: checkLower, Required: true},
	{Name: "loop invariant", Fn: licm},
	{Name: "late phielim and copyelim", Fn: copyelim},
	{Name: "tighten", Fn: tighten, Required: true}, // move values closer to their uses
	// TODO: fix 80102 and re-enable.
	//{name: "merge conditional branches", fn: mergeConditionalBranches}, // generate conditional comparison instructions on ARM64 architecture
	{Name: "late deadcode", Fn: deadcode},
	{Name: "critical", Fn: critical, Required: true}, // remove critical edges
	{Name: "phi tighten", Fn: phiTighten},            // place rematerializable phi args near uses to reduce value lifetimes
	{Name: "likelyadjust", Fn: likelyadjust},
	{Name: "layout", Fn: layout, Required: true},     // schedule blocks
	{Name: "schedule", Fn: schedule, Required: true}, // schedule values
	{Name: "late nilcheck", Fn: nilcheckelim2},
	{Name: "flagalloc", Fn: flagalloc, Required: true}, // allocate flags register
	{Name: "regalloc", Fn: regalloc, Required: true},   // allocate int & float registers + stack slots
	{Name: "loop rotate", Fn: loopRotate},
	{Name: "trim", Fn: trim}, // remove empty blocks
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
	// divisible after prove to let prove analyze div and mod
	{"prove", "divisible"},
	// divmod after divisible to avoid rewriting subexpressions of ones divisible will handle
	{"divisible", "divmod"},
	// divmod before decompose builtin to handle 64-bit on 32-bit systems
	{"divmod", "decompose builtin"},
	// common-subexpression before dead-store elim, so that we recognize
	// when two address expressions are the same.
	{"generic cse", "dse"},
	// cse substantially improves nilcheckelim efficacy
	{"generic cse", "nilcheckelim"},
	// allow deadcode to clean up after nilcheckelim
	{"nilcheckelim", "generic deadcode"},
	// nilcheckelim generates sequences of plain basic blocks
	{"nilcheckelim", "late fuse"},
	// nilcheckelim relies on the first opt to rewrite user nil checks
	{"opt", "nilcheckelim"},
	// tighten will be most effective when as many values have been removed as possible
	{"generic deadcode", "tighten"},
	{"generic cse", "tighten"},
	// checkbce needs the values removed
	{"generic deadcode", "check bce"},
	// decompose builtin now also cleans up after expand calls
	{"expand calls", "decompose builtin"},
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
	// the rules in late lower run after the general rules.
	{"lower", "late lower"},
	// late lower may generate some values that need to be CSEed.
	{"late lower", "lowered cse"},
	// checkLower must run after lowering & subsequent dead code elim
	{"lower", "checkLower"},
	{"lowered deadcode", "checkLower"},
	{"late lower", "checkLower"},
	// late nilcheck needs instructions to be scheduled.
	{"schedule", "late nilcheck"},
	// flagalloc needs instructions to be scheduled.
	{"schedule", "flagalloc"},
	// regalloc needs flags to be allocated first.
	{"flagalloc", "regalloc"},
	// loopRotate will confuse regalloc.
	{"regalloc", "loop rotate"},
	// trim needs regalloc to be done first.
	{"regalloc", "trim"},
	// memcombine works better if fuse happens first, to help merge stores.
	{"late fuse", "memcombine"},
	// memcombine is a arch-independent pass.
	{"memcombine", "lower"},
	// late opt transform some CondSelects into math.
	{"branchelim", "late opt"},
	// branchelim is an arch-independent pass.
	{"branchelim", "lower"},
	// lower needs cpu feature information (for SIMD)
	{"cpufeatures", "lower"},
	// known bits is an arch-independent pass.
	{"known bits", "lower"},
	// known bits does very little except some fancy constant folding and we need opt to clean it up.
	{"known bits", "late opt"},
	// known bits does a better job once prove cleaned up some always taken and never taken branches.
	// known bits also relies on the output to be mostly topo-sorted (for recursion limit purposes) which prove does.
	{"prove", "known bits"},
}

func PostCompile() {
	for _, c := range passes {
		if c.Keywords != nil {
			for k := range c.Keywords {
				if !c.UsedKW[k] {
					// If someone specified a debugging keyword that was not
					// consumed, they might want to know about this.
					base.Warn("Keyword %s for pass %s was not used", k, c.Name)
				}
			}
		}
	}
}

func init() {
	for _, c := range passOrder {
		a, b := c.a, c.b
		i := -1
		j := -1
		for k, p := range passes {
			if p.Name == a {
				i = k
			}
			if p.Name == b {
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
