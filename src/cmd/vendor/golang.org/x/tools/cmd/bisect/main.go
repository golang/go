// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bisect finds changes responsible for causing a failure.
// A typical use is to identify the source locations in a program
// that are miscompiled by a given compiler optimization.
//
// Usage:
//
//	bisect [flags] [var=value...] command [arguments...]
//
// Bisect operates on a target command line – the target – that can be
// run with various changes individually enabled or disabled. With none
// of the changes enabled, the target is known to succeed (exit with exit
// code zero). With all the changes enabled, the target is known to fail
// (exit any other way). Bisect repeats the target with different sets of
// changes enabled, using binary search to find (non-overlapping) minimal
// change sets that provoke the failure.
//
// The target must cooperate with bisect by accepting a change pattern
// and then enabling and reporting the changes that match that pattern.
// The change pattern is passed to the target by substituting it anywhere
// the string PATTERN appears in the environment values or the command
// arguments. For each change that matches the pattern, the target must
// enable that change and also print one or more “match lines”
// (to standard output or standard error) describing the change.
// The [golang.org/x/tools/internal/bisect] package provides functions to help
// targets implement this protocol. We plan to publish that package
// in a non-internal location after finalizing its API.
//
// Bisect starts by running the target with no changes enabled and then
// with all changes enabled. It expects the former to succeed and the latter to fail,
// and then it will search for the minimal set of changes that must be enabled
// to provoke the failure. If the situation is reversed – the target fails with no
// changes enabled and succeeds with all changes enabled – then bisect
// automatically runs in reverse as well, searching for the minimal set of changes
// that must be disabled to provoke the failure.
//
// Bisect prints tracing logs to standard error and the minimal change sets
// to standard output.
//
// # Command Line Flags
//
// Bisect supports the following command-line flags:
//
//	-max=M
//
// Stop after finding M minimal change sets. The default is no maximum, meaning to run until
// all changes that provoke a failure have been identified.
//
//	-maxset=S
//
// Disallow change sets larger than S elements. The default is no maximum.
//
//	-timeout=D
//
// If the target runs for longer than duration D, stop the target and interpret that as a failure.
// The default is no timeout.
//
//	-count=N
//
// Run each trial N times (default 2), checking for consistency.
//
//	-v
//
// Print verbose output, showing each run and its match lines.
//
// In addition to these general flags,
// bisect supports a few “shortcut” flags that make it more convenient
// to use with specific targets.
//
//	-compile=<rewrite>
//
// This flag is equivalent to adding an environment variable
// “GOCOMPILEDEBUG=<rewrite>hash=PATTERN”,
// which, as discussed in more detail in the example below,
// allows bisect to identify the specific source locations where the
// compiler rewrite causes the target to fail.
//
//	-godebug=<name>=<value>
//
// This flag is equivalent to adding an environment variable
// “GODEBUG=<name>=<value>#PATTERN”,
// which allows bisect to identify the specific call stacks where
// the changed [GODEBUG setting] value causes the target to fail.
//
// # Example
//
// The Go compiler provides support for enabling or disabling certain rewrites
// and optimizations to allow bisect to identify specific source locations where
// the rewrite causes the program to fail. For example, to bisect a failure caused
// by the new loop variable semantics:
//
//	bisect go test -gcflags=all=-d=loopvarhash=PATTERN
//
// The -gcflags=all= instructs the go command to pass the -d=... to the Go compiler
// when compiling all packages. Bisect varies PATTERN to determine the minimal set of changes
// needed to reproduce the failure.
//
// The go command also checks the GOCOMPILEDEBUG environment variable for flags
// to pass to the compiler, so the above command is equivalent to:
//
//	bisect GOCOMPILEDEBUG=loopvarhash=PATTERN go test
//
// Finally, as mentioned earlier, the -compile flag allows shortening this command further:
//
//	bisect -compile=loopvar go test
//
// # Defeating Build Caches
//
// Build systems cache build results, to avoid repeating the same compilations
// over and over. When using a cached build result, the go command (correctly)
// reprints the cached standard output and standard error associated with that
// command invocation. (This makes commands like 'go build -gcflags=-S' for
// printing an assembly listing work reliably.)
//
// Unfortunately, most build systems, including Bazel, are not as careful
// as the go command about reprinting compiler output. If the compiler is
// what prints match lines, a build system that suppresses compiler
// output when using cached compiler results will confuse bisect.
// To defeat such build caches, bisect replaces the literal text “RANDOM”
// in environment values and command arguments with a random 64-bit value
// during each invocation. The Go compiler conveniently accepts a
// -d=ignore=... debug flag that ignores its argument, so to run the
// previous example using Bazel, the invocation is:
//
//	bazel test --define=gc_goopts=-d=loopvarhash=PATTERN,unused=RANDOM //path/to:test
//
// [GODEBUG setting]: https://tip.golang.org/doc/godebug
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math/bits"
	"math/rand"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"golang.org/x/tools/internal/bisect"
)

// Preserve import of bisect, to allow [bisect.Match] in the doc comment.
var _ bisect.Matcher

func usage() {
	fmt.Fprintf(os.Stderr, "usage: bisect [flags] [var=value...] command [arguments...]\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("bisect: ")

	var b Bisect
	b.Stdout = os.Stdout
	b.Stderr = os.Stderr
	flag.IntVar(&b.Max, "max", 0, "stop after finding `m` failing change sets")
	flag.IntVar(&b.MaxSet, "maxset", 0, "do not search for change sets larger than `s` elements")
	flag.DurationVar(&b.Timeout, "timeout", 0, "stop target and consider failed after duration `d`")
	flag.IntVar(&b.Count, "count", 2, "run target `n` times for each trial")
	flag.BoolVar(&b.Verbose, "v", false, "enable verbose output")

	env := ""
	envFlag := ""
	flag.Func("compile", "bisect source locations affected by Go compiler `rewrite` (fma, loopvar, ...)", func(value string) error {
		if envFlag != "" {
			return fmt.Errorf("cannot use -%s and -compile", envFlag)
		}
		envFlag = "compile"
		env = "GOCOMPILEDEBUG=" + value + "hash=PATTERN"
		return nil
	})
	flag.Func("godebug", "bisect call stacks affected by GODEBUG setting `name=value`", func(value string) error {
		if envFlag != "" {
			return fmt.Errorf("cannot use -%s and -godebug", envFlag)
		}
		envFlag = "godebug"
		env = "GODEBUG=" + value + "#PATTERN"
		return nil
	})

	flag.Usage = usage
	flag.Parse()
	args := flag.Args()

	// Split command line into env settings, command name, args.
	i := 0
	for i < len(args) && strings.Contains(args[i], "=") {
		i++
	}
	if i == len(args) {
		usage()
	}
	b.Env, b.Cmd, b.Args = args[:i], args[i], args[i+1:]
	if env != "" {
		b.Env = append([]string{env}, b.Env...)
	}

	// Check that PATTERN is available for us to vary.
	found := false
	for _, e := range b.Env {
		if _, v, _ := strings.Cut(e, "="); strings.Contains(v, "PATTERN") {
			found = true
		}
	}
	for _, a := range b.Args {
		if strings.Contains(a, "PATTERN") {
			found = true
		}
	}
	if !found {
		log.Fatalf("no PATTERN in target environment or args")
	}

	if !b.Search() {
		os.Exit(1)
	}
}

// A Bisect holds the state for a bisect invocation.
type Bisect struct {
	// Env is the additional environment variables for the command.
	// PATTERN and RANDOM are substituted in the values, but not the names.
	Env []string

	// Cmd is the command (program name) to run.
	// PATTERN and RANDOM are not substituted.
	Cmd string

	// Args is the command arguments.
	// PATTERN and RANDOM are substituted anywhere they appear.
	Args []string

	// Command-line flags controlling bisect behavior.
	Max     int           // maximum number of sets to report (0 = unlimited)
	MaxSet  int           // maximum number of elements in a set (0 = unlimited)
	Timeout time.Duration // kill target and assume failed after this duration (0 = unlimited)
	Count   int           // run target this many times for each trial and give up if flaky (min 1 assumed; default 2 on command line set in main)
	Verbose bool          // print long output about each trial (only useful for debugging bisect itself)

	// State for running bisect, replaced during testing.
	// Failing change sets are printed to Stdout; all other output goes to Stderr.
	Stdout  io.Writer                                                             // where to write standard output (usually os.Stdout)
	Stderr  io.Writer                                                             // where to write standard error (usually os.Stderr)
	TestRun func(env []string, cmd string, args []string) (out []byte, err error) // if non-nil, used instead of exec.Command

	// State maintained by Search.

	// By default, Search looks for a minimal set of changes that cause a failure when enabled.
	// If Disable is true, the search is inverted and seeks a minimal set of changes that
	// cause a failure when disabled. In this case, the search proceeds as normal except that
	// each pattern starts with a !.
	Disable bool

	// SkipHexDigits is the number of hex digits to use in skip messages.
	// If the set of available changes is the same in each run, as it should be,
	// then this doesn't matter: we'll only exclude suffixes that uniquely identify
	// a given change. But for some programs, especially bisecting runtime
	// behaviors, sometimes enabling one change unlocks questions about other
	// changes. Strictly speaking this is a misuse of bisect, but just to make
	// bisect more robust, we use the y and n runs to create an estimate of the
	// number of bits needed for a unique suffix, and then we round it up to
	// a number of hex digits, with one extra digit for good measure, and then
	// we always use that many hex digits for skips.
	SkipHexDigits int

	// Add is a list of suffixes to add to every trial, because they
	// contain changes that are necessary for a group we are assembling.
	Add []string

	// Skip is a list of suffixes that uniquely identify changes to exclude from every trial,
	// because they have already been used in failing change sets.
	// Suffixes later in the list may only be unique after removing
	// the ones earlier in the list.
	// Skip applies after Add.
	Skip []string
}

// A Result holds the result of a single target trial.
type Result struct {
	Success bool   // whether the target succeeded (exited with zero status)
	Cmd     string // full target command line
	Out     string // full target output (stdout and stderr combined)

	Suffix    string   // the suffix used for collecting MatchIDs, MatchText, and MatchFull
	MatchIDs  []uint64 // match IDs enabled during this trial
	MatchText []string // match reports for the IDs, with match markers removed
	MatchFull []string // full match lines for the IDs, with match markers kept
}

// &searchFatal is a special panic value to signal that Search failed.
// This lets us unwind the search recursion on a fatal error
// but have Search return normally.
var searchFatal int

// Search runs a bisect search according to the configuration in b.
// It reports whether any failing change sets were found.
func (b *Bisect) Search() bool {
	defer func() {
		// Recover from panic(&searchFatal), implicitly returning false from Search.
		// Re-panic on any other panic.
		if e := recover(); e != nil && e != &searchFatal {
			panic(e)
		}
	}()

	// Run with no changes and all changes, to figure out which direction we're searching.
	// The goal is to find the minimal set of changes to toggle
	// starting with the state where everything works.
	// If "no changes" succeeds and "all changes" fails,
	// we're looking for a minimal set of changes to enable to provoke the failure
	// (broken = runY, b.Negate = false)
	// If "no changes" fails and "all changes" succeeds,
	// we're looking for a minimal set of changes to disable to provoke the failure
	// (broken = runN, b.Negate = true).

	b.Logf("checking target with all changes disabled")
	runN := b.Run("n")

	b.Logf("checking target with all changes enabled")
	runY := b.Run("y")

	var broken *Result
	switch {
	case runN.Success && !runY.Success:
		b.Logf("target succeeds with no changes, fails with all changes")
		b.Logf("searching for minimal set of enabled changes causing failure")
		broken = runY
		b.Disable = false

	case !runN.Success && runY.Success:
		b.Logf("target fails with no changes, succeeds with all changes")
		b.Logf("searching for minimal set of disabled changes causing failure")
		broken = runN
		b.Disable = true

	case runN.Success && runY.Success:
		b.Fatalf("target succeeds with no changes and all changes")

	case !runN.Success && !runY.Success:
		b.Fatalf("target fails with no changes and all changes")
	}

	// Compute minimum number of bits needed to distinguish
	// all the changes we saw during N and all the changes we saw during Y.
	b.SkipHexDigits = skipHexDigits(runN.MatchIDs, runY.MatchIDs)

	// Loop finding and printing change sets, until none remain.
	found := 0
	for {
		// Find set.
		bad := b.search(broken)
		if bad == nil {
			if found == 0 {
				b.Fatalf("cannot find any failing change sets of size ≤ %d", b.MaxSet)
			}
			break
		}

		// Confirm that set really does fail, to avoid false accusations.
		// Also asking for user-visible output; earlier runs did not.
		b.Logf("confirming failing change set")
		b.Add = append(b.Add[:0], bad...)
		broken = b.Run("v")
		if broken.Success {
			b.Logf("confirmation run succeeded unexpectedly")
		}
		b.Add = b.Add[:0]

		// Print confirmed change set.
		found++
		b.Logf("FOUND failing change set")
		desc := "(enabling changes causes failure)"
		if b.Disable {
			desc = "(disabling changes causes failure)"
		}
		fmt.Fprintf(b.Stdout, "--- change set #%d %s\n%s\n---\n", found, desc, strings.Join(broken.MatchText, "\n"))

		// Stop if we've found enough change sets.
		if b.Max > 0 && found >= b.Max {
			break
		}

		// If running bisect target | tee bad.txt, prints to stdout and stderr
		// both appear on the terminal, but the ones to stdout go through tee
		// and can take a little bit of extra time. Sleep 1 millisecond to give
		// tee time to catch up, so that its stdout print does not get interlaced
		// with the stderr print from the next b.Log message.
		time.Sleep(1 * time.Millisecond)

		// Disable the now-known-bad changes and see if any failures remain.
		b.Logf("checking for more failures")
		b.Skip = append(bad, b.Skip...)
		broken = b.Run("")
		if broken.Success {
			what := "enabled"
			if b.Disable {
				what = "disabled"
			}
			b.Logf("target succeeds with all remaining changes %s", what)
			break
		}
		b.Logf("target still fails; searching for more bad changes")
	}
	return true
}

// Fatalf prints a message to standard error and then panics,
// causing Search to return false.
func (b *Bisect) Fatalf(format string, args ...any) {
	s := fmt.Sprintf("bisect: fatal error: "+format, args...)
	if !strings.HasSuffix(s, "\n") {
		s += "\n"
	}
	b.Stderr.Write([]byte(s))
	panic(&searchFatal)
}

// Logf prints a message to standard error.
func (b *Bisect) Logf(format string, args ...any) {
	s := fmt.Sprintf("bisect: "+format, args...)
	if !strings.HasSuffix(s, "\n") {
		s += "\n"
	}
	b.Stderr.Write([]byte(s))
}

func skipHexDigits(idY, idN []uint64) int {
	var all []uint64
	seen := make(map[uint64]bool)
	for _, x := range idY {
		seen[x] = true
		all = append(all, x)
	}
	for _, x := range idN {
		if !seen[x] {
			seen[x] = true
			all = append(all, x)
		}
	}
	sort.Slice(all, func(i, j int) bool { return bits.Reverse64(all[i]) < bits.Reverse64(all[j]) })
	digits := sort.Search(64/4, func(digits int) bool {
		mask := uint64(1)<<(4*digits) - 1
		for i := 0; i+1 < len(all); i++ {
			if all[i]&mask == all[i+1]&mask {
				return false
			}
		}
		return true
	})
	if digits < 64/4 {
		digits++
	}
	return digits
}

// search searches for a single locally minimal change set.
//
// Invariant: r describes the result of r.Suffix + b.Add, which failed.
// (There's an implicit -b.Skip everywhere here. b.Skip does not change.)
// We want to extend r.Suffix to preserve the failure, working toward
// a suffix that identifies a single change.
func (b *Bisect) search(r *Result) []string {
	// The caller should be passing in a failure result that we diagnose.
	if r.Success {
		b.Fatalf("internal error: unexpected success") // mistake by caller
	}

	// If the failure reported no changes, the target is misbehaving.
	if len(r.MatchIDs) == 0 {
		b.Fatalf("failure with no reported changes:\n\n$ %s\n%s\n", r.Cmd, r.Out)
	}

	// If there's one matching change, that's the one we're looking for.
	if len(r.MatchIDs) == 1 {
		return []string{fmt.Sprintf("x%0*x", b.SkipHexDigits, r.MatchIDs[0]&(1<<(4*b.SkipHexDigits)-1))}
	}

	// If the suffix we were tracking in the trial is already 64 bits,
	// either the target is bad or bisect itself is buggy.
	if len(r.Suffix) >= 64 {
		b.Fatalf("failed to isolate a single change with very long suffix")
	}

	// We want to split the current matchIDs by left-extending the suffix with 0 and 1.
	// If all the matches have the same next bit, that won't cause a split, which doesn't
	// break the algorithm but does waste time. Avoid wasting time by left-extending
	// the suffix to the longest suffix shared by all the current match IDs
	// before adding 0 or 1.
	suffix := commonSuffix(r.MatchIDs)
	if !strings.HasSuffix(suffix, r.Suffix) {
		b.Fatalf("internal error: invalid common suffix") // bug in commonSuffix
	}

	// Run 0suffix and 1suffix. If one fails, chase down the failure in that half.
	r0 := b.Run("0" + suffix)
	if !r0.Success {
		return b.search(r0)
	}
	r1 := b.Run("1" + suffix)
	if !r1.Success {
		return b.search(r1)
	}

	// suffix failed, but 0suffix and 1suffix succeeded.
	// Assuming the target isn't flaky, this means we need
	// at least one change from 0suffix AND at least one from 1suffix.
	// We are already tracking N = len(b.Add) other changes and are
	// allowed to build sets of size at least 1+N (or we shouldn't be here at all).
	// If we aren't allowed to build sets of size 2+N, give up this branch.
	if b.MaxSet > 0 && 2+len(b.Add) > b.MaxSet {
		return nil
	}

	// Adding all matches for 1suffix, recurse to narrow down 0suffix.
	old := len(b.Add)
	b.Add = append(b.Add, "1"+suffix)
	r0 = b.Run("0" + suffix)
	if r0.Success {
		// 0suffix + b.Add + 1suffix = suffix + b.Add is what r describes, and it failed.
		b.Fatalf("target fails inconsistently")
	}
	bad0 := b.search(r0)
	if bad0 == nil {
		// Search failed due to MaxSet limit.
		return nil
	}
	b.Add = b.Add[:old]

	// Adding the specific match we found in 0suffix, recurse to narrow down 1suffix.
	b.Add = append(b.Add[:old], bad0...)
	r1 = b.Run("1" + suffix)
	if r1.Success {
		// 1suffix + b.Add + bad0 = bad0 + b.Add + 1suffix is what b.search(r0) reported as a failure.
		b.Fatalf("target fails inconsistently")
	}
	bad1 := b.search(r1)
	if bad1 == nil {
		// Search failed due to MaxSet limit.
		return nil
	}
	b.Add = b.Add[:old]

	// bad0 and bad1 together provoke the failure.
	return append(bad0, bad1...)
}

// Run runs a set of trials selecting changes with the given suffix,
// plus the ones in b.Add and not the ones in b.Skip.
// The returned result's MatchIDs, MatchText, and MatchFull
// only list the changes that match suffix.
// When b.Count > 1, Run runs b.Count trials and requires
// that they all succeed or they all fail. If not, it calls b.Fatalf.
func (b *Bisect) Run(suffix string) *Result {
	out := b.run(suffix)
	for i := 1; i < b.Count; i++ {
		r := b.run(suffix)
		if r.Success != out.Success {
			b.Fatalf("target fails inconsistently")
		}
	}
	return out
}

// run runs a single trial for Run.
func (b *Bisect) run(suffix string) *Result {
	random := fmt.Sprint(rand.Uint64())

	// Accept suffix == "v" to mean we need user-visible output.
	visible := ""
	if suffix == "v" {
		visible = "v"
		suffix = ""
	}

	// Construct change ID pattern.
	var pattern string
	if suffix == "y" || suffix == "n" {
		pattern = suffix
		suffix = ""
	} else {
		var elem []string
		if suffix != "" {
			elem = append(elem, "+", suffix)
		}
		for _, x := range b.Add {
			elem = append(elem, "+", x)
		}
		for _, x := range b.Skip {
			elem = append(elem, "-", x)
		}
		pattern = strings.Join(elem, "")
		if pattern == "" {
			pattern = "y"
		}
	}
	if b.Disable {
		pattern = "!" + pattern
	}
	pattern = visible + pattern

	// Construct substituted env and args.
	env := make([]string, len(b.Env))
	for i, x := range b.Env {
		k, v, _ := strings.Cut(x, "=")
		env[i] = k + "=" + replace(v, pattern, random)
	}
	args := make([]string, len(b.Args))
	for i, x := range b.Args {
		args[i] = replace(x, pattern, random)
	}

	// Construct and log command line.
	// There is no newline in the log print.
	// The line will be completed when the command finishes.
	cmdText := strings.Join(append(append(env, b.Cmd), args...), " ")
	fmt.Fprintf(b.Stderr, "bisect: run: %s...", cmdText)

	// Run command with args and env.
	var out []byte
	var err error
	if b.TestRun != nil {
		out, err = b.TestRun(env, b.Cmd, args)
	} else {
		ctx := context.Background()
		if b.Timeout != 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, b.Timeout)
			defer cancel()
		}
		cmd := exec.CommandContext(ctx, b.Cmd, args...)
		cmd.Env = append(os.Environ(), env...)
		// Set up cmd.Cancel, cmd.WaitDelay on Go 1.20 and later
		// TODO(rsc): Inline go120.go's cmdInterrupt once we stop supporting Go 1.19.
		cmdInterrupt(cmd)
		out, err = cmd.CombinedOutput()
	}

	// Parse output to construct result.
	r := &Result{
		Suffix:  suffix,
		Success: err == nil,
		Cmd:     cmdText,
		Out:     string(out),
	}

	// Calculate bits, mask to identify suffix matches.
	var bits, mask uint64
	if suffix != "" && suffix != "y" && suffix != "n" && suffix != "v" {
		var err error
		bits, err = strconv.ParseUint(suffix, 2, 64)
		if err != nil {
			b.Fatalf("internal error: bad suffix")
		}
		mask = uint64(1<<len(suffix)) - 1
	}

	// Process output, collecting match reports for suffix.
	have := make(map[uint64]bool)
	all := r.Out
	for all != "" {
		var line string
		line, all, _ = strings.Cut(all, "\n")
		short, id, ok := bisect.CutMarker(line)
		if !ok || (id&mask) != bits {
			continue
		}

		if !have[id] {
			have[id] = true
			r.MatchIDs = append(r.MatchIDs, id)
		}
		r.MatchText = append(r.MatchText, short)
		r.MatchFull = append(r.MatchFull, line)
	}

	// Finish log print from above, describing the command's completion.
	if err == nil {
		fmt.Fprintf(b.Stderr, " ok (%d matches)\n", len(r.MatchIDs))
	} else {
		fmt.Fprintf(b.Stderr, " FAIL (%d matches)\n", len(r.MatchIDs))
	}

	if err != nil && len(r.MatchIDs) == 0 {
		b.Fatalf("target failed without printing any matches\n%s", r.Out)
	}

	// In verbose mode, print extra debugging: all the lines with match markers.
	if b.Verbose {
		b.Logf("matches:\n%s", strings.Join(r.MatchFull, "\n\t"))
	}

	return r
}

// replace returns x with literal text PATTERN and RANDOM replaced by pattern and random.
func replace(x, pattern, random string) string {
	x = strings.ReplaceAll(x, "PATTERN", pattern)
	x = strings.ReplaceAll(x, "RANDOM", random)
	return x
}

// commonSuffix returns the longest common binary suffix shared by all uint64s in list.
// If list is empty, commonSuffix returns an empty string.
func commonSuffix(list []uint64) string {
	if len(list) == 0 {
		return ""
	}
	b := list[0]
	n := 64
	for _, x := range list {
		for x&((1<<n)-1) != b {
			n--
			b &= (1 << n) - 1
		}
	}
	s := make([]byte, n)
	for i := n - 1; i >= 0; i-- {
		s[i] = '0' + byte(b&1)
		b >>= 1
	}
	return string(s[:])
}
