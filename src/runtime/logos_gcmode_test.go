// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"internal/testenv"
)

// M19-S1 attacks (SPEC Sec 19, T-301.2, red-team row #166). The helper
// program testdata/testlogosgc is built with go build -gc=MODE and
// logos build -gc=MODE and run under the memory limits each attack needs.
// Epsilon never starts a collection. Batch starts none before heap use
// reaches 90% of GOMEMLIMIT (physical memory when it is unset), then
// stops the world for the collection, and its barrier-free stores leave a
// heap that gccheckmark accepts.

const logosGCMiB = 1 << 20

// logosGCTools lists the commands every attack builds with.
var logosGCTools = []string{"go", "logos"}

var logosGCBuilds struct {
	sync.Mutex
	dir    string
	target map[string]*logosGCBuild
}

type logosGCBuild struct {
	once sync.Once
	id   int
	exe  string
	err  error
}

// logosGCToolPath resolves the go or logos command of the toolchain under
// test; both sit next to each other in GOROOT/bin.
func logosGCToolPath(t *testing.T, tool string) string {
	t.Helper()
	goTool := testenv.GoToolPath(t)
	if tool == "go" {
		return goTool
	}
	path := filepath.Join(filepath.Dir(goTool), "logos"+filepath.Ext(goTool))
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("logos command missing next to go: %v", err)
	}
	return path
}

// logosGCPrefix is the message prefix of tool.
func logosGCPrefix(tool string) string { return tool + " build:" }

// logosGCEnv returns the environment of a child: the parent's without the
// collector knobs, plus the given settings.
func logosGCEnv(set ...string) []string {
	var env []string
	for _, kv := range os.Environ() {
		switch {
		case strings.HasPrefix(kv, "GOGC="), strings.HasPrefix(kv, "GOMEMLIMIT="),
			strings.HasPrefix(kv, "GODEBUG="), strings.HasPrefix(kv, "GOMAXPROCS="):
		default:
			env = append(env, kv)
		}
	}
	return append(env, set...)
}

// logosGCCommand runs tool with args in the helper program directory.
func logosGCCommand(t *testing.T, tool string, env []string, args ...string) (stdout, stderr string, err error) {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	if *flagQuick {
		t.Skip("-quick")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, logosGCToolPath(t, tool), args...)
	cmd.Dir = "testdata/testlogosgc"
	cmd.Env = env
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	err = cmd.Run()
	return out.String(), errb.String(), err
}

// logosGCBuildExe builds the helper program once per tool and flag set.
func logosGCBuildExe(t *testing.T, tool string, flags ...string) string {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	if *flagQuick {
		t.Skip("-quick")
	}
	logosGCBuilds.Lock()
	if logosGCBuilds.dir == "" {
		dir, err := os.MkdirTemp("", "logos-gcmode")
		if err != nil {
			logosGCBuilds.Unlock()
			t.Fatal(err)
		}
		logosGCBuilds.dir = dir
		logosGCBuilds.target = make(map[string]*logosGCBuild)
		testprog.Lock()
		toRemove = append(toRemove, dir)
		testprog.Unlock()
	}
	key := tool + " " + strings.Join(flags, " ")
	b := logosGCBuilds.target[key]
	if b == nil {
		b = &logosGCBuild{id: len(logosGCBuilds.target)}
		logosGCBuilds.target[key] = b
	}
	dir := logosGCBuilds.dir
	logosGCBuilds.Unlock()

	b.once.Do(func() {
		b.err = errors.New("build did not complete")
		exe := filepath.Join(dir, fmt.Sprintf("%s-%d.exe", tool, b.id))
		args := append([]string{"build", "-o", exe}, flags...)
		_, stderr, err := logosGCCommand(t, tool, logosGCEnv(), args...)
		if err != nil {
			b.err = fmt.Errorf("%s %v: %v\n%s", tool, args, err, stderr)
			return
		}
		b.exe, b.err = exe, nil
	})
	if b.err != nil {
		t.Fatal(b.err)
	}
	return b.exe
}

// logosGCRun runs a built helper and returns its output; a failing run
// fails the test.
func logosGCRun(t *testing.T, exe string, env []string, args ...string) (stdout, stderr string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Env = env
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		t.Fatalf("%s %v: %v\nstdout:\n%s\nstderr:\n%s", exe, args, err, out.String(), errb.String())
	}
	return out.String(), errb.String()
}

// logosGCFields returns the groups of the first match of re in out as
// integers.
func logosGCFields(t *testing.T, out string, re *regexp.Regexp) []uint64 {
	t.Helper()
	m := re.FindStringSubmatch(out)
	if m == nil {
		t.Fatalf("no line matching %v in output:\n%s", re, out)
	}
	vals := make([]uint64, len(m)-1)
	for i, s := range m[1:] {
		v, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			t.Fatalf("field %d of %q: %v", i, m[0], err)
		}
		vals[i] = v
	}
	return vals
}

// TestLogosGCModeEpsilonNeverCollects: allocating 120 MiB with GOMEMLIMIT
// unset starts no collection under -gc=epsilon, not even after GOGC is
// raised at run time. Only an explicit runtime.GC collects, as the API
// promises. The same program under the upstream collector collects.
func TestLogosGCModeEpsilonNeverCollects(t *testing.T) {
	re := regexp.MustCompile(`numgc=(\d+) raised=(\d+) explicit=(\d+)`)
	for _, tool := range logosGCTools {
		t.Run(tool, func(t *testing.T) {
			exe := logosGCBuildExe(t, tool, "-gc=epsilon")
			out, _ := logosGCRun(t, exe, logosGCEnv(), "epsilon")
			got := logosGCFields(t, out, re)
			if got[0] != 0 || got[1] != 0 {
				t.Errorf("epsilon collected on its own: NumGC %d after 120 MiB, %d after raising GOGC; want 0 and 0", got[0], got[1])
			}
			if got[2] != 1 {
				t.Errorf("NumGC after one explicit runtime.GC = %d, want 1", got[2])
			}
		})
	}
	t.Run("control", func(t *testing.T) {
		exe := logosGCBuildExe(t, "go")
		out, _ := logosGCRun(t, exe, logosGCEnv(), "epsilon")
		if got := logosGCFields(t, out, re); got[0] == 0 {
			t.Errorf("upstream collector ran no GC over 120 MiB; the epsilon check cannot discriminate")
		}
	})
}

// TestLogosGCModeBatchTriggersAtNinetyPercent (#166): with GOMEMLIMIT set
// and GOGC=100 in the environment, batch runs no collection until heap use
// reaches 90% of the limit; the collection that starts there stops the
// world, which the pacer shows as one dedicated worker per P.
func TestLogosGCModeBatchTriggersAtNinetyPercent(t *testing.T) {
	const limit = 256 * logosGCMiB
	eighty := regexp.MustCompile(`eighty numgc=(\d+) heap=(\d+)`)
	trigger := regexp.MustCompile(`trigger numgc=(\d+) before=(\d+) limit=(\d+)`)
	workers := regexp.MustCompile(`pacer: assist ratio=\S+ .* workers=(\d+)\+`)
	env := logosGCEnv("GOMEMLIMIT=256MiB", "GOGC=100", "GOMAXPROCS=4", "GODEBUG=gcpacertrace=1")
	for _, tool := range logosGCTools {
		t.Run(tool, func(t *testing.T) {
			exe := logosGCBuildExe(t, tool, "-gc=batch")
			out, stderr := logosGCRun(t, exe, env, "batch")
			if got := logosGCFields(t, out, eighty); got[0] != 0 || got[1] < limit/10*8 {
				t.Errorf("at 80%% of the limit: NumGC=%d HeapAlloc=%d; want no collection with HeapAlloc >= %d", got[0], got[1], limit/10*8)
			}
			got := logosGCFields(t, out, trigger)
			if got[0] == 0 {
				t.Fatalf("batch started no collection within 2x the limit:\n%s", out)
			}
			if lo, hi := uint64(limit/100*85), uint64(limit/100*95); got[1] < lo || got[1] > hi {
				t.Errorf("first collection started at heap %d, want within [%d, %d] around 90%% of %d", got[1], lo, hi, limit)
			}
			lines := workers.FindAllStringSubmatch(stderr, -1)
			if len(lines) == 0 {
				t.Fatalf("no pacer trace line in stderr:\n%s", stderr)
			}
			for _, l := range lines {
				if l[1] != "4" {
					t.Errorf("collection ran %s dedicated workers on 4 Ps, want 4 (stop the world): %s", l[1], l[0])
				}
			}
		})
	}
	t.Run("control", func(t *testing.T) {
		exe := logosGCBuildExe(t, "go")
		out, _ := logosGCRun(t, exe, env, "batch")
		if got := logosGCFields(t, out, eighty); got[0] == 0 {
			t.Errorf("upstream collector ran no GC before 80%% of the limit; the batch check cannot discriminate")
		}
	})
}

// TestLogosGCModeBatchHeapCorrect: stores compiled without write barriers
// leave a correct heap across automatic and explicit collections taken
// while several goroutines mutate it, and gccheckmark=1 stays silent. Under
// epsilon only the two explicit collections run.
func TestLogosGCModeBatchHeapCorrect(t *testing.T) {
	re := regexp.MustCompile(`ok numgc=(\d+)`)
	cases := []struct {
		mode   string
		env    []string
		rounds string
		check  func(numgc uint64) bool
		want   string
	}{
		{"batch", logosGCEnv("GOMEMLIMIT=64MiB", "GOMAXPROCS=4", "GODEBUG=gccheckmark=1"), "200",
			func(n uint64) bool { return n >= 5 }, "at least 5 (2 explicit and 3 automatic)"},
		{"epsilon", logosGCEnv("GOMAXPROCS=4", "GODEBUG=gccheckmark=1"), "20",
			func(n uint64) bool { return n == 2 }, "2 (explicit only)"},
	}
	for _, tc := range cases {
		for _, tool := range logosGCTools {
			t.Run(tc.mode+"/"+tool, func(t *testing.T) {
				exe := logosGCBuildExe(t, tool, "-gc="+tc.mode)
				out, stderr := logosGCRun(t, exe, tc.env, "graph", tc.rounds)
				if got := logosGCFields(t, out, re); !tc.check(got[0]) {
					t.Errorf("NumGC=%d, want %s", got[0], tc.want)
				}
				if stderr != "" {
					t.Errorf("gccheckmark=1 was not silent:\n%s", stderr)
				}
			})
		}
	}
}

// TestLogosGCModeStoresPlain (#166): a pointer store compiles to a plain
// move under batch and epsilon, and keeps its write barrier under the
// upstream collector.
func TestLogosGCModeStoresPlain(t *testing.T) {
	store := func(t *testing.T, tool string, flags ...string) string {
		t.Helper()
		exe := filepath.Join(t.TempDir(), "gcasm.exe")
		args := append([]string{"build", "-o", exe, "-gcflags=-S"}, flags...)
		stdout, stderr, err := logosGCCommand(t, tool, logosGCEnv(), args...)
		if err != nil {
			t.Fatalf("%s %v: %v\n%s", tool, args, err, stderr)
		}
		out := stdout + stderr
		_, rest, ok := strings.Cut(out, "main.storePtr STEXT")
		if !ok {
			t.Fatalf("no assembly for main.storePtr in:\n%s", out)
		}
		body, _, _ := strings.Cut(rest, "STEXT")
		return body
	}
	for _, tool := range logosGCTools {
		t.Run(tool, func(t *testing.T) {
			for _, mode := range []string{"batch", "epsilon"} {
				body := store(t, tool, "-gc="+mode)
				if strings.Contains(body, "writeBarrier") || strings.Contains(body, "gcWriteBarrier") {
					t.Errorf("-gc=%s storePtr has a write barrier:\n%s", mode, body)
				}
			}
		})
	}
	if body := store(t, "go"); !strings.Contains(body, "writeBarrier") {
		t.Errorf("upstream storePtr has no write barrier; the plain-store check cannot discriminate:\n%s", body)
	}
}

// logosGCPlan lists the commands tool would run to build the helper under
// the given environment and flags.
func logosGCPlan(t *testing.T, tool string, env []string, flags ...string) string {
	t.Helper()
	args := append([]string{"build", "-n", "-a", "-o", filepath.Join(t.TempDir(), "plan.exe")}, flags...)
	stdout, stderr, err := logosGCCommand(t, tool, env, args...)
	if err != nil {
		t.Fatalf("%s %v: %v\n%s", tool, args, err, stderr)
	}
	return stdout + stderr
}

// logosGCCompileRE extracts the compiled package from one `build -n` plan
// line. Tool lines are printed by work.Shell.runOut as an optional
// `NAME='value'` env prefix (single-quoted, or Go double-quoted when the
// value holds a quote) followed by joinUnambiguously(cmdline), whose tool
// token follows quoted.Split conventions (bare, or ”-/" "-quoted when the
// path holds spaces). The env group mirrors that printing; the tool group
// accepts all three token forms.
var logosGCCompileRE = regexp.MustCompile(`(?m)^(?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|"(?:[^"\\]|\\.)*"|\S+) )*(?:"[^"\n]*?/compile"|'[^'\n]*?/compile'|\S*?/compile) .* -p (\S+) .*$`)

func TestLogosGCCompileRE(t *testing.T) {
	match := []struct{ line, pkg string }{
		{`/pkg/tool/compile -o $WORK/b1 -p fmt -wb=false -complete`, "fmt"},
		{`GOAMD64='v3' GOARM64='v8.2' /pkg/tool/compile -o $WORK/b2 -p runtime -complete`, "runtime"},
		{`GOAMD64='v3' "/tmp/with space/compile" -o $WORK/b3 -p main -wb=false`, "main"},
		{`'/tmp/with space/compile' -o $WORK/b4 -p os -wb=false`, "os"},
		{`GOOS='linux' GOARCH='amd64' GOAMD64='v3' /pkg/tool/compile -o $WORK/b5 -p sync -wb=false`, "sync"},
	}
	for _, tc := range match {
		m := logosGCCompileRE.FindAllStringSubmatch(tc.line, -1)
		if len(m) != 1 || m[0][1] != tc.pkg {
			t.Errorf("line %q: want one -p %s match, got %q", tc.line, tc.pkg, m)
			continue
		}
		if strings.Contains(tc.line, " -wb=false") != strings.Contains(m[0][0], " -wb=false") {
			t.Errorf("line %q: -wb=false visibility changed through match", tc.line)
		}
	}
	nomatch := []string{
		`/pkg/tool/link -o $WORK/x -importcfg $WORK/importcfg -X=runtime.logosGCMode=batch`,
		`GOAMD64='v3' /pkg/tool/link -o $WORK/x -importcfg $WORK/importcfg`,
		`/pkg/tool/compile -o $WORK/b6 -complete`,
		`GOAMD64='v3' /pkg/tool/compile -o $WORK/b7 -complete`,
		`cd /tmp/testlogosgc`,
		`GOAMD64='v3' /pkg/tool/vet -p fmt check`,
	}
	for _, line := range nomatch {
		if m := logosGCCompileRE.FindAllStringSubmatch(line, -1); len(m) != 0 {
			t.Errorf("line %q: want no match, got %q", line, m)
		}
	}
}

// TestLogosGCModeCompileFlags: -gc=batch and -gc=epsilon compile every
// package outside the runtime and its internal dependencies with -wb=false
// and link with -X=runtime.logosGCMode=MODE; the runtime keeps barriers.
func TestLogosGCModeCompileFlags(t *testing.T) {
	keep := []string{"runtime", "internal/abi", "internal/bytealg", "internal/cpu", "internal/goarch", "internal/stringslite",
		"internal/runtime/atomic", "internal/runtime/gc", "internal/runtime/maps", "internal/runtime/math", "internal/runtime/sys"}
	drop := []string{"fmt", "os", "sync", "reflect", "strconv", "main"}
	for _, tool := range logosGCTools {
		for _, mode := range []string{"batch", "epsilon"} {
			t.Run(tool+"/"+mode, func(t *testing.T) {
				plan := logosGCPlan(t, tool, logosGCEnv(), "-gc="+mode)
				if !strings.Contains(plan, " -X=runtime.logosGCMode="+mode) {
					t.Errorf("link line lacks -X=runtime.logosGCMode=%s", mode)
				}
				seen := make(map[string]bool)
				for _, m := range logosGCCompileRE.FindAllStringSubmatch(plan, -1) {
					seen[m[1]] = true
					has := strings.Contains(m[0], " -wb=false")
					if slices.Contains(keep, m[1]) && has {
						t.Errorf("package %s of the runtime closure compiles with -wb=false", m[1])
					}
					if slices.Contains(drop, m[1]) && !has {
						t.Errorf("package %s compiles with write barriers", m[1])
					}
				}
				for _, p := range append(slices.Clone(keep), drop...) {
					if !seen[p] {
						t.Errorf("plan compiles no package %s", p)
					}
				}
			})
		}
	}
	t.Run("go/default", func(t *testing.T) {
		plan := logosGCPlan(t, "go", logosGCEnv())
		if strings.Contains(plan, "-wb=false") || strings.Contains(plan, "logosGCMode") {
			t.Errorf("go build without -gc keeps the upstream collector, but the plan mentions the mode")
		}
	})
}

// TestLogosGCModeSelection: every mode is accepted with -gc, logos
// defaults to greentea (wasm on GOARCH=wasm), unknown modes and the
// unfinished moving modes fail with the pinned text and the command prefix.
func TestLogosGCModeSelection(t *testing.T) {
	for _, mode := range []string{"greentea", "wasm", "batch", "epsilon"} {
		for _, tool := range logosGCTools {
			t.Run("accept/"+tool+"/"+mode, func(t *testing.T) {
				plan := logosGCPlan(t, tool, logosGCEnv(), "-gc="+mode)
				if !strings.Contains(plan, " -X=runtime.logosGCMode="+mode) {
					t.Errorf("-gc=%s did not reach the linker", mode)
				}
			})
		}
	}
	t.Run("default/logos", func(t *testing.T) {
		plan := logosGCPlan(t, "logos", logosGCEnv())
		if !strings.Contains(plan, " -X=runtime.logosGCMode=greentea") || strings.Contains(plan, "-wb=false") {
			t.Errorf("logos default: want -X=runtime.logosGCMode=greentea and compiled barriers")
		}
	})
	t.Run("default/logos/wasm", func(t *testing.T) {
		plan := logosGCPlan(t, "logos", logosGCEnv("GOOS=js", "GOARCH=wasm", "GOEXPERIMENT="))
		if !strings.Contains(plan, " -X=runtime.logosGCMode=wasm") {
			t.Errorf("logos default on GOARCH=wasm: want -X=runtime.logosGCMode=wasm")
		}
	})
	wants := map[string]string{
		"x":      `unknown -gc mode "x" (want greentea, zgc, immix, wasm, batch, epsilon)`,
		"":       `unknown -gc mode "" (want greentea, zgc, immix, wasm, batch, epsilon)`,
		"immix":  `-gc=immix requires GOEXPERIMENT=logosgcmoving phase 1, which is not complete`,
		"zgc":    `-gc=zgc requires GOEXPERIMENT=logosgcmoving phase 2, which is not complete`,
		"Batch":  `unknown -gc mode "Batch" (want greentea, zgc, immix, wasm, batch, epsilon)`,
		"batch ": `unknown -gc mode "batch " (want greentea, zgc, immix, wasm, batch, epsilon)`,
	}
	for mode, want := range wants {
		for _, tool := range logosGCTools {
			t.Run("reject/"+tool+"/"+mode, func(t *testing.T) {
				_, stderr, err := logosGCCommand(t, tool, logosGCEnv(), "build", "-n", "-o", filepath.Join(t.TempDir(), "x.exe"), "-gc="+mode)
				if err == nil {
					t.Fatalf("%s build -gc=%q succeeded", tool, mode)
				}
				if line := logosGCPrefix(tool) + " " + want; !strings.Contains(stderr, line) {
					t.Errorf("stderr lacks %q:\n%s", line, stderr)
				}
			})
		}
	}
}

// TestLogosBatchLimitAndTrigger: the batch trigger is exactly 90% of
// GOMEMLIMIT, or of physical memory when GOMEMLIMIT is unset, while the live
// heap is below it. A live heap at or above that mark moves the trigger past
// it by a growth floor, so a heap that stays above the mark does not collect
// on every refill.
func TestLogosBatchLimitAndTrigger(t *testing.T) {
	const phys = 16 << 30
	for _, tc := range []struct {
		name  string
		limit int64
		want  uint64
	}{
		{"set", 256 * logosGCMiB, 256 * logosGCMiB},
		{"zero", 0, 0},
		{"unset", math.MaxInt64, phys},
	} {
		if got := runtime.LogosBatchLimit(tc.limit, phys); got != tc.want {
			t.Errorf("%s: limit = %d, want %d", tc.name, got, tc.want)
		}
	}
	if got := runtime.LogosBatchLimit(math.MaxInt64, 0); got != math.MaxUint64 {
		t.Errorf("unset limit with unknown physical memory = %d, want no trigger (%d)", got, uint64(math.MaxUint64))
	}
	heapMin := uint64(runtime.LogosDefaultHeapMinimum)
	for _, tc := range []struct {
		name          string
		limit, marked uint64
		want          uint64
	}{
		{"empty heap", 256 * logosGCMiB, 0, 256 * logosGCMiB / 10 * 9},
		{"live below mark", 256 * logosGCMiB, 100 * logosGCMiB, 256 * logosGCMiB / 10 * 9},
		{"live just below mark", 256 * logosGCMiB, 256*logosGCMiB/10*9 - 1, 256 * logosGCMiB / 10 * 9},
		{"live at mark", 256 * logosGCMiB, 256 * logosGCMiB / 10 * 9, 256*logosGCMiB/10*9 + 256*logosGCMiB/10*9/4},
		{"live above mark", 256 * logosGCMiB, 240 * logosGCMiB, 300 * logosGCMiB},
		{"tiny limit", 1 * logosGCMiB, 0, 1 * logosGCMiB / 10 * 9},
		{"zero limit", 0, 0, heapMin},
		{"zero limit live", 0, 64 * logosGCMiB, 80 * logosGCMiB},
		{"physical", phys, 0, phys / 10 * 9},
		{"unknown physical", math.MaxUint64, 0, math.MaxUint64 / 10 * 9},
	} {
		if got := runtime.LogosBatchTrigger(tc.limit, tc.marked); got != tc.want {
			t.Errorf("%s: trigger = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// TestLogosModeRefusals: epsilon refuses every collection but an explicit
// runtime.GC, batch refuses the periodic one (its heap trigger is the only
// automatic start), and the other modes and plain go refuse none.
func TestLogosModeRefusals(t *testing.T) {
	for _, tc := range []struct {
		mode                     string
		heap, periodic, explicit bool
	}{
		{"epsilon", true, true, false},
		{"batch", false, true, false},
		{"greentea", false, false, false},
		{"wasm", false, false, false},
		{"", false, false, false},
	} {
		heap, periodic, explicit := runtime.LogosModeRefuses(tc.mode)
		if heap != tc.heap || periodic != tc.periodic || explicit != tc.explicit {
			t.Errorf("mode %q refuses heap=%v periodic=%v explicit=%v, want %v %v %v",
				tc.mode, heap, periodic, explicit, tc.heap, tc.periodic, tc.explicit)
		}
	}
}

// TestLogosSanePhysMem: a physical memory reading below 64 MiB is a failed
// read, so batch treats it as unknown.
func TestLogosSanePhysMem(t *testing.T) {
	for _, tc := range []struct{ in, want uint64 }{
		{0, 0},
		{4096, 0},
		{64<<20 - 1, 0},
		{64 << 20, 64 << 20},
		{16 << 30, 16 << 30},
	} {
		if got := runtime.LogosSanePhysMem(tc.in); got != tc.want {
			t.Errorf("LogosSanePhysMem(%d) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

// TestLogosParseUint: the number parser behind /proc/meminfo and /dev/swap
// skips leading spaces, stops at the first non-digit, and reports no digits
// as 0.
func TestLogosParseUint(t *testing.T) {
	for _, tc := range []struct {
		in     string
		want   uint64
		digits int
	}{
		{"16384 kB\n", 16384, 5},
		{"   8192 kB", 8192, 4},
		{"2147483648 memory\n4096 pagesize\n", 2147483648, 10},
		{"0", 0, 1},
		{"", 0, 0},
		{"kB", 0, 0},
		{"  ", 0, 0},
		{"-5", 0, 0},
		{"18446744073709551615", math.MaxUint64, 20},
	} {
		got, digits := runtime.LogosParseUint([]byte(tc.in))
		if got != tc.want || digits != tc.digits {
			t.Errorf("parse %q = %d, %d digits; want %d, %d digits", tc.in, got, digits, tc.want, tc.digits)
		}
	}
}

// TestLogosPhysicalMemory: the runtime reads the machine's physical memory
// the way the platform reports it.
func TestLogosPhysicalMemory(t *testing.T) {
	got := runtime.LogosPhysicalMemory()
	switch runtime.GOOS {
	case "linux", "android":
		data, err := os.ReadFile("/proc/meminfo")
		if err != nil {
			t.Skipf("no /proc/meminfo: %v", err)
		}
		var kb uint64
		if _, err := fmt.Sscanf(string(data), "MemTotal: %d kB", &kb); err != nil {
			t.Fatalf("parse /proc/meminfo: %v", err)
		}
		if got != kb<<10 {
			t.Errorf("physical memory = %d, want MemTotal %d", got, kb<<10)
		}
	case "js", "wasip1":
		if got != 1<<32 {
			t.Errorf("physical memory = %d on %s, want the wasm32 limit %d", got, runtime.GOOS, uint64(1<<32))
		}
	case "aix":
		if got != 0 {
			t.Errorf("physical memory = %d on aix, want 0 (no source)", got)
		}
	default:
		if got < 64<<20 {
			t.Errorf("physical memory = %d on %s, want at least 64 MiB", got, runtime.GOOS)
		}
	}
}
