// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package goparity holds the offensive differential harness for SPEC §0.1
// rules R-GO ("The go command MUST keep Go language semantics") and R-LOGOS
// ("The logos command MUST build Go modules with Go language semantics").
//
// # Corpus layout
//
// One program per directory: testdata/run/<id>_<name>/ with <id> the design's
// attack ID lower-cased without the dash (at01). Every directory is package
// main inside the staged module example.com/parity (go 1.27); extra packages
// of one program live in subdirectories and are imported as
// example.com/parity/<id>_<name>/<pkg>. A directory may carry a generator
// gen.go with //go:build ignore; the harness runs `go run gen.go` inside the
// staged copy before building and generator output is never committed.
// Invalid-Go cases live in testdata/fail/<id>_<name>.go, one defect per file.
//
// # Program contract
//
// A program reads no input, uses no network and no wall clock in its output,
// prints deterministic text, and exits 0. A broken law prints a line with the
// word LAW on stdout or stderr and exits 1, and the harness fails a program
// that does either. Programs must build and run identically under
// upstream Go: they may not use the logos build tag, //logos: directives,
// runtime.Version(), loGo/s packages, or any sanctioned SPEC §0.1 difference.
// They are float-free or pass every float result through an explicit
// conversion (fusion-free), print decimal values and never pointers, and use
// deterministic synchronization only.
//
// # Oracles
//
// Tier U compares each fork configuration against the pinned upstream oracle
// (flag -goparity.upstream, default $UPROOT1274/bin/go). The fork VERSION and
// the oracle VERSION are pinned exactly; any other pair fails with a re-audit
// message. A missing oracle fails the test (owner answer Q2) unless
// -goparity.upstream=off is passed, in which case only Tier M runs and the
// fail corpus is skipped. Tier M asserts metamorphic equality across fork
// configurations with equal effective GOAMD64.
//
// # Configurations
//
// Fast tier (default): K1 go, K3 go -O=3, K4 go -O=3 GOAMD64=v3, K5 logos
// default, K6 logos -O=0, K7 logos -O=3, K8 logos -O=3 arena=false, K10
// logos -O=2 target_cpu=baseline against U1, U1N, U3, U3N. -short runs K1,
// K5, K8, U1, U3. Release tier (flag -goparity.release) adds K2 go -O=0,
// K11 go -O=2, K12 go -O=s, K13 logos -O=1, K14 logos -O=s, K15 logos -O=2
// arena=false, K16 logos -race, and the GOEXPERIMENT rows K17 logosarena,
// K18 logosescape, K19 logosstack, K20 logosfastcgo, K21 preemptibleloops
// on plain go (owner answer Q4). Fork rows compare against the matched
// upstream row (U1 v1, U1N v1 with -gcflags=all=-N -l, U3 v3, U3N v3 with
// -N -l, U3R v3 with -race). Programs importing "C" build with CGO_ENABLED=1
// as an explicit package list; in the fast tier they run under K1, K5, U1,
// and U3 only, in release under every active configuration. A program may
// declare level-dependent stdout lines (currently the unconverted float
// sums that upstream itself fuses at v3) in level-dependent.txt, one line
// prefix per line; those lines compare fork-vs-matched-upstream as usual
// but the invariance and metamorphic gates hold them only within the same
// GOAMD64 level and optimization setting. Gates before comparison: every
// program must give equal observations under all upstream rows present
// (invariance gate) and under upstream U1 at GOMAXPROCS=1 and 4 (stability
// gate); a violation is a corpus defect, never a silent skip.
//
// # Mutants
//
// testdata/mutants/<id>.txt holds one or more negative controls per attack
// in txtar form: header lines kind, file, expect, config, then one -- old --
// / -- new -- pair with the exact replaced text; a later control starts at
// the next column-zero kind line. file is GOROOT-relative; old must occur
// exactly once in the target or the mutant is invalid and the test fails
// loudly, so stale mutants are never silent. kinds compiler, linker, cgo,
// and rules build the named tool through an overlay and swap it into fork
// builds only via -toolexec with the shim in testdata/toolexec (env
// GOPARITY_MUTANT_TOOL=name=path); kinds runtime and std pass the overlay
// to the fork build directly. kind cmdgo is not supported here.
// Mutants never touch the tree and never reach upstream builds.
// TestParityMutants (release tier, flag -goparity.release) applies each
// mutant, requires the build to succeed, and requires the program named
// by expect to turn red in the named configuration: nonzero exit, a LAW
// line, or a difference from the matched upstream row; an expect naming a
// fail-corpus file turns red when the fork build stops rejecting it.
//
// Build caches live under os.TempDir() (goparity-fork-gocache,
// goparity-upstream-gocache, goparity-mutant-<id>-gocache) and persist
// across runs; toolexec mutant builds always use their own cache so a
// mutant compiler can never be served clean cache entries. Program runs are
// bounded by 10 s (50 s with -race) at GOMAXPROCS=4 with empty GODEBUG and GOTRACEBACK=single.
// Stdout is compared byte for byte; stderr is cut at the first traceback
// line and masked only for pc=, addr=, sp=, fp= and lr= values and 0xc...
// pointers (the upstream sweep still masks every 0x value of six or more
// digits). Under cmd/dist the package is discovered by `go list std cmd` and
// runs in the batched "Testing packages." pass with the per-binary
// -timeout=180s budget, which the fast tier must fit.
//
// # Release tier and the gate
//
// cmd/dist and the default gate run the fast tier only. The release tier,
// which holds the release configurations, TestParityMutants and the upstream
// sweep, runs on a quiet host with
//
//	go test -count=1 -timeout 3h cmd/internal/goparity \
//	    -goparity.release -goparity.stamp=FILE
//
// -goparity.release with -short is rejected, since it would run part of the
// tier. -goparity.stamp is accepted only for a complete run: release, oracle
// on, no -short, no -run, no -skip. The harness removes FILE first and
// writes the git tip of the fork tree into it only when every test passed
// and the tree has no uncommitted change to a tracked file. A gate that needs
// the release tier compares FILE with git rev-parse HEAD and fails when FILE
// is absent or differs: the release tier has then not run on the tip.
package goparity

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
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
)

const (
	forkVersionPin     = "go1.27.4"
	upstreamVersionPin = "go1.27.4"
)

var (
	upstreamFlag = flag.String("goparity.upstream", "", `upstream oracle go binary, or "off" (default $UPROOT1274/bin/go)`)
	releaseFlag  = flag.Bool("goparity.release", false, "add the release tier configurations and the mutation probes")
)

var stampFlag = flag.String("goparity.stamp", "", "after a complete green release run, write the git tip of the fork tree to this file")

type tierRequest struct {
	release  bool
	short    bool
	upstream string
	run      string
	skip     string
	stamp    string
}

func tierProblem(r tierRequest) string {
	switch {
	case r.release && r.short:
		return "-goparity.release with -short runs only part of the release tier"
	case r.stamp == "":
		return ""
	case !r.release:
		return "-goparity.stamp needs -goparity.release"
	case r.upstream == "off":
		return "-goparity.stamp needs the upstream oracle"
	case r.run != "" || r.skip != "":
		return "-goparity.stamp needs the whole suite, and -run or -skip selects a part"
	}
	return ""
}

func TestTierProblem(t *testing.T) {
	complete := tierRequest{release: true, upstream: "/oracle/bin/go", stamp: "/gate/stamp"}
	with := func(change func(*tierRequest)) tierRequest {
		r := complete
		change(&r)
		return r
	}
	for _, tc := range []struct {
		name string
		req  tierRequest
		bad  bool
	}{
		{"default tier", tierRequest{}, false},
		{"default tier with -short and -run", tierRequest{short: true, run: "TestParity"}, false},
		{"release", tierRequest{release: true}, false},
		{"complete stamped release", complete, false},
		{"release with -short", tierRequest{release: true, short: true}, true},
		{"stamped release with -short", with(func(r *tierRequest) { r.short = true }), true},
		{"stamp without release", tierRequest{stamp: "/gate/stamp"}, true},
		{"stamp with the oracle off", with(func(r *tierRequest) { r.upstream = "off" }), true},
		{"stamp with -run", with(func(r *tierRequest) { r.run = "TestParityMutants" }), true},
		{"stamp with -skip", with(func(r *tierRequest) { r.skip = "TestParityUpstreamSweep" }), true},
	} {
		if got := tierProblem(tc.req) != ""; got != tc.bad {
			t.Errorf("%s: problem reported = %v, want %v", tc.name, got, tc.bad)
		}
	}
}

func flagValue(name string) string {
	if f := flag.Lookup(name); f != nil {
		return f.Value.String()
	}
	return ""
}

func tipIdentity(goroot string) (string, error) {
	status, err := exec.Command("git", "-C", goroot, "status", "--porcelain", "--untracked-files=no").Output()
	if err != nil {
		return "", fmt.Errorf("git status in %s: %w", goroot, err)
	}
	if len(bytes.TrimSpace(status)) != 0 {
		return "", fmt.Errorf("%s has uncommitted changes to tracked files", goroot)
	}
	head, err := exec.Command("git", "-C", goroot, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", fmt.Errorf("git rev-parse in %s: %w", goroot, err)
	}
	return strings.TrimSpace(string(head)), nil
}

func writeStamp(path, goroot string) error {
	tip, err := tipIdentity(goroot)
	if err != nil {
		return err
	}
	return os.WriteFile(path, []byte(tip+"\n"), 0o644)
}

func runTier(m *testing.M, req tierRequest) int {
	if problem := tierProblem(req); problem != "" {
		fmt.Fprintln(os.Stderr, "goparity:", problem)
		return 2
	}
	if req.stamp == "" {
		return m.Run()
	}
	if err := os.Remove(req.stamp); err != nil && !errors.Is(err, fs.ErrNotExist) {
		fmt.Fprintln(os.Stderr, "goparity: removing the old stamp:", err)
		return 2
	}
	if code := m.Run(); code != 0 {
		return code
	}
	if err := writeStamp(req.stamp, forkGoroot()); err != nil {
		fmt.Fprintln(os.Stderr, "goparity: the release run passed but left no stamp:", err)
		return 1
	}
	return 0
}

func TestMain(m *testing.M) {
	flag.Parse()
	os.Exit(runTier(m, tierRequest{
		release:  *releaseFlag,
		short:    testing.Short(),
		upstream: *upstreamFlag,
		run:      flagValue("test.run"),
		skip:     flagValue("test.skip"),
		stamp:    *stampFlag,
	}))
}

func TestTipIdentityNeedsACleanTrackedTree(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not installed")
	}
	dir := t.TempDir()
	git := func(args ...string) string {
		full := append([]string{"-C", dir, "-c", "user.name=goparity", "-c", "user.email=goparity@example.com", "-c", "commit.gpgsign=false"}, args...)
		out, err := exec.Command("git", full...).CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	file := filepath.Join(dir, "tracked")
	if err := os.WriteFile(file, []byte("1"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("init", "-q")
	git("add", "tracked")
	git("commit", "-q", "-m", "tip")
	want := git("rev-parse", "HEAD")
	if err := os.WriteFile(filepath.Join(dir, "untracked"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := tipIdentity(dir); err != nil || got != want {
		t.Errorf("tipIdentity with an untracked file = %q, %v; want %q", got, err, want)
	}
	stamp := filepath.Join(t.TempDir(), "stamp")
	if err := writeStamp(stamp, dir); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(stamp); err != nil || string(got) != want+"\n" {
		t.Errorf("stamp = %q, %v; want %q", got, err, want+"\n")
	}
	if err := os.WriteFile(file, []byte("2"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := tipIdentity(dir); err == nil {
		t.Errorf("tipIdentity of a tree with an uncommitted change = %q, want an error", got)
	}
	if err := writeStamp(filepath.Join(t.TempDir(), "stamp2"), dir); err == nil {
		t.Error("a stamp was written for a tree with an uncommitted change")
	}
}

type config struct {
	id       string
	logos    bool
	args     []string
	env      []string
	toml     string
	upstream string
	group    string
	release  bool
}

type upConfig struct {
	id    string
	args  []string
	env   []string
	race  bool
	level string
	noopt bool
}

var forkConfigs = []config{
	{id: "K1", upstream: "U1", group: "v1"},
	{id: "K3", args: []string{"-O=3"}, upstream: "U1", group: "v1"},
	{id: "K4", args: []string{"-O=3"}, env: []string{"GOAMD64=v3"}, upstream: "U3", group: "v3"},
	{id: "K5", logos: true, upstream: "U3", group: "v3"},
	{id: "K6", logos: true, args: []string{"-O=0"}, upstream: "U3N", group: "v3"},
	{id: "K7", logos: true, args: []string{"-O=3"}, upstream: "U3", group: "v3"},
	{id: "K8", logos: true, args: []string{"-O=3"}, toml: "[build]\narena = false\n", upstream: "U3", group: "v3"},
	{id: "K10", logos: true, args: []string{"-O=2"}, toml: "[build]\ntarget_cpu = \"baseline\"\n", upstream: "U1", group: "v1"},
	{id: "K2", args: []string{"-O=0"}, upstream: "U1N", group: "v1", release: true},
	{id: "K11", args: []string{"-O=2"}, upstream: "U1", group: "v1", release: true},
	{id: "K12", args: []string{"-O=s"}, upstream: "U1", group: "v1", release: true},
	{id: "K13", logos: true, args: []string{"-O=1"}, upstream: "U3", group: "v3", release: true},
	{id: "K14", logos: true, args: []string{"-O=s"}, upstream: "U3", group: "v3", release: true},
	{id: "K15", logos: true, args: []string{"-O=2"}, toml: "[build]\narena = false\n", upstream: "U3", group: "v3", release: true},
	{id: "K16", logos: true, args: []string{"-race"}, env: []string{"CGO_ENABLED=1"}, toml: "[build]\ncgo = \"allow\"\n", upstream: "U3R", group: "v3", release: true},
	{id: "K17", env: []string{"GOEXPERIMENT=logosarena"}, upstream: "U1", group: "v1", release: true},
	{id: "K18", env: []string{"GOEXPERIMENT=logosescape"}, upstream: "U1", group: "v1", release: true},
	{id: "K19", env: []string{"GOEXPERIMENT=logosstack"}, upstream: "U1", group: "v1", release: true},
	{id: "K20", env: []string{"GOEXPERIMENT=logosfastcgo"}, upstream: "U1", group: "v1", release: true},
	{id: "K21", env: []string{"GOEXPERIMENT=preemptibleloops"}, upstream: "U1", group: "v1", release: true},
}

var upConfigs = []upConfig{
	{id: "U1", level: "v1"},
	{id: "U1N", args: []string{"-gcflags=all=-N -l"}, level: "v1", noopt: true},
	{id: "U3", env: []string{"GOAMD64=v3"}, level: "v3"},
	{id: "U3N", args: []string{"-gcflags=all=-N -l"}, env: []string{"GOAMD64=v3"}, level: "v3", noopt: true},
	{id: "U3R", args: []string{"-race"}, env: []string{"GOAMD64=v3", "CGO_ENABLED=1"}, race: true, level: "v3"},
}

type toolchain struct {
	bin    string
	logos  string
	goroot string
}

type observation struct {
	exit   int
	stdout string
	stderr string
}

type buildResult struct {
	outDir string
	output string
	err    error
	progs  []string
}

func toolVersion(bin, goroot string) (string, error) {
	cmd := exec.Command(bin, "env", "GOVERSION")
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + os.TempDir(), "GOROOT=" + goroot, "GOENV=off", "GOTOOLCHAIN=local"}
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("%s env GOVERSION: %w", bin, err)
	}
	return strings.TrimSpace(string(out)), nil
}

func fakeToolchain(t *testing.T, fileVersion, binVersion string) (goroot, bin string) {
	t.Helper()
	goroot = t.TempDir()
	if err := os.MkdirAll(filepath.Join(goroot, "bin"), 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(goroot, "VERSION"), []byte(fileVersion+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	bin = filepath.Join(goroot, "bin", "go")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\necho "+binVersion+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return goroot, bin
}

func TestToolVersionComesFromTheBinary(t *testing.T) {
	for _, tc := range []struct{ name, file, bin string }{
		{"file claims the pin, binary does not", "go1.27.4", "go1.0.0"},
		{"binary reports the pin, file does not", "go1.0.0", "go1.27.4"},
	} {
		goroot, bin := fakeToolchain(t, tc.file, tc.bin)
		if v, err := toolVersion(bin, goroot); err != nil || v != tc.bin {
			t.Errorf("%s: toolVersion = %q, %v; want %q", tc.name, v, err, tc.bin)
		}
	}
}

func TestForkToolIgnoresAStaleLogosBinary(t *testing.T) {
	goroot, bin := fakeToolchain(t, forkVersionPin, forkVersionPin)
	if err := os.WriteFile(filepath.Join(goroot, "bin", "logos"), []byte("#!/bin/sh\necho stale\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOROOT", goroot)
	tc := forkTool(t)
	got, err := os.ReadFile(tc.logos)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(bin)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("the logos binary %s differs from bin/go: a stale bin/logos was used", tc.logos)
	}
}

func forkGoroot() string {
	return cmp.Or(os.Getenv("GOROOT"), runtime.GOROOT())
}

func forkTool(t *testing.T) toolchain {
	t.Helper()
	goroot := forkGoroot()
	if goroot == "" {
		t.Fatal("GOROOT is not set")
	}
	bin := filepath.Join(goroot, "bin", "go")
	if _, err := os.Stat(bin); err != nil {
		t.Fatalf("fork go binary missing: %v", err)
	}
	v, err := toolVersion(bin, goroot)
	if err != nil {
		t.Fatal(err)
	}
	if v != forkVersionPin {
		t.Fatalf("fork version skew not audited: got %s want %s; re-run `git diff --stat %s <oracle> -- src test` and update the pin in parity_test.go", v, forkVersionPin, forkVersionPin)
	}
	logos := filepath.Join(t.TempDir(), "logos")
	cloneBinary(t, bin, logos)
	return toolchain{bin: bin, logos: logos, goroot: goroot}
}

func cloneBinary(t *testing.T, src, dst string) {
	t.Helper()
	if os.Link(src, dst) == nil {
		return
	}
	data, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("reading %s: %v", src, err)
	}
	if err := os.WriteFile(dst, data, 0o755); err != nil {
		t.Fatalf("writing %s: %v", dst, err)
	}
}

func upstreamTool(t *testing.T) (toolchain, bool) {
	t.Helper()
	if *upstreamFlag == "off" {
		return toolchain{}, false
	}
	bin := *upstreamFlag
	if bin == "" {
		goroot := os.Getenv("UPROOT1274")
		if goroot == "" {
			t.Fatalf("upstream oracle missing: $UPROOT1274 is not set (pass -goparity.upstream=off to run the metamorphic tier only)")
		}
		bin = filepath.Join(goroot, "bin", "go")
	}
	if _, err := os.Stat(bin); err != nil {
		t.Fatalf("upstream oracle %s missing: %v (pass -goparity.upstream=off to run the metamorphic tier only)", bin, err)
	}
	goroot := filepath.Clean(filepath.Join(filepath.Dir(bin), ".."))
	v, err := toolVersion(bin, goroot)
	if err != nil {
		t.Fatal(err)
	}
	if v != upstreamVersionPin {
		t.Fatalf("upstream oracle version skew not audited: got %s want %s; re-run `git diff --stat %s <oracle tag> -- src test` and update the pin in parity_test.go", v, upstreamVersionPin, upstreamVersionPin)
	}
	return toolchain{bin: bin, goroot: goroot}, true
}

func baseEnv(cacheDir, goroot string) []string {
	os.MkdirAll(cacheDir, 0o777)
	os.MkdirAll(cacheDir+"-path", 0o777)
	return []string{
		"PATH=/usr/bin:/bin",
		"HOME=" + cacheDir,
		"GOENV=off",
		"GOTOOLCHAIN=local",
		"GOROOT=" + goroot,
		"GOWORK=off",
		"GOPROXY=off",
		"GOMAXPROCS=8",
		"CGO_ENABLED=0",
		"GOCACHE=" + cacheDir,
		"GOTMPDIR=" + os.TempDir(),
		"TMPDIR=" + os.TempDir(),
		"GOPATH=" + cacheDir + "-path",
	}
}

func appendEnv(env []string, kv ...string) []string {
	out := slices.Clone(env)
	return append(out, kv...)
}

func forkCache() string { return filepath.Join(os.TempDir(), "goparity-fork-gocache") }

func upCache() string { return filepath.Join(os.TempDir(), "goparity-upstream-gocache") }

func mutantCache(id string) string {
	return filepath.Join(os.TempDir(), "goparity-mutant-"+id+"-gocache")
}

func corpusPrograms(t *testing.T) []string {
	t.Helper()
	out, err := listPrograms("testdata/run")
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func corpusFails(t *testing.T) []string {
	t.Helper()
	out, err := listFails("testdata/fail")
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func listPrograms(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", dir, err)
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() {
			return nil, fmt.Errorf("%s: %s is not a program directory", dir, e.Name())
		}
		out = append(out, e.Name())
	}
	return out, nil
}

func listFails(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", dir, err)
	}
	var out []string
	for _, e := range entries {
		stem, ok := strings.CutSuffix(e.Name(), ".go")
		if !ok || e.IsDir() {
			return nil, fmt.Errorf("%s: %s is not a fail corpus .go file", dir, e.Name())
		}
		out = append(out, stem)
	}
	return out, nil
}

func TestCorpusDiscoveryRejectsStrayEntries(t *testing.T) {
	layout := func(t *testing.T, files, dirs []string) string {
		dir := t.TempDir()
		for _, f := range files {
			if err := os.WriteFile(filepath.Join(dir, f), nil, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		for _, d := range dirs {
			if err := os.MkdirAll(filepath.Join(dir, d), 0o777); err != nil {
				t.Fatal(err)
			}
		}
		return dir
	}
	if got, err := listPrograms(layout(t, nil, []string{"at02_b", "at01_a"})); err != nil || !slices.Equal(got, []string{"at01_a", "at02_b"}) {
		t.Errorf("programs = %v, %v; want the two directories", got, err)
	}
	if got, err := listFails(layout(t, []string{"at14_b.go", "at14_a.go"}, nil)); err != nil || !slices.Equal(got, []string{"at14_a", "at14_b"}) {
		t.Errorf("fail files = %v, %v; want the two stems", got, err)
	}
	if _, err := listPrograms(layout(t, []string{"at99.go"}, []string{"at01_a"})); err == nil {
		t.Error("a stray file next to the program directories was accepted")
	}
	if _, err := listFails(layout(t, []string{"at14_a.go", "notes.txt"}, nil)); err == nil {
		t.Error("a stray non-Go file among the fail files was accepted")
	}
	if _, err := listFails(layout(t, []string{"at14_a.go"}, []string{"nested"})); err == nil {
		t.Error("a subdirectory among the fail files was accepted")
	}
}

func activeForkConfigs() []config {
	var out []config
	for _, c := range forkConfigs {
		if c.release && !*releaseFlag {
			continue
		}
		if testing.Short() && c.id != "K1" && c.id != "K5" && c.id != "K8" {
			continue
		}
		out = append(out, c)
	}
	return out
}

func copyTree(src, dst string) error {
	entries, err := os.ReadDir(src)
	if err != nil {
		return fmt.Errorf("reading %s: %w", src, err)
	}
	if err := os.MkdirAll(dst, 0o777); err != nil {
		return fmt.Errorf("creating %s: %w", dst, err)
	}
	for _, e := range entries {
		s := filepath.Join(src, e.Name())
		d := filepath.Join(dst, e.Name())
		if e.IsDir() {
			if err := copyTree(s, d); err != nil {
				return err
			}
			continue
		}
		data, err := os.ReadFile(s)
		if err != nil {
			return fmt.Errorf("reading %s: %w", s, err)
		}
		if err := os.WriteFile(d, data, 0o644); err != nil {
			return fmt.Errorf("writing %s: %w", d, err)
		}
	}
	return nil
}

func stageCorpus(tc toolchain, dst, cacheDir string, progs []string) error {
	for _, p := range progs {
		srcDir := filepath.Join("testdata", "run", p)
		if err := copyTree(srcDir, filepath.Join(dst, p)); err != nil {
			return fmt.Errorf("staging %s: %w", p, err)
		}
		if _, err := os.Stat(filepath.Join(srcDir, "gen.go")); err == nil {
			if _, err := runGenerator(tc.bin, p, filepath.Join(dst, p), baseEnv(cacheDir, tc.goroot), generatorBudget); err != nil {
				return err
			}
		}
	}
	if err := os.WriteFile(filepath.Join(dst, "go.mod"), []byte("module example.com/parity\n\ngo 1.27\n"), 0o644); err != nil {
		return fmt.Errorf("writing go.mod: %w", err)
	}
	return nil
}

func goBuild(bin, dir string, args, env []string, budget time.Duration) (string, error) {
	ctx, cancel := context.WithTimeoutCause(context.Background(), budget, errors.New("goparity build timeout"))
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Dir = dir
	cmd.Env = env
	cmd.WaitDelay = pipeWaitDelay
	out, err := cmd.CombinedOutput()
	return string(out), err
}

const generatorBudget = 5 * time.Minute

const pipeWaitDelay = 10 * time.Second

func runGenerator(bin, prog, dir string, env []string, budget time.Duration) (string, error) {
	ctx, cancel := context.WithTimeoutCause(context.Background(), budget, errors.New("goparity generator timeout"))
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "run", "gen.go")
	cmd.Dir = dir
	cmd.Env = env
	cmd.WaitDelay = pipeWaitDelay
	setNewProcessGroup(cmd)
	cmd.Cancel = func() error {
		killProcessGroup(cmd)
		return nil
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			if cmd.ProcessState == nil || !cmd.ProcessState.Exited() {
				killProcessGroup(cmd)
			}
			return string(out), fmt.Errorf("generator for %s timed out: %w", prog, context.Cause(ctx))
		}
		return string(out), fmt.Errorf("generator for %s failed: %v\n%s", prog, err, clip(string(out)))
	}
	return string(out), nil
}

const moduleBuildBudget = 15 * time.Minute

type stagedBuild struct {
	bin     string
	stage   string
	outDir  string
	args    []string
	env     []string
	progs   []string
	wantCgo bool
}

func (b stagedBuild) build() (built []string, output string, err error) {
	build := func(pkgs, env []string) (string, error) {
		if err := os.MkdirAll(b.outDir, 0o777); err != nil {
			return "", err
		}
		full := append([]string{"build", "-o", b.outDir}, b.args...)
		full = append(full, pkgs...)
		return goBuild(b.bin, b.stage, full, env, moduleBuildBudget)
	}
	plain, cgo, err := splitStaged(b.stage, b.progs)
	if err != nil {
		return nil, "", err
	}
	type buildList struct {
		names []string
		env   []string
	}
	lists := []buildList{{plain, b.env}}
	if b.wantCgo {
		lists = append(lists, buildList{cgo, appendEnv(b.env, "CGO_ENABLED=1")})
	}
	for _, l := range lists {
		if len(l.names) == 0 {
			continue
		}
		out, err := build(pkgArgs(l.names), l.env)
		output += out
		if err != nil {
			return built, output, err
		}
		built = append(built, l.names...)
	}
	if len(built) == 0 && err == nil {
		return built, output, errors.New("no programs built")
	}
	return built, output, nil
}

func splitStaged(stage string, progs []string) (plain, cgo []string, err error) {
	for _, p := range progs {
		isCgo, err := cgoProgram(filepath.Join(stage, p))
		if err != nil {
			return nil, nil, err
		}
		if isCgo {
			cgo = append(cgo, p)
		} else {
			plain = append(plain, p)
		}
	}
	return plain, cgo, nil
}

func cgoProgram(dir string) (bool, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false, err
	}
	fset := token.NewFileSet()
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, e.Name()), nil, parser.ImportsOnly)
		if err != nil {
			return false, err
		}
		for _, imp := range f.Imports {
			if imp.Path.Value == `"C"` {
				return true, nil
			}
		}
	}
	return false, nil
}

func TestCgoProgramClassifiesByImport(t *testing.T) {
	for _, tc := range []struct {
		name    string
		src     string
		want    bool
		wantErr bool
	}{
		{name: "import C", src: "package main\n\nimport \"C\"\n\nfunc main() {}\n", want: true},
		{name: "import group", src: "package main\n\nimport (\n\t\"fmt\"\n\t\"C\"\n)\n\nfunc main() { fmt.Println() }\n", want: true},
		{name: "import group on one line", src: "package main\n\nimport (\"C\")\n\nfunc main() {}\n", want: true},
		{name: "string continuation line", src: "package main\n\nvar s = \"x\" +\n\t\"C\"\n\nfunc main() {}\n"},
		{name: "raw string line", src: "package main\n\nvar s = `\n\"C\"\n`\n\nfunc main() {}\n"},
		{name: "commented import", src: "package main\n\n// import \"C\"\nfunc main() {}\n"},
		{name: "no cgo", src: "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println() }\n"},
		{name: "unparsable import", src: "package main\n\nimport \"C\n", wantErr: true},
	} {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(tc.src), 0o644); err != nil {
			t.Fatal(err)
		}
		got, err := cgoProgram(dir)
		if (err != nil) != tc.wantErr || got != tc.want {
			t.Errorf("%s: cgoProgram = %v, %v; want %v, error %v", tc.name, got, err, tc.want, tc.wantErr)
		}
	}
}

func pkgArgs(progs []string) []string {
	out := make([]string, 0, len(progs))
	for _, p := range progs {
		out = append(out, "./"+p)
	}
	return out
}

type upKey struct {
	level string
	noopt bool
}

func (c config) levelKey() string {
	for _, a := range c.args {
		if level, ok := strings.CutPrefix(a, "-O="); ok {
			return level
		}
	}
	if c.logos {
		return "2"
	}
	return "1"
}

func TestLevelKeySeparatesOptimizationSettings(t *testing.T) {
	effective := map[string]string{
		"K1": "1", "K3": "3", "K4": "3", "K5": "2", "K6": "0", "K7": "3", "K8": "3", "K10": "2",
		"K2": "0", "K11": "2", "K12": "s", "K13": "1", "K14": "s", "K15": "2", "K16": "2",
		"K17": "1", "K18": "1", "K19": "1", "K20": "1", "K21": "1",
	}
	if len(effective) != len(forkConfigs) {
		t.Fatalf("expectation table has %d rows for %d configurations", len(effective), len(forkConfigs))
	}
	for i, a := range forkConfigs {
		for _, b := range forkConfigs[i+1:] {
			sameKey := a.levelKey() == b.levelKey()
			sameOpt := effective[a.id] == effective[b.id]
			if sameKey != sameOpt {
				t.Errorf("%s (-O %s) and %s (-O %s): level key equal = %v, want %v", a.id, effective[a.id], b.id, effective[b.id], sameKey, sameOpt)
			}
		}
	}
}

func levelPrefixes(prog string) []string {
	data, err := os.ReadFile(filepath.Join("testdata", "run", prog, "level-dependent.txt"))
	if err != nil {
		return nil
	}
	var out []string
	for ln := range strings.SplitSeq(string(data), "\n") {
		if t := strings.TrimSpace(ln); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func levelLine(ln, prefix string) bool {
	return ln == prefix || strings.HasPrefix(ln, prefix+" ")
}

func splitLevel(stdout string, prefixes []string) (global, level string) {
	var g, l []string
	for ln := range strings.SplitSeq(stdout, "\n") {
		if slices.ContainsFunc(prefixes, func(p string) bool { return levelLine(ln, p) }) {
			l = append(l, ln)
		} else {
			g = append(g, ln)
		}
	}
	return strings.Join(g, "\n"), strings.Join(l, "\n")
}

func unmatchedPrefixes(stdout string, prefixes []string) []string {
	lines := strings.Split(stdout, "\n")
	var out []string
	for _, p := range prefixes {
		if !slices.ContainsFunc(lines, func(ln string) bool { return levelLine(ln, p) }) {
			out = append(out, p)
		}
	}
	return out
}

func TestUnmatchedLevelPrefixes(t *testing.T) {
	out := "ok\nraw64 a raw=1\nraw32 b raw=2\n"
	if got := unmatchedPrefixes(out, []string{"raw64", "raw32"}); len(got) != 0 {
		t.Errorf("matching prefixes reported as unmatched: %v", got)
	}
	if got := unmatchedPrefixes(out, []string{"raw64", "raw16", "ra"}); !slices.Equal(got, []string{"raw16", "ra"}) {
		t.Errorf("unmatched prefixes = %v, want [raw16 ra]", got)
	}
}

var sweepHexMask = regexp.MustCompile(`0[xX][0-9a-fA-F]{6,}`)

var addrMask = regexp.MustCompile(`\b(?:pc|addr|sp|fp|lr)=0[xX][0-9a-fA-F]+|0xc[0-9a-f]{9,}`)

func cutTraceback(s string) string {
	lines := strings.Split(s, "\n")
	for i, ln := range lines {
		if strings.HasPrefix(ln, "goroutine ") {
			return strings.Join(lines[:i], "\n")
		}
	}
	return s
}

func runBinary(path, maxProcs string, args []string) (observation, error) {
	return runBinaryTimeout(path, maxProcs, runBudget(args, 10*time.Second), corpusNorm)
}

func runBudget(args []string, plain time.Duration) time.Duration {
	if slices.Contains(args, "-race") {
		return 5 * plain
	}
	return plain
}

func runSweepBinary(path string, args []string) (observation, error) {
	return runBinaryTimeout(path, "4", runBudget(args, time.Minute), sweepNorm)
}

func TestRunBudget(t *testing.T) {
	for _, tc := range []struct {
		args  []string
		plain time.Duration
		want  time.Duration
	}{
		{nil, 10 * time.Second, 10 * time.Second},
		{[]string{"-gcflags=all=-N -l"}, 10 * time.Second, 10 * time.Second},
		{[]string{"-race"}, 10 * time.Second, 50 * time.Second},
		{nil, time.Minute, time.Minute},
		{[]string{"-gcflags=all=-N -l"}, time.Minute, time.Minute},
		{[]string{"-race"}, time.Minute, 5 * time.Minute},
	} {
		if got := runBudget(tc.args, tc.plain); got != tc.want {
			t.Errorf("runBudget(%q, %s) = %s, want %s", tc.args, tc.plain, got, tc.want)
		}
	}
}

type normalizer func(stdout, stderr string) (string, string)

func corpusNorm(stdout, stderr string) (string, string) {
	return stdout, addrMask.ReplaceAllString(cutTraceback(stderr), "0xADDR")
}

func sweepNorm(stdout, stderr string) (string, string) {
	return sweepHexMask.ReplaceAllString(stdout, "0xADDR"), sweepHexMask.ReplaceAllString(cutTraceback(stderr), "0xADDR")
}

func TestCorpusNormKeepsProgramValues(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		outA, outB, errA, errB string
		same                   bool
	}{
		{name: "stdout six digit hex", outA: "sum 0x123456\n", outB: "sum 0x654321\n"},
		{name: "stdout ten digit hex", outA: "sum 0x1234567890\n", outB: "sum 0x1234567891\n"},
		{name: "stdout decimal", outA: "sum 123456\n", outB: "sum 654321\n"},
		{name: "stderr six digit hex value", errA: "value 0x123456\n", errB: "value 0x654321\n"},
		{name: "stderr heap pointer", errA: "panic: (*main.T) 0xc000123456\n", errB: "panic: (*main.T) 0xc0009a8b40\n", same: true},
		{name: "stderr fault pc", errA: "[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x47f4a8]\n", errB: "[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x48aa10]\n", same: true},
		{name: "stderr fault code", errA: "[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x47f4a8]\n", errB: "[signal SIGSEGV: segmentation violation code=0x2 addr=0x0 pc=0x47f4a8]\n"},
		{name: "stderr after the traceback", errA: "panic: x\ngoroutine 1 [running]:\n\t0x400000\n", errB: "panic: x\ngoroutine 1 [running]:\n\t0x500000\n", same: true},
	} {
		outA, errA := corpusNorm(tc.outA, tc.errA)
		outB, errB := corpusNorm(tc.outB, tc.errB)
		if got := outA == outB && errA == errB; got != tc.same {
			t.Errorf("%s: normalized equal = %v, want %v", tc.name, got, tc.same)
		}
	}
}

// binaryRunSlots bounds active execution-helper calls across all rows. Its
// capacity snapshots GOMAXPROCS at package initialization.
var binaryRunSlots = make(chan struct{}, runtime.GOMAXPROCS(0))

func runBinaryTimeout(path, maxProcs string, budget time.Duration, norm normalizer) (observation, error) {
	// Queue time does not consume the child execution budget.
	binaryRunSlots <- struct{}{}
	defer func() { <-binaryRunSlots }()

	ctx, cancel := context.WithTimeoutCause(context.Background(), budget, errors.New("goparity run timeout"))
	defer cancel()
	cmd := exec.CommandContext(ctx, path)
	cmd.Env = []string{
		"PATH=/usr/bin:/bin",
		"HOME=" + os.TempDir(),
		"GOMAXPROCS=" + maxProcs,
		"GODEBUG=",
		"GOTRACEBACK=single",
	}
	cmd.WaitDelay = pipeWaitDelay
	setNewProcessGroup(cmd)
	cmd.Cancel = func() error {
		killProcessGroup(cmd)
		return nil
	}
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	runErr := cmd.Run()
	obs := observation{}
	obs.stdout, obs.stderr = norm(out.String(), errb.String())
	if runErr == nil {
		obs.exit = 0
		return obs, nil
	}
	if errors.Is(runErr, exec.ErrWaitDelay) && cmd.ProcessState != nil && cmd.ProcessState.Exited() {
		obs.exit = cmd.ProcessState.ExitCode()
		return obs, nil
	}
	if ctx.Err() != nil {
		if cmd.ProcessState == nil || !cmd.ProcessState.Exited() {
			killProcessGroup(cmd)
		}
		return obs, fmt.Errorf("running %s: %w", path, context.Cause(ctx))
	}
	if ee, ok := errors.AsType[*exec.ExitError](runErr); ok {
		obs.exit = ee.ExitCode()
		return obs, nil
	}
	return obs, fmt.Errorf("running %s: %w", path, runErr)
}

func TestRunBinaryTimeoutKillsSleeper(t *testing.T) {
	if !processGroupKillSupported() {
		skipUnsupportedProcessGroupKill(t)
	}
	fork := forkTool(t)
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "pid")
	src := fmt.Sprintf(`package main

import (
	"fmt"
	"os"
	"time"
)

func main() {
	_ = os.WriteFile(%q, []byte(fmt.Sprint(os.Getpid())), 0o644)
	time.Sleep(8 * time.Second)
}
`, pidFile)
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module sleeper\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "sleeper")
	if out, err := goBuild(fork.bin, dir, []string{"build", "-o", bin, "."}, baseEnv(forkCache(), fork.goroot), 5*time.Minute); err != nil {
		t.Fatalf("building sleeper: %v\n%s", err, clip(out))
	}
	start := time.Now()
	errCh := make(chan error, 1)
	go func() {
		_, err := runBinaryTimeout(bin, "4", 2*time.Second, corpusNorm)
		errCh <- err
	}()
	var pid int
	for {
		if b, rerr := os.ReadFile(pidFile); rerr == nil {
			if p, perr := strconv.Atoi(strings.TrimSpace(string(b))); perr == nil {
				pid = p
				break
			}
		}
		if time.Now().After(start.Add(30 * time.Second)) {
			t.Fatalf("sleeper wrote no readable pid file within %s", time.Since(start).Round(time.Second))
		}
		time.Sleep(20 * time.Millisecond)
	}
	select {
	case err := <-errCh:
		if err == nil || !strings.Contains(err.Error(), "goparity run timeout") {
			t.Fatalf("sleeper run = %v, want the goparity run timeout error", err)
		}
	case <-time.After(time.Until(start.Add(6 * time.Second))):
		t.Fatal("sleeper run did not return within 6s of start: the kill did not beat the 8s sleep")
	}
	if !procGroupGone(pid, 4*time.Second) {
		t.Fatalf("sleeper process group %d still alive after the run returned: child survived the kill", pid)
	}
}

func TestRunBinaryTimeoutReleasesGrandchildPipe(t *testing.T) {
	if !processGroupKillSupported() {
		skipUnsupportedProcessGroupKill(t)
	}
	fork := forkTool(t)
	dir := t.TempDir()
	heldFile := filepath.Join(dir, "held.marker")
	heldDir := filepath.Join(dir, "heldmod")
	heldSrc := fmt.Sprintf(`package main

import (
	"os"
	"time"
)

func main() {
	time.Sleep(25 * time.Second)
	_ = os.WriteFile(%q, []byte("held"), 0o644)
}
`, heldFile)
	if err := os.MkdirAll(heldDir, 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(heldDir, "go.mod"), []byte("module held\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(heldDir, "main.go"), []byte(heldSrc), 0o644); err != nil {
		t.Fatal(err)
	}
	heldBin := filepath.Join(heldDir, "held")
	if out, err := goBuild(fork.bin, heldDir, []string{"build", "-o", heldBin, "."}, baseEnv(forkCache(), fork.goroot), 5*time.Minute); err != nil {
		t.Fatalf("building pipe holder: %v\n%s", err, clip(out))
	}
	src := fmt.Sprintf(`package main

import (
	"fmt"
	"os"
	"os/exec"
)

func main() {
	fmt.Println("done")
	c := exec.Command(%q)
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	_ = c.Start()
}
`, heldBin)
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module daemon\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "daemon")
	if out, err := goBuild(fork.bin, dir, []string{"build", "-o", bin, "."}, baseEnv(forkCache(), fork.goroot), 5*time.Minute); err != nil {
		t.Fatalf("building daemon: %v\n%s", err, clip(out))
	}
	obs, err := runBinaryTimeout(bin, "4", 15*time.Second, corpusNorm)
	if err != nil {
		t.Fatalf("daemon run = %v, want the exited parent observation", err)
	}
	if obs.exit != 0 {
		t.Fatalf("daemon exit = %d, want 0", obs.exit)
	}
	if !strings.Contains(obs.stdout, "done") {
		t.Fatalf("daemon stdout = %q, want the parent output drained before return", obs.stdout)
	}
	if _, serr := os.Stat(heldFile); serr == nil {
		t.Fatalf("pipe holder wrote its marker before the run returned: the run waited for the pipe-holding grandchild")
	}
}

func TestRunGeneratorTimeoutKillsSleeper(t *testing.T) {
	if !processGroupKillSupported() {
		skipUnsupportedProcessGroupKill(t)
	}
	fork := forkTool(t)
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "pgid")
	src := fmt.Sprintf(`package main

import (
	"fmt"
	"os"
	"syscall"
	"time"
)

func main() {
	if os.Getenv("GOPARITY_SLEEPGEN_WARM") != "" {
		return
	}
	pgid, _ := syscall.Getpgid(0)
	_ = os.WriteFile(%q, []byte(fmt.Sprint(pgid)), 0o644)
	time.Sleep(10 * time.Second)
}
`, pidFile)
	if err := os.WriteFile(filepath.Join(dir, "gen.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	env := baseEnv(forkCache(), fork.goroot)
	if out, err := runGenerator(fork.bin, "sleepgen", dir, appendEnv(env, "GOPARITY_SLEEPGEN_WARM=1"), 5*time.Minute); err != nil {
		t.Fatalf("building the sleeping generator: %v\n%s", err, clip(out))
	}
	start := time.Now()
	type runRes struct {
		out string
		err error
	}
	ch := make(chan runRes, 1)
	go func() {
		out, err := runGenerator(fork.bin, "sleepgen", dir, env, 2*time.Second)
		ch <- runRes{out, err}
	}()
	var pgid int
	for {
		if b, rerr := os.ReadFile(pidFile); rerr == nil {
			if p, perr := strconv.Atoi(strings.TrimSpace(string(b))); perr == nil {
				pgid = p
				break
			}
		}
		if time.Now().After(start.Add(30 * time.Second)) {
			t.Fatalf("sleeping generator wrote no readable pgid file within %s", time.Since(start).Round(time.Second))
		}
		time.Sleep(20 * time.Millisecond)
	}
	select {
	case res := <-ch:
		if res.err == nil || !strings.Contains(res.err.Error(), "timed out") {
			t.Fatalf("sleeping generator = %v, want a timeout error", res.err)
		}
	case <-time.After(time.Until(start.Add(7 * time.Second))):
		t.Fatal("sleeping generator did not return within 7s of start: the kill did not beat the 10s sleep")
	}
	if !procGroupGone(pgid, 4*time.Second) {
		t.Fatalf("generator process group %d still alive after the generator returned: child survived the kill", pgid)
	}
}

func obsEqual(a, b observation) bool {
	return a.exit == b.exit && a.stdout == b.stdout && a.stderr == b.stderr
}

var lawWord = regexp.MustCompile(`\bLAW\b`)

func corpusDefect(o observation) bool {
	return o.exit != 0 || lawWord.MatchString(o.stdout) || lawWord.MatchString(o.stderr)
}

func TestCorpusDefectProbes(t *testing.T) {
	for _, tc := range []struct {
		name string
		o    observation
		want bool
	}{
		{"clean", observation{stdout: "ok\n"}, false},
		{"nonzero exit", observation{exit: 1}, true},
		{"law on stdout", observation{stdout: "LAW x\n"}, true},
		{"law mid line", observation{stdout: "case 1 LAW clobbered\n"}, true},
		{"law on stderr", observation{stderr: "LAW x\n"}, true},
		{"word containing law", observation{stdout: "OUTLAW\n"}, false},
		{"word containing law on stderr", observation{stderr: "claw LAWN\n"}, false},
	} {
		if got := corpusDefect(tc.o); got != tc.want {
			t.Errorf("%s: corpusDefect = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func firstDiff(a, b string) string {
	al := strings.Split(a, "\n")
	bl := strings.Split(b, "\n")
	for i := range max(len(al), len(bl)) {
		var x, y string
		if i < len(al) {
			x = al[i]
		}
		if i < len(bl) {
			y = bl[i]
		}
		if x != y {
			return fmt.Sprintf("line %d: got %q want %q", i+1, clip(x), clip(y))
		}
	}
	return ""
}

func firstDivergence(a, b observation) string {
	return cmp.Or(firstDiff(a.stdout, b.stdout), firstDiff(a.stderr, b.stderr), fmt.Sprintf("exit %d vs %d", a.exit, b.exit))
}

func stabilityDiff(ref, got observation) string {
	if obsEqual(ref, got) {
		return ""
	}
	return firstDivergence(ref, got)
}

func TestStabilityDiffSeesLevelLines(t *testing.T) {
	ref := observation{stdout: "ok\nraw64 a raw=1\n"}
	for _, tc := range []struct {
		name string
		got  observation
		same bool
	}{
		{"identical", ref, true},
		{"level line differs", observation{stdout: "ok\nraw64 a raw=2\n"}, false},
		{"compared line differs", observation{stdout: "no\nraw64 a raw=1\n"}, false},
		{"stderr differs", observation{stdout: ref.stdout, stderr: "x\n"}, false},
		{"exit differs", observation{stdout: ref.stdout, exit: 1}, false},
	} {
		if got := stabilityDiff(ref, tc.got) == ""; got != tc.same {
			t.Errorf("%s: stable = %v, want %v", tc.name, got, tc.same)
		}
	}
}

func clip(s string) string {
	if len(s) > 120 {
		return s[:117] + "..."
	}
	return s
}

func obsDesc(o observation) string {
	return fmt.Sprintf("exit %d stdout %d bytes stderr %d bytes", o.exit, len(o.stdout), len(o.stderr))
}

func runAll(r buildResult, maxProcs string, args []string) (map[string]observation, error) {
	out := make(map[string]observation, len(r.progs))
	var firstErr error
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, p := range r.progs {
		wg.Go(func() {
			o, err := runBinary(filepath.Join(r.outDir, p), maxProcs, args)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = err
				}
				return
			}
			out[p] = o
		})
	}
	wg.Wait()
	return out, firstErr
}

func activeUpConfigs(ks []config) []upConfig {
	need := make(map[string]bool)
	for _, c := range ks {
		need[c.upstream] = true
	}
	if !testing.Short() {
		need["U1N"] = true
	}
	var out []upConfig
	for _, u := range upConfigs {
		if need[u.id] {
			out = append(out, u)
		}
	}
	return out
}

func buildAllFork(fork toolchain, base string, progs []string, ks []config) map[string]buildResult {
	results := make(map[string]buildResult, len(ks))
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, c := range ks {
		wg.Go(func() {
			r := buildFork(fork, base, progs, c)
			mu.Lock()
			results[c.id] = r
			mu.Unlock()
		})
	}
	wg.Wait()
	return results
}

func buildFork(fork toolchain, base string, progs []string, c config) buildResult {
	stage := filepath.Join(base, "fork-"+c.id)
	if err := stageCorpus(fork, stage, forkCache(), progs); err != nil {
		return buildResult{err: fmt.Errorf("fork stage: %w", err)}
	}
	if c.toml != "" {
		if err := os.WriteFile(filepath.Join(stage, "logos.toml"), []byte(c.toml), 0o644); err != nil {
			return buildResult{err: fmt.Errorf("writing logos.toml: %w", err)}
		}
	}
	env := appendEnv(baseEnv(forkCache(), fork.goroot), c.env...)
	bin := fork.bin
	if c.logos {
		bin = fork.logos
	}
	outDir := filepath.Join(base, "out-fork-"+c.id)
	wantCgo := *releaseFlag || c.id == "K1" || c.id == "K5"
	sb := stagedBuild{bin: bin, stage: stage, outDir: outDir, args: c.args, env: env, progs: progs, wantCgo: wantCgo}
	built, out, err := sb.build()
	return buildResult{outDir: outDir, output: out, err: err, progs: built}
}

func buildAllUpstream(up toolchain, base string, progs []string, ups []upConfig) map[string]buildResult {
	results := make(map[string]buildResult, len(ups))
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, u := range ups {
		wg.Go(func() {
			r := buildUpstream(up, base, progs, u)
			mu.Lock()
			results[u.id] = r
			mu.Unlock()
		})
	}
	wg.Wait()
	return results
}

func buildUpstream(up toolchain, base string, progs []string, u upConfig) buildResult {
	stage := filepath.Join(base, "up-"+u.id)
	if err := stageCorpus(up, stage, upCache(), progs); err != nil {
		return buildResult{err: fmt.Errorf("upstream stage: %w", err)}
	}
	env := baseEnv(upCache(), up.goroot)
	if u.race {
		env = appendEnv(env, "CGO_ENABLED=1")
	}
	env = appendEnv(env, u.env...)
	outDir := filepath.Join(base, "out-up-"+u.id)
	wantCgo := *releaseFlag || u.id == "U1" || u.id == "U3"
	sb := stagedBuild{bin: up.bin, stage: stage, outDir: outDir, args: u.args, env: env, progs: progs, wantCgo: wantCgo}
	built, out, err := sb.build()
	return buildResult{outDir: outDir, output: out, err: err, progs: built}
}

func TestStageFailureIsABuildError(t *testing.T) {
	progs := []string{"no-such-program"}
	forkBuilds := buildAllFork(toolchain{}, t.TempDir(), progs, []config{{id: "K1"}})
	if r, ok := forkBuilds["K1"]; !ok || r.err == nil {
		t.Errorf("a fork stage failure left the result %+v, want a build error", r)
	}
	upBuilds := buildAllUpstream(toolchain{}, t.TempDir(), progs, []upConfig{{id: "U1"}})
	if r, ok := upBuilds["U1"]; !ok || r.err == nil {
		t.Errorf("an upstream stage failure left the result %+v, want a build error", r)
	}
}

var (
	mainCompileLine = regexp.MustCompile(`(?m)^.* -p main .*$`)
	modinfoLine     = regexp.MustCompile(`(?m)^modinfo .*$`)
	compileAmd64    = regexp.MustCompile(`GOAMD64='(v\d)'`)
	compileBudget   = regexp.MustCompile(`inlbudget=(\d+)`)
	buildInfoExp    = regexp.MustCompile(`GOEXPERIMENT=([^\\"\s]*)`)
)

func (c config) experiment() string {
	for _, kv := range c.env {
		if v, ok := strings.CutPrefix(kv, "GOEXPERIMENT="); ok {
			return v
		}
	}
	if c.logos && !strings.Contains(c.toml, "arena = false") {
		return "logosarena"
	}
	return ""
}

func dryBuild(t *testing.T, fork toolchain, c config) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"go.mod":  "module example.com/dry\n\ngo 1.27\n",
		"main.go": "package main\n\nfunc main() {}\n",
	}
	if c.toml != "" {
		files["logos.toml"] = c.toml
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	bin := fork.bin
	if c.logos {
		bin = fork.logos
	}
	args := append([]string{"build", "-n", "-o", filepath.Join(dir, "dry")}, c.args...)
	out, err := goBuild(bin, dir, append(args, "."), appendEnv(baseEnv(forkCache(), fork.goroot), c.env...), 5*time.Minute)
	if err != nil {
		t.Fatalf("dry build under %s: %v\n%s", c.id, err, clip(out))
	}
	return out
}

func compileProblems(c config, out string) []string {
	line := mainCompileLine.FindString(out)
	if line == "" {
		return []string{"no compile command for package main"}
	}
	var problems []string
	budgets := map[string]string{"0": "", "1": "", "2": "120", "3": "400", "s": "15"}
	var budget string
	if m := compileBudget.FindStringSubmatch(line); m != nil {
		budget = m[1]
	}
	if want := budgets[c.levelKey()]; budget != want {
		problems = append(problems, fmt.Sprintf("-O %s: compile inlbudget %q, want %q", c.levelKey(), budget, want))
	}
	if got, want := strings.Contains(line, " -N "), c.levelKey() == "0"; got != want {
		problems = append(problems, fmt.Sprintf("-O %s: compile -N = %v, want %v", c.levelKey(), got, want))
	}
	level := "v1"
	if m := compileAmd64.FindStringSubmatch(line); m != nil {
		level = m[1]
	}
	if level != c.group {
		problems = append(problems, fmt.Sprintf("compile GOAMD64 %s, want the %s of the matched upstream row", level, c.group))
	}
	if got, want := strings.Contains(line, " -race "), slices.Contains(c.args, "-race"); got != want {
		problems = append(problems, fmt.Sprintf("compile -race = %v, want %v", got, want))
	}
	var experiment string
	if m := buildInfoExp.FindStringSubmatch(modinfoLine.FindString(out)); m != nil {
		experiment = m[1]
	}
	if experiment != c.experiment() {
		problems = append(problems, fmt.Sprintf("build info GOEXPERIMENT %q, want %q", experiment, c.experiment()))
	}
	return problems
}

func TestForkConfigsReachTheCompiler(t *testing.T) {
	fork := forkTool(t)
	for _, c := range activeForkConfigs() {
		t.Run(c.id, func(t *testing.T) {
			for _, p := range compileProblems(c, dryBuild(t, fork, c)) {
				t.Error(p)
			}
		})
	}
}

func TestCompileProblemsSeeAnIgnoredFlag(t *testing.T) {
	byID := make(map[string]config, len(forkConfigs))
	for _, c := range forkConfigs {
		byID[c.id] = c
	}
	const base = "/x/compile -o $WORK/b001/_pkg_.a -p main -lang=go1.27 -c=80"
	for _, tc := range []struct {
		name string
		cfg  string
		out  string
		want int
	}{
		{"O3 honoured", "K3", base + " -d=inlbudget=400\n", 0},
		{"O3 ignored", "K3", base + "\n", 1},
		{"O0 without -N", "K2", base + "\n", 1},
		{"O0 with -N", "K2", base + " -N -l\n", 0},
		{"v3 row built at v1", "K4", base + " -d=inlbudget=400\n", 1},
		{"v3 row built at v3", "K4", "GOAMD64='v3' " + base + " -d=inlbudget=400\n", 0},
		{"arena off but build info has the arena", "K8", "GOAMD64='v3' " + base + " -d=inlbudget=400\nmodinfo \"build\\tGOEXPERIMENT=logosarena\\n\"\n", 1},
		{"race ignored", "K16", "GOAMD64='v3' " + base + " -d=inlbudget=120\nmodinfo \"build\\tGOEXPERIMENT=logosarena\\n\"\n", 1},
		{"no compile line", "K1", "nothing\n", 1},
	} {
		if got := compileProblems(byID[tc.cfg], tc.out); len(got) != tc.want {
			t.Errorf("%s: %d problems %v, want %d", tc.name, len(got), got, tc.want)
		}
	}
}

func TestParity(t *testing.T) {
	start := time.Now()
	fork := forkTool(t)
	up, upOn := upstreamTool(t)
	progs := corpusPrograms(t)
	if len(progs) == 0 {
		t.Fatal("no corpus programs under testdata/run")
	}
	ks := activeForkConfigs()
	base := t.TempDir()

	forkBuilds := buildAllFork(fork, base, progs, ks)
	for _, c := range ks {
		if r := forkBuilds[c.id]; r.err != nil {
			t.Errorf("fork build %s failed: %v\n%s", c.id, r.err, clip(r.output))
		}
	}
	var ups []upConfig
	var upBuilds map[string]buildResult
	if upOn {
		ups = activeUpConfigs(ks)
		upBuilds = buildAllUpstream(up, base, progs, ups)
		for _, u := range ups {
			if r := upBuilds[u.id]; r.err != nil {
				t.Errorf("upstream build %s failed: %v\n%s", u.id, r.err, clip(r.output))
			}
		}
	}
	forkObs := map[string]map[string]observation{}
	upObs := map[string]map[string]observation{}
	forkRunErrs := map[string]error{}
	upRunErrs := map[string]error{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, c := range ks {
		if forkBuilds[c.id].err != nil {
			continue
		}
		wg.Go(func() {
			o, err := runAll(forkBuilds[c.id], "4", c.args)
			mu.Lock()
			if err != nil {
				forkRunErrs[c.id] = err
			} else {
				forkObs[c.id] = o
			}
			mu.Unlock()
		})
	}
	if upOn {
		for _, u := range ups {
			if upBuilds[u.id].err != nil {
				continue
			}
			wg.Go(func() {
				o, err := runAll(upBuilds[u.id], "4", u.args)
				mu.Lock()
				if err != nil {
					upRunErrs[u.id] = err
				} else {
					upObs[u.id] = o
				}
				mu.Unlock()
			})
		}
	}
	wg.Wait()
	for id, err := range forkRunErrs {
		t.Errorf("fork runs %s failed: %v", id, err)
	}
	if upOn {
		for id, err := range upRunErrs {
			t.Errorf("upstream runs %s failed: %v", id, err)
		}
	}
	for _, c := range ks {
		for _, p := range progs {
			if o, ok := forkObs[c.id][p]; ok && corpusDefect(o) {
				t.Errorf("corpus defect: program %s failed under fork %s: %s\n%s", p, c.id, obsDesc(o), clip(o.stdout))
			}
		}
	}
	if upOn {
		for _, u := range ups {
			for _, p := range progs {
				if o, ok := upObs[u.id][p]; ok && corpusDefect(o) {
					t.Errorf("corpus defect: program %s failed under upstream %s: %s\n%s", p, u.id, obsDesc(o), clip(o.stdout))
				}
			}
		}
	}
	stability := map[string]observation{}
	if upOn && upBuilds["U1"].err == nil {
		stab, stabErr := runAll(upBuilds["U1"], "1", nil)
		if stabErr != nil {
			t.Errorf("upstream stability runs failed: %v", stabErr)
		} else {
			stability = stab
		}
	}
	if !upOn && forkBuilds["K1"].err == nil {
		stab, stabErr := runAll(forkBuilds["K1"], "1", nil)
		if stabErr != nil {
			t.Errorf("fork stability runs failed: %v", stabErr)
		} else {
			stability = stab
		}
	}
	upByID := make(map[string]upConfig, len(ups))
	for _, u := range ups {
		upByID[u.id] = u
	}
	stabilityRefs, stabilityLabel := forkObs["K1"], "fork K1"
	if upOn {
		stabilityRefs, stabilityLabel = upObs["U1"], "upstream U1"
	}
	for _, p := range progs {
		prefs := levelPrefixes(p)
		if o, ok := stabilityRefs[p]; ok && len(prefs) > 0 {
			if gs, _ := splitLevel(o.stdout, prefs); strings.TrimSpace(gs) == "" {
				t.Errorf("program %s: level-dependent prefixes mark every line, no compared lines", p)
			}
			for _, unmatched := range unmatchedPrefixes(o.stdout, prefs) {
				t.Errorf("program %s: level-dependent prefix %q matches no output line", p, unmatched)
			}
		}
		var gref *observation
		lrefs := map[upKey]string{}
		for _, u := range ups {
			o, ok := upObs[u.id][p]
			if !ok {
				continue
			}
			gs, lv := splitLevel(o.stdout, prefs)
			og := observation{exit: o.exit, stdout: gs, stderr: o.stderr}
			if gref == nil {
				cp := og
				gref = &cp
			} else if !obsEqual(*gref, og) {
				t.Errorf("invariance gate: program %s differs between upstream rows: %s vs %s (%s)", p, obsDesc(*gref), obsDesc(og), firstDiff(gref.stdout, og.stdout))
			}
			k := upKey{level: upByID[u.id].level, noopt: upByID[u.id].noopt}
			if prev, ok := lrefs[k]; ok && prev != lv {
				t.Errorf("invariance gate: program %s level-dependent lines differ within upstream %s,noopt=%v rows: %s", p, k.level, k.noopt, firstDiff(prev, lv))
			} else if !ok {
				lrefs[k] = lv
			}
		}
		if o, ok := stability[p]; ok {
			if ref, ok := stabilityRefs[p]; ok {
				if d := stabilityDiff(ref, o); d != "" {
					t.Errorf("stability gate: program %s differs under %s at GOMAXPROCS=1 vs 4: %s", p, stabilityLabel, d)
				}
			}
		}
	}
	for _, c := range ks {
		if forkBuilds[c.id].err != nil {
			continue
		}
		uo, ok := upObs[c.upstream]
		if !ok {
			continue
		}
		for _, p := range progs {
			fo, ok1 := forkObs[c.id][p]
			uo1, ok2 := uo[p]
			if !ok1 || !ok2 {
				continue
			}
			if !obsEqual(fo, uo1) {
				d := firstDivergence(fo, uo1)
				t.Errorf("parity: program %s: %s vs %s: %s (%s vs %s)", p, c.id, c.upstream, d, obsDesc(fo), obsDesc(uo1))
			}
		}
	}
	for _, group := range []string{"v1", "v3"} {
		var members []config
		for _, c := range ks {
			if c.group == group && forkBuilds[c.id].err == nil {
				members = append(members, c)
			}
		}
		for _, p := range progs {
			prefs := levelPrefixes(p)
			var gref *observation
			lrefs := map[string]string{}
			for _, m := range members {
				o, ok := forkObs[m.id][p]
				if !ok {
					continue
				}
				gs, lv := splitLevel(o.stdout, prefs)
				og := observation{exit: o.exit, stdout: gs, stderr: o.stderr}
				if gref == nil {
					cp := og
					gref = &cp
				} else if !obsEqual(*gref, og) {
					t.Errorf("metamorphic: program %s differs between %s and %s (%s group): %s", p, members[0].id, m.id, group, firstDiff(gref.stdout, og.stdout))
				}
				if prev, ok := lrefs[m.levelKey()]; ok && prev != lv {
					t.Errorf("metamorphic: program %s level-dependent lines differ between %s and %s (%s group, opt %s): %s", p, members[0].id, m.id, group, m.levelKey(), firstDiff(prev, lv))
				} else if !ok {
					lrefs[m.levelKey()] = lv
				}
			}
		}
	}
	t.Logf("parity: %d programs, %d fork configs, %d upstream rows, wall %s", len(progs), len(ks), len(ups), time.Since(start))
}

var failPosPattern = regexp.MustCompile(`(?m)(?:^|vet: )(\S+\.go):(\d+):(\d+): `)

func failPositions(s string) []string {
	var out []string
	for _, m := range failPosPattern.FindAllStringSubmatch(s, -1) {
		out = append(out, m[2]+":"+m[3])
	}
	slices.Sort(out)
	return out
}

func stageFailCorpus(dir, toml string, stems []string) error {
	if err := os.MkdirAll(dir, 0o777); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module example.com/parity\n\ngo 1.27\n"), 0o644); err != nil {
		return err
	}
	if toml != "" {
		if err := os.WriteFile(filepath.Join(dir, "logos.toml"), []byte(toml), 0o644); err != nil {
			return err
		}
	}
	for _, s := range stems {
		data, err := os.ReadFile(filepath.Join("testdata", "fail", s+".go"))
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Join(dir, s), 0o777); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, s, "main.go"), data, 0o644); err != nil {
			return err
		}
	}
	return nil
}

type failRun struct {
	fork toolchain
	base string
}

type failRef struct {
	stem   string
	pos    []string
	vetPos []string
}

func checkFailCorpus(t *testing.T, fork, up toolchain, base string, ks []config) []string {
	t.Helper()
	stems := corpusFails(t)
	if len(stems) == 0 {
		t.Fatal("no fail corpus files under testdata/fail")
	}
	stage := filepath.Join(base, "failmod")
	if err := stageFailCorpus(stage, "", stems); err != nil {
		t.Fatalf("staging the fail corpus: %v", err)
	}
	stages := make(map[string]string, len(ks))
	var ran []string
	for _, c := range ks {
		stages[c.id] = filepath.Join(base, "failmod-"+c.id)
		if err := stageFailCorpus(stages[c.id], c.toml, stems); err != nil {
			t.Fatalf("staging the fail corpus for %s: %v", c.id, err)
		}
		ran = append(ran, c.id)
	}
	run := failRun{fork: fork, base: base}
	for _, s := range stems {
		upOut, upErr := goBuild(up.bin, stage, []string{"build", "-o", filepath.Join(base, "failbin-"+s), "./" + s}, baseEnv(upCache(), up.goroot), 5*time.Minute)
		if upErr == nil {
			t.Errorf("fail corpus %s: upstream build succeeded, corpus defect", s)
			continue
		}
		if _, err := os.Stat(filepath.Join(base, "failbin-"+s)); err == nil {
			t.Errorf("fail corpus %s: upstream wrote a binary for a rejected program", s)
		}
		ref := failRef{stem: s, pos: failPositions(upOut)}
		if len(ref.pos) == 0 {
			t.Errorf("fail corpus %s: upstream reported no file:line:col, corpus defect", s)
			continue
		}
		vetOut, vetErr := goBuild(up.bin, stage, []string{"vet", "./" + s}, baseEnv(upCache(), up.goroot), 5*time.Minute)
		if vetErr == nil {
			t.Errorf("fail corpus %s: upstream vet succeeded, corpus defect", s)
			continue
		}
		ref.vetPos = failPositions(vetOut)
		if len(ref.vetPos) == 0 {
			t.Errorf("fail corpus %s: upstream vet reported no file:line:col, corpus defect", s)
			continue
		}
		var wg sync.WaitGroup
		for _, c := range ks {
			wg.Go(func() { checkFailUnder(t, run, c, stages[c.id], ref) })
		}
		wg.Wait()
	}
	return ran
}

func checkFailUnder(t *testing.T, run failRun, c config, stage string, ref failRef) {
	t.Helper()
	bin := run.fork.bin
	if c.logos {
		bin = run.fork.logos
	}
	fenv := appendEnv(baseEnv(forkCache(), run.fork.goroot), c.env...)
	out := filepath.Join(run.base, "failbin-fork-"+c.id+"-"+ref.stem)
	fout, ferr := goBuild(bin, stage, append(append([]string{"build"}, c.args...), "-o", out, "./"+ref.stem), fenv, 5*time.Minute)
	if ferr == nil {
		t.Errorf("fail corpus %s: fork %s accepted invalid Go", ref.stem, c.id)
	}
	if _, err := os.Stat(out); err == nil {
		t.Errorf("fail corpus %s: fork %s wrote a binary for a rejected program", ref.stem, c.id)
	}
	if got := failPositions(fout); len(got) == 0 {
		t.Errorf("fail corpus %s: fork %s reported no file:line:col", ref.stem, c.id)
	} else if !slices.Equal(got, ref.pos) {
		t.Errorf("fail corpus %s: fork %s positions %v differ from upstream %v", ref.stem, c.id, got, ref.pos)
	}
	if c.id != "K1" {
		return
	}
	vout, verr := goBuild(bin, stage, []string{"vet", "./" + ref.stem}, fenv, 5*time.Minute)
	if verr == nil {
		t.Errorf("fail corpus %s: fork vet accepted invalid Go", ref.stem)
		return
	}
	if got := failPositions(vout); len(got) == 0 {
		t.Errorf("fail corpus %s: fork vet reported no file:line:col", ref.stem)
	} else if !slices.Equal(got, ref.vetPos) {
		t.Errorf("fail corpus %s: fork vet positions %v differ from upstream %v", ref.stem, got, ref.vetPos)
	}
}

func TestParityFailCorpus(t *testing.T) {
	fork := forkTool(t)
	up, upOn := upstreamTool(t)
	if !upOn {
		t.Skip("upstream oracle off: the fail corpus compares positions with live upstream output")
	}
	ks := activeForkConfigs()
	ran := checkFailCorpus(t, fork, up, t.TempDir(), ks)
	want := make([]string, 0, len(ks))
	for _, c := range ks {
		want = append(want, c.id)
	}
	if !slices.Equal(ran, want) {
		t.Errorf("the fail corpus ran under %v, want every active configuration %v", ran, want)
	}
}

type mutant struct {
	id     string
	kind   string
	file   string
	expect string
	cfg    string
	old    string
	new    string
}

func loadMutants(t *testing.T) []mutant {
	t.Helper()
	matches, err := filepath.Glob("testdata/mutants/*.txt")
	if err != nil {
		t.Fatalf("globbing testdata/mutants: %v", err)
	}
	slices.Sort(matches)
	var out []mutant
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("reading %s: %v", path, err)
		}
		ms, err := parseMutants(string(data))
		if err != nil {
			t.Fatalf("mutant %s: %v", path, err)
		}
		stem := strings.TrimSuffix(filepath.Base(path), ".txt")
		for i := range ms {
			ms[i].id = stem
			if len(ms) > 1 {
				ms[i].id = fmt.Sprintf("%s-%d", stem, i+1)
			}
			out = append(out, ms[i])
		}
	}
	return out
}

func parseMutants(data string) ([]mutant, error) {
	var chunks [][]string
	var cur []string
	seenNew := false
	for ln := range strings.SplitSeq(data, "\n") {
		if ln == "-- new --" {
			seenNew = true
		}
		if seenNew && isMutantHeader(ln) {
			chunks = append(chunks, cur)
			cur = nil
			seenNew = false
		}
		cur = append(cur, ln)
	}
	chunks = append(chunks, cur)
	out := make([]mutant, 0, len(chunks))
	for _, c := range chunks {
		m, err := parseMutant(strings.Join(c, "\n"))
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, nil
}

func isMutantHeader(ln string) bool {
	if strings.HasPrefix(ln, " ") || strings.HasPrefix(ln, "\t") {
		return false
	}
	k, _, ok := strings.Cut(ln, ":")
	if !ok {
		return false
	}
	switch k {
	case "kind", "file", "expect", "config":
		return true
	}
	return false
}

func parseMutant(data string) (mutant, error) {
	var m mutant
	var oldLines, newLines []string
	section := ""
	for _, ln := range strings.Split(data, "\n") {
		switch ln {
		case "-- old --":
			section = "old"
			continue
		case "-- new --":
			section = "new"
			continue
		}
		switch section {
		case "":
			if strings.TrimSpace(ln) == "" {
				continue
			}
			k, v, ok := strings.Cut(ln, ":")
			if !ok {
				return m, fmt.Errorf("bad header line %q", ln)
			}
			v = strings.TrimSpace(v)
			switch strings.TrimSpace(k) {
			case "kind":
				m.kind = v
			case "file":
				m.file = v
			case "expect":
				m.expect = v
			case "config":
				m.cfg = v
			default:
				return m, fmt.Errorf("unknown header key %q", strings.TrimSpace(k))
			}
		case "old":
			oldLines = append(oldLines, ln)
		case "new":
			newLines = append(newLines, ln)
		}
	}
	m.old = joinSection(oldLines)
	m.new = joinSection(newLines)
	if m.kind == "" || m.file == "" || m.expect == "" || m.cfg == "" {
		return m, errors.New("missing kind, file, expect, or config header")
	}
	if m.old == "" || m.new == "" {
		return m, errors.New("empty old or new edit")
	}
	return m, nil
}

func joinSection(lines []string) string {
	if n := len(lines); n > 0 && lines[n-1] == "" {
		lines = lines[:n-1]
	}
	return strings.Join(lines, "\n")
}

func configIndex() map[string]config {
	out := make(map[string]config, len(forkConfigs))
	for _, c := range forkConfigs {
		out[c.id] = c
	}
	return out
}

var mutantToolNames = map[string]string{"compiler": "compile", "linker": "link", "cgo": "cgo", "rules": "compile"}

type mutBuild struct {
	fork        toolchain
	cfg         config
	stage       string
	scratch     string
	shim        string
	overlayPath string
	env         []string
	args        []string
	outDir      string
	plain       []string
	cgoProg     []string
}

func (mb *mutBuild) driverBin() string {
	bin := mb.fork.bin
	if mb.cfg.logos {
		bin = mb.fork.logos
	}
	return bin
}

func (mb *mutBuild) corpus(t *testing.T, progs []string) {
	t.Helper()
	if err := stageCorpus(mb.fork, mb.stage, forkCache(), progs); err != nil {
		t.Fatalf("staging corpus: %v", err)
	}
	if mb.cfg.toml != "" {
		if err := os.WriteFile(filepath.Join(mb.stage, "logos.toml"), []byte(mb.cfg.toml), 0o644); err != nil {
			t.Fatalf("writing logos.toml: %v", err)
		}
	}
	for _, s := range corpusFails(t) {
		data, err := os.ReadFile(filepath.Join("testdata", "fail", s+".go"))
		if err != nil {
			t.Fatalf("reading fail corpus %s: %v", s, err)
		}
		if err := os.MkdirAll(filepath.Join(mb.stage, s), 0o777); err != nil {
			t.Fatalf("creating fail stage %s: %v", s, err)
		}
		if err := os.WriteFile(filepath.Join(mb.stage, s, "main.go"), data, 0o644); err != nil {
			t.Fatalf("writing fail stage %s: %v", s, err)
		}
	}
	mb.env = appendEnv(baseEnv(forkCache(), mb.fork.goroot), mb.cfg.env...)
	mb.args = slices.Clone(mb.cfg.args)
	mb.outDir = filepath.Join(mb.scratch, "out")
	plain, cgoProg, err := splitStaged(mb.stage, progs)
	if err != nil {
		t.Fatalf("classifying corpus: %v", err)
	}
	mb.plain, mb.cgoProg = plain, cgoProg
}

func (mb *mutBuild) apply(t *testing.T, m mutant) {
	t.Helper()
	switch m.kind {
	case "compiler", "linker", "cgo", "rules":
		toolName := mutantToolNames[m.kind]
		toolBin, err := mb.buildTool(t, toolName)
		if err != nil {
			t.Fatalf("mutant %s: building mutant tool failed: %v", m.id, err)
		}
		cache := mutantCache(m.id)
		os.MkdirAll(cache, 0o777)
		os.MkdirAll(cache+"-path", 0o777)
		mb.env = appendEnv(mb.env, "GOPARITY_MUTANT_TOOL="+toolName+"="+toolBin, "GOCACHE="+cache, "GOPATH="+cache+"-path")
		mb.args = append(mb.args, "-toolexec="+mb.shim)
	case "runtime", "std":
		mb.args = append(mb.args, "-overlay="+mb.overlayPath)
	default:
		t.Fatalf("mutant %s: unsupported kind %q", m.id, m.kind)
	}
}

func (mb *mutBuild) buildTool(t *testing.T, toolName string) (string, error) {
	t.Helper()
	outPath := filepath.Join(mb.scratch, toolName)
	ctx, cancel := context.WithTimeoutCause(context.Background(), 15*time.Minute, errors.New("goparity tool build timeout"))
	defer cancel()
	cmd := exec.CommandContext(ctx, mb.fork.bin, "build", "-overlay="+mb.overlayPath, "-o", outPath, "cmd/"+toolName)
	cmd.Dir = filepath.Join(mb.fork.goroot, "src")
	cmd.Env = baseEnv(forkCache(), mb.fork.goroot)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%v\n%s", err, clip(string(out)))
	}
	return outPath, nil
}

func (mb *mutBuild) run(t *testing.T) (string, error) {
	t.Helper()
	sb := stagedBuild{bin: mb.driverBin(), stage: mb.stage, outDir: mb.outDir, args: mb.args, env: mb.env, progs: slices.Concat(mb.plain, mb.cgoProg), wantCgo: true}
	_, out, err := sb.build()
	return out, err
}

func (mb *mutBuild) buildFile(t *testing.T, stem string) error {
	t.Helper()
	args := append([]string{"build"}, append(slices.Clone(mb.args), "-o", filepath.Join(mb.scratch, "failbin-"+stem), "./"+stem)...)
	ctx, cancel := context.WithTimeoutCause(context.Background(), 5*time.Minute, errors.New("goparity fail build timeout"))
	defer cancel()
	cmd := exec.CommandContext(ctx, mb.driverBin(), args...)
	cmd.Dir = mb.stage
	cmd.Env = mb.env
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("mutant build of %s failed: %v\n%s", stem, err, clip(string(out)))
	}
	return err
}

func (mb *mutBuild) cleanExpect(t *testing.T, prog string) (observation, error) {
	t.Helper()
	outDir := filepath.Join(mb.scratch, "out-clean")
	env := appendEnv(baseEnv(forkCache(), mb.fork.goroot), mb.cfg.env...)
	sb := stagedBuild{bin: mb.driverBin(), stage: mb.stage, outDir: outDir, args: slices.Clone(mb.cfg.args), env: env, progs: []string{prog}, wantCgo: true}
	if _, out, err := sb.build(); err != nil {
		return observation{}, fmt.Errorf("clean build failed: %v\n%s", err, clip(out))
	}
	return runBinary(filepath.Join(outDir, prog), "4", mb.cfg.args)
}

func (mb *mutBuild) cleanFile(t *testing.T, stem string) error {
	t.Helper()
	env := appendEnv(baseEnv(forkCache(), mb.fork.goroot), mb.cfg.env...)
	args := append([]string{"build"}, append(slices.Clone(mb.cfg.args), "-o", filepath.Join(mb.scratch, "failbin-clean-"+stem), "./"+stem)...)
	ctx, cancel := context.WithTimeoutCause(context.Background(), 5*time.Minute, errors.New("goparity fail build timeout"))
	defer cancel()
	cmd := exec.CommandContext(ctx, mb.driverBin(), args...)
	cmd.Dir = mb.stage
	cmd.Env = env
	return cmd.Run()
}

var attackID = regexp.MustCompile(`^at\d+`)

func attacksWithoutMutant(entries, mutantStems []string) []string {
	var out []string
	for _, e := range entries {
		id := attackID.FindString(e)
		if id != "" && !slices.Contains(mutantStems, id) && !slices.Contains(out, id) {
			out = append(out, id)
		}
	}
	return out
}

func TestAttacksWithoutMutantProbe(t *testing.T) {
	entries := []string{"at01_a", "at17a_x", "at17b_y", "at22_z", "at14_w", "README"}
	if got := attacksWithoutMutant(entries, []string{"at01", "at17", "at22", "at14"}); len(got) != 0 {
		t.Errorf("covered attacks reported: %v", got)
	}
	if got := attacksWithoutMutant(entries, []string{"at01", "at14"}); !slices.Equal(got, []string{"at17", "at22"}) {
		t.Errorf("uncovered attacks = %v, want [at17 at22]", got)
	}
}

func TestParityEveryAttackHasAMutant(t *testing.T) {
	matches, err := filepath.Glob("testdata/mutants/*.txt")
	if err != nil {
		t.Fatalf("globbing testdata/mutants: %v", err)
	}
	stems := make([]string, 0, len(matches))
	for _, m := range matches {
		stems = append(stems, strings.TrimSuffix(filepath.Base(m), ".txt"))
	}
	for _, id := range attacksWithoutMutant(slices.Concat(corpusPrograms(t), corpusFails(t)), stems) {
		t.Errorf("attack %s has corpus entries and no testdata/mutants/%s.txt", id, id)
	}
}

func TestParityMutantProbesDefined(t *testing.T) {
	fork := forkTool(t)
	progs := corpusPrograms(t)
	fails := corpusFails(t)
	ms := loadMutants(t)
	if len(ms) == 0 {
		t.Fatal("no mutants under testdata/mutants")
	}
	cfgs := configIndex()
	for _, m := range ms {
		t.Run(m.id+"-"+m.expect, func(t *testing.T) {
			if _, ok := mutantToolNames[m.kind]; !ok && m.kind != "runtime" && m.kind != "std" {
				t.Errorf("mutant %s: unsupported kind %q (the harness applies compiler, linker, cgo, rules, runtime, std)", m.id, m.kind)
			}
			if _, ok := cfgs[m.cfg]; !ok {
				t.Errorf("mutant %s: unknown config %s", m.id, m.cfg)
			}
			if !slices.Contains(progs, m.expect) && !slices.Contains(fails, strings.TrimSuffix(m.expect, ".go")) {
				t.Errorf("mutant %s: expect target %q is neither a run program nor a fail corpus file", m.id, m.expect)
			}
			target := filepath.Join(fork.goroot, filepath.FromSlash(m.file))
			src, err := os.ReadFile(target)
			if err != nil {
				t.Fatalf("reading mutant target: %v", err)
			}
			if n := strings.Count(string(src), m.old); n != 1 {
				t.Errorf("mutant %s: old text occurs %d times in %s: stale probe", m.id, n, m.file)
			}
		})
	}
}

func TestParityMutants(t *testing.T) {
	if !*releaseFlag {
		t.Skip("release tier only; pass -goparity.release")
	}
	fork := forkTool(t)
	up, upOn := upstreamTool(t)
	progs := corpusPrograms(t)
	fails := corpusFails(t)
	ms := loadMutants(t)
	if len(ms) == 0 {
		t.Fatal("no mutants under testdata/mutants")
	}
	base := t.TempDir()
	cfgs := configIndex()
	var shim string
	needShim := false
	for _, m := range ms {
		if _, ok := mutantToolNames[m.kind]; ok {
			needShim = true
		} else if m.kind != "runtime" && m.kind != "std" {
			t.Errorf("mutant %s: unsupported kind %q (the harness applies compiler, linker, cgo, rules, runtime, std)", m.id, m.kind)
		}
	}
	if needShim {
		shim = filepath.Join(base, "toolexec-shim")
		pkgDir, err := filepath.Abs("testdata/toolexec")
		if err != nil {
			t.Fatalf("resolving testdata/toolexec: %v", err)
		}
		ctx, cancel := context.WithTimeoutCause(context.Background(), 5*time.Minute, errors.New("goparity shim build timeout"))
		defer cancel()
		cmd := exec.CommandContext(ctx, fork.bin, "build", "-o", shim, pkgDir)
		cmd.Env = baseEnv(forkCache(), fork.goroot)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("building toolexec shim: %v\n%s", err, clip(string(out)))
		}
	}
	upOnce := &upOnceBuild{up: up, base: base, progs: progs, on: upOn}
	for _, m := range ms {
		t.Run(m.id+"-"+m.expect, func(t *testing.T) {
			cfg, ok := cfgs[m.cfg]
			if !ok {
				t.Fatalf("mutant %s names unknown config %s", m.id, m.cfg)
			}
			target := filepath.Join(fork.goroot, filepath.FromSlash(m.file))
			src, err := os.ReadFile(target)
			if err != nil {
				t.Fatalf("reading mutant target: %v", err)
			}
			if n := strings.Count(string(src), m.old); n != 1 {
				t.Fatalf("mutant %s: old text occurs %d times in %s: stale mutant", m.id, n, m.file)
			}
			scratch := t.TempDir()
			mutPath := filepath.Join(scratch, "with-"+filepath.Base(target))
			if err := os.WriteFile(mutPath, []byte(strings.Replace(string(src), m.old, m.new, 1)), 0o644); err != nil {
				t.Fatalf("writing mutant copy: %v", err)
			}
			overlayBytes, err := json.Marshal(map[string]map[string]string{"Replace": {target: mutPath}})
			if err != nil {
				t.Fatalf("marshalling overlay: %v", err)
			}
			overlayPath := filepath.Join(scratch, "overlay.json")
			if err := os.WriteFile(overlayPath, overlayBytes, 0o644); err != nil {
				t.Fatalf("writing overlay: %v", err)
			}
			mb := &mutBuild{fork: fork, cfg: cfg, stage: filepath.Join(scratch, "src"), scratch: scratch, shim: shim, overlayPath: overlayPath}
			mb.corpus(t, progs)
			mb.apply(t, m)
			failStem := strings.TrimSuffix(m.expect, ".go")
			if slices.Contains(fails, failStem) {
				if cerr := mb.cleanFile(t, failStem); cerr == nil {
					t.Errorf("mutant %s unevaluable: fail corpus file %s already builds under %s without the mutant", m.id, m.expect, m.cfg)
					return
				}
				if err := mb.buildFile(t, failStem); err != nil {
					t.Errorf("mutant %s survived: fail corpus file %s is still rejected under %s", m.id, m.expect, m.cfg)
					return
				}
				t.Logf("mutant killed: fail corpus file %s now builds under %s", m.expect, m.cfg)
				return
			}
			if !slices.Contains(progs, m.expect) {
				t.Fatalf("mutant %s: expect target %q is neither a run program nor a fail corpus file", m.id, m.expect)
			}
			buildOut, buildErr := mb.run(t)
			if buildErr != nil {
				t.Fatalf("mutant %s: corpus build failed under mutant (invalid probe): %v\n%s", m.id, buildErr, clip(buildOut))
			}
			fobs, ferr := runBinary(filepath.Join(mb.outDir, m.expect), "4", mb.args)
			if ferr != nil {
				t.Fatalf("mutant %s: %v", m.id, ferr)
			}
			red := corpusDefect(fobs)
			if upOn {
				ub, err := upOnce.build(t, cfg.upstream)
				if err != nil {
					t.Fatalf("upstream build %s failed: %v\n%s", cfg.upstream, err, clip(ub.output))
				}
				uobs, uerr := runBinary(filepath.Join(ub.outDir, m.expect), "4", cfg.args)
				if uerr != nil {
					t.Fatalf("mutant %s: upstream run: %v", m.id, uerr)
				}
				clean, cerr := mb.cleanExpect(t, m.expect)
				if cerr != nil {
					t.Fatalf("mutant %s: clean baseline: %v", m.id, cerr)
				}
				if !obsEqual(clean, uobs) {
					t.Errorf("mutant %s unevaluable: %s already red under %s without the mutant (clean %s vs upstream %s)", m.id, m.expect, m.cfg, obsDesc(clean), obsDesc(uobs))
					return
				}
				if !obsEqual(fobs, uobs) {
					red = true
				}
			}
			if !red {
				if !upOn {
					t.Skipf("mutant %s: cannot evaluate detection with the upstream oracle off: the fork stayed green under %s, so a surviving mutant is indistinguishable from an equivalent one", m.id, m.cfg)
				}
				t.Errorf("mutant %s survived: %s stayed green under %s (exit %d)", m.id, m.expect, m.cfg, fobs.exit)
			}
		})
	}
}

type upOnceBuild struct {
	mu    sync.Mutex
	up    toolchain
	base  string
	progs []string
	on    bool
	done  map[string]buildResult
}

func (uo *upOnceBuild) build(t *testing.T, uID string) (buildResult, error) {
	t.Helper()
	if !uo.on {
		return buildResult{}, errors.New("upstream oracle off")
	}
	var u *upConfig
	for i := range upConfigs {
		if upConfigs[i].id == uID {
			u = &upConfigs[i]
		}
	}
	if u == nil {
		return buildResult{}, fmt.Errorf("unknown upstream config %s", uID)
	}
	uo.mu.Lock()
	defer uo.mu.Unlock()
	if b, ok := uo.done[u.id]; ok {
		return b, b.err
	}
	if uo.done == nil {
		uo.done = map[string]buildResult{}
	}
	stage := filepath.Join(uo.base, "mut-up-"+u.id)
	if err := stageCorpus(uo.up, stage, upCache(), uo.progs); err != nil {
		return buildResult{}, fmt.Errorf("staging corpus: %w", err)
	}
	env := baseEnv(upCache(), uo.up.goroot)
	if u.race {
		env = appendEnv(env, "CGO_ENABLED=1")
	}
	env = appendEnv(env, u.env...)
	outDir := filepath.Join(uo.base, "mut-out-up-"+u.id)
	sb := stagedBuild{bin: uo.up.bin, stage: stage, outDir: outDir, args: u.args, env: env, progs: uo.progs, wantCgo: true}
	built, out, err := sb.build()
	b := buildResult{outDir: outDir, output: out, err: err, progs: built}
	uo.done[u.id] = b
	return b, b.err
}

func sweepEligible(t *testing.T, goroot string) []string {
	t.Helper()
	var out []string
	for _, dir := range []string{".", "ken", "chan", "interface", "fixedbugs", "typeparam"} {
		matches, err := filepath.Glob(filepath.Join(goroot, "test", dir, "*.go"))
		if err != nil {
			t.Fatalf("globbing sweep dir %s: %v", dir, err)
		}
		for _, m := range matches {
			data, err := os.ReadFile(m)
			if err != nil {
				t.Fatalf("reading sweep file %s: %v", m, err)
			}
			if hasBareRun(data) {
				out = append(out, m)
			}
		}
	}
	slices.Sort(out)
	return out
}

func hasBareRun(data []byte) bool {
	bare := false
	for ln := range strings.SplitSeq(string(data), "\n") {
		t := strings.TrimSpace(ln)
		if t == "// run" {
			bare = true
			continue
		}
		for _, p := range []string{"// run", "// runoutput", "// rundir", "// runindir", "// buildrundir"} {
			if t == p || strings.HasPrefix(t, p+" ") || strings.HasPrefix(t, p+"\t") {
				return false
			}
		}
	}
	return bare
}

const sweepExclusionFrac = 0.10

type sweepFile struct {
	pkg   string
	fname string
	data  []byte
}

type sweepJob struct {
	fork toolchain
	up   toolchain
	base string
	cfgs []config
	ups  []upConfig
}

func maskBin(o observation, bin string) observation {
	o.stdout = strings.ReplaceAll(o.stdout, bin, "BIN")
	o.stderr = strings.ReplaceAll(o.stderr, bin, "BIN")
	return o
}

var sweepPanicHead = regexp.MustCompile(`^(panic: |fatal error: |\[recovered\]|\s+panic: |\s*\[signal SIG)`)

func sweepObs(o observation, bin string) observation {
	o = maskBin(o, bin)
	var kept []string
	for ln := range strings.SplitSeq(o.stderr, "\n") {
		if sweepPanicHead.MatchString(ln) {
			kept = append(kept, ln)
		}
	}
	o.stderr = strings.Join(kept, "\n")
	return o
}

func stageSweepPkg(dir, pkg string, data []byte, fname, toml string) error {
	if err := os.MkdirAll(filepath.Join(dir, pkg), 0o777); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module sweep\n\ngo 1.27\n"), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, pkg, fname), data, 0o644); err != nil {
		return err
	}
	if toml != "" {
		if err := os.WriteFile(filepath.Join(dir, "logos.toml"), []byte(toml), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func (j sweepJob) buildRow(bin, stage, outDir string, args, env []string, pkg string) (string, error) {
	sb := stagedBuild{bin: bin, stage: stage, outDir: outDir, args: args, env: env, progs: []string{pkg}, wantCgo: true}
	_, out, err := sb.build()
	return out, err
}

type sweepPlaced struct {
	bin string
	obs observation
}

func (j sweepJob) compareRow(c config, fbin string, fobs observation, upObs map[string]sweepPlaced) string {
	uo := upObs[c.upstream]
	if obsEqual(fobs, uo.obs) {
		return ""
	}
	fo2, err := runSweepBinary(fbin, c.args)
	if err != nil {
		return fmt.Sprintf("fork %s rerun failed: %v", c.id, err)
	}
	uo2, err := runSweepBinary(uo.bin, c.args)
	if err != nil {
		return fmt.Sprintf("upstream %s rerun failed: %v", c.upstream, err)
	}
	if obsEqual(sweepObs(fo2, fbin), sweepObs(uo2, uo.bin)) {
		return ""
	}
	d := firstDivergence(fobs, uo.obs)
	return fmt.Sprintf("%s vs %s: %s (%s vs %s)", c.id, c.upstream, d, obsDesc(fobs), obsDesc(uo.obs))
}

func (j sweepJob) check(f sweepFile) (excluded, failed string) {
	var dirs []string
	defer func() {
		for _, d := range dirs {
			os.RemoveAll(d)
		}
	}()
	upObs := make(map[string]sweepPlaced, len(j.ups))
	for _, u := range j.ups {
		st := filepath.Join(j.base, "st-"+f.pkg+"-"+u.id)
		if err := stageSweepPkg(st, f.pkg, f.data, f.fname, ""); err != nil {
			return fmt.Sprintf("staging failed: %v", err), ""
		}
		dirs = append(dirs, st)
		outDir := filepath.Join(j.base, "out-"+f.pkg+"-"+u.id)
		dirs = append(dirs, outDir)
		env := baseEnv(upCache(), j.up.goroot)
		if u.race {
			env = appendEnv(env, "CGO_ENABLED=1")
		}
		env = appendEnv(env, u.env...)
		out, err := j.buildRow(j.up.bin, st, outDir, u.args, env, f.pkg)
		if err != nil {
			return "upstream build failed: " + clip(err.Error()+" "+out), ""
		}
		bin := filepath.Join(outDir, f.pkg)
		o, err := runSweepBinary(bin, u.args)
		if err != nil {
			return "", fmt.Sprintf("upstream %s run failed: %v", u.id, err)
		}
		upObs[u.id] = sweepPlaced{bin, sweepObs(o, bin)}
	}
	for _, u := range j.ups {
		if u.race {
			continue
		}
		if !obsEqual(upObs[u.id].obs, upObs[j.ups[0].id].obs) {
			return "upstream rows differ (no stable oracle)", ""
		}
	}
	for _, c := range j.cfgs {
		st := filepath.Join(j.base, "st-"+f.pkg+"-"+c.id)
		if err := stageSweepPkg(st, f.pkg, f.data, f.fname, c.toml); err != nil {
			return fmt.Sprintf("staging failed: %v", err), ""
		}
		dirs = append(dirs, st)
		outDir := filepath.Join(j.base, "out-"+f.pkg+"-"+c.id)
		dirs = append(dirs, outDir)
		bin := j.fork.bin
		if c.logos {
			bin = j.fork.logos
		}
		out, err := j.buildRow(bin, st, outDir, c.args, appendEnv(baseEnv(forkCache(), j.fork.goroot), c.env...), f.pkg)
		if err != nil {
			return "", fmt.Sprintf("fork %s build failed where upstream builds: %v\n%s", c.id, err, clip(out))
		}
		fbin := filepath.Join(outDir, f.pkg)
		fo, err := runSweepBinary(fbin, c.args)
		if err != nil {
			return "", fmt.Sprintf("fork %s run failed: %v", c.id, err)
		}
		if failed := j.compareRow(c, fbin, sweepObs(fo, fbin), upObs); failed != "" {
			return "", failed
		}
	}
	return "", ""
}

func TestParitySweepCorpusDefined(t *testing.T) {
	up, upOn := upstreamTool(t)
	if !upOn {
		t.Skip("upstream oracle off: the sweep compares against live upstream output")
	}
	files := sweepEligible(t, up.goroot)
	if len(files) == 0 {
		t.Fatal("no sweep-eligible files under upstream test/")
	}
	if sweepExclusionFrac <= 0 || sweepExclusionFrac >= 1 {
		t.Fatalf("sweep exclusion fraction %v outside (0, 1)", sweepExclusionFrac)
	}
	t.Logf("sweep corpus: %d eligible files", len(files))
}

func TestParityUpstreamSweep(t *testing.T) {
	if !*releaseFlag {
		t.Skip("release tier only; pass -goparity.release")
	}
	start := time.Now()
	fork := forkTool(t)
	up, upOn := upstreamTool(t)
	if !upOn {
		t.Skip("upstream oracle off: the sweep compares against live upstream output")
	}
	files := sweepEligible(t, up.goroot)
	if len(files) == 0 {
		t.Fatal("no sweep-eligible files under upstream test/")
	}
	sfiles := make([]sweepFile, 0, len(files))
	for i, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("reading sweep file %s: %v", f, err)
		}
		sfiles = append(sfiles, sweepFile{pkg: fmt.Sprintf("f%03d", i), fname: filepath.Base(f), data: data})
	}
	ups := activeUpConfigs(forkConfigs)
	base := t.TempDir()
	type outcome struct {
		excluded string
		failed   string
	}
	outcomes := make([]outcome, len(files))
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	job := sweepJob{fork: fork, up: up, base: base, cfgs: forkConfigs, ups: ups}
	for i := range files {
		wg.Go(func() {
			sem <- struct{}{}
			defer func() { <-sem }()
			ex, fa := job.check(sfiles[i])
			outcomes[i] = outcome{ex, fa}
		})
	}
	wg.Wait()
	ran, skipped := 0, 0
	for i, o := range outcomes {
		rel, _ := filepath.Rel(filepath.Join(up.goroot, "test"), files[i])
		if o.failed != "" {
			t.Errorf("sweep: %s: %s", rel, o.failed)
			continue
		}
		if o.excluded != "" {
			t.Logf("sweep: excluded %s: %s", rel, o.excluded)
			skipped++
			continue
		}
		ran++
	}
	if ran == 0 {
		t.Errorf("sweep: no files compared")
	}
	if budget := int(sweepExclusionFrac * float64(len(files))); skipped > budget {
		t.Errorf("sweep: %d exclusions exceed budget %d (%d files)", skipped, budget, len(files))
	}
	t.Logf("sweep: %d files compared, %d excluded, %d fork configs, %d upstream rows, wall %s", ran, skipped, len(forkConfigs), len(ups), time.Since(start))
}
