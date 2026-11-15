// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
)

func ejectTestSaveToolEnv(t *testing.T) {
	t.Helper()
	savedGOROOT, savedGOBIN := cfg.GOROOT, cfg.GOBIN
	savedSuffix, savedLimit := ejectToolExeSuffix, ejectArgLengthLimit
	t.Cleanup(func() {
		cfg.GOROOT, cfg.GOBIN = savedGOROOT, savedGOBIN
		ejectToolExeSuffix, ejectArgLengthLimit = savedSuffix, savedLimit
	})
}

func ejectTestWriteScript(t *testing.T, path, script string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
}

func TestGofmtBinaryUsesToolExeSuffix(t *testing.T) {
	ejectTestSaveToolEnv(t)
	ejectToolExeSuffix = ".exe"
	goroot := t.TempDir()
	binDir := filepath.Join(goroot, "bin")
	if err := os.MkdirAll(binDir, 0777); err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(binDir, "gofmt"+ejectToolExeSuffix)
	ejectTestWriteScript(t, want, "#!/bin/sh\nexit 0\n")
	decoy := t.TempDir()
	ejectTestWriteScript(t, filepath.Join(decoy, "gofmt"), "#!/bin/sh\nexit 0\n")
	cfg.GOROOT = goroot
	cfg.GOBIN = t.TempDir()
	t.Setenv("PATH", decoy)
	got := gofmtBinary()
	if !strings.HasSuffix(got, ejectToolExeSuffix) {
		t.Fatalf("gofmtBinary() = %q, must end with the tool executable suffix %q", got, ejectToolExeSuffix)
	}
	if got != want {
		t.Fatalf("gofmtBinary() = %q, want %q (resolution ignored the tool executable suffix and fell back to PATH)", got, want)
	}
}

func TestRunEjectGofmtBatchesUnderArgLimit(t *testing.T) {
	ejectTestSaveToolEnv(t)
	ejectToolExeSuffix = ""
	goroot := t.TempDir()
	if err := os.MkdirAll(filepath.Join(goroot, "bin"), 0777); err != nil {
		t.Fatal(err)
	}
	stub := filepath.Join(goroot, "bin", "gofmt")
	script := "#!/bin/sh\necho \"$#\" >> \"$EJECT_TEST_GOFMT_LOG\"\nprintf '%s\\n' \"$@\" >> \"$EJECT_TEST_GOFMT_LOG\"\n"
	ejectTestWriteScript(t, stub, script)
	cfg.GOROOT = goroot
	cfg.GOBIN = t.TempDir()
	t.Setenv("PATH", t.TempDir())

	var paths []string
	for i := range 1000 {
		paths = append(paths, "/"+strings.Repeat("d", 95)+fmt.Sprintf("%04d", i))
	}
	base := len(stub) + len(" -w")
	limit := base + 12*101 + 1
	var counts []int
	for _, l := range []int{limit - 1, limit, limit + 1} {
		t.Run(strconv.Itoa(l), func(t *testing.T) {
			log := filepath.Join(t.TempDir(), "gofmt.log")
			t.Setenv("EJECT_TEST_GOFMT_LOG", log)
			ejectArgLengthLimit = l
			ejectGofmtFiles = append(ejectGofmtFiles[:0], paths...)
			runEjectGofmt(t.Context())
			data, err := os.ReadFile(log)
			if err != nil {
				t.Fatal(err)
			}
			lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
			passed := map[string][]string{}
			batches := map[string]int{}
			for i := 0; i < len(lines); {
				n, err := strconv.Atoi(lines[i])
				if err != nil {
					t.Fatalf("bad argc line %q: %v", lines[i], err)
				}
				if i+1+n > len(lines) {
					t.Fatalf("truncated gofmt log")
				}
				args := lines[i+1 : i+1+n]
				i += 1 + n
				if len(args) == 0 || (args[0] != "-w" && args[0] != "-l") {
					t.Fatalf("unexpected gofmt argv %q", args)
				}
				batches[args[0]]++
				total := len(stub) + 1 + len(args[0])
				for _, f := range args[1:] {
					total += 1 + len(f)
				}
				if total >= l {
					t.Fatalf("gofmt %s batch %d: argument length %d, limit %d (batched by count, not argument length)", args[0], batches[args[0]], total, l)
				}
				passed[args[0]] = append(passed[args[0]], args[1:]...)
			}
			for _, flag := range []string{"-w", "-l"} {
				if !slices.Equal(passed[flag], paths) {
					t.Fatalf("files passed to gofmt %s diverge from input: got %d, want %d", flag, len(passed[flag]), len(paths))
				}
			}
			counts = append(counts, batches["-w"])
		})
	}
	if len(counts) == 3 && !(counts[0] > counts[1] && counts[1] >= counts[2]) {
		t.Fatalf("batch counts must not grow as the limit loosens: limit-1=%d limit=%d limit+1=%d", counts[0], counts[1], counts[2])
	}
}
