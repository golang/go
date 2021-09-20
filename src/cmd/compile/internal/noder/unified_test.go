// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder_test

import (
	"encoding/json"
	"flag"
	exec "internal/execabs"
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

var (
	flagCmp      = flag.Bool("cmp", false, "enable TestUnifiedCompare")
	flagPkgs     = flag.String("pkgs", "std", "list of packages to compare (ignored in -short mode)")
	flagAll      = flag.Bool("all", false, "enable testing of all GOOS/GOARCH targets")
	flagParallel = flag.Bool("parallel", false, "test GOOS/GOARCH targets in parallel")
)

// TestUnifiedCompare implements a test similar to running:
//
//	$ go build -toolexec="toolstash -cmp" std
//
// The -pkgs flag controls the list of packages tested.
//
// By default, only the native GOOS/GOARCH target is enabled. The -all
// flag enables testing of non-native targets. The -parallel flag
// additionally enables testing of targets in parallel.
//
// Caution: Testing all targets is very resource intensive! On an IBM
// P920 (dual Intel Xeon Gold 6154 CPUs; 36 cores, 192GB RAM), testing
// all targets in parallel takes about 5 minutes. Using the 'go test'
// command's -run flag for subtest matching is recommended for less
// powerful machines.
func TestUnifiedCompare(t *testing.T) {
	// TODO(mdempsky): Either re-enable or delete. Disabled for now to
	// avoid impeding others' forward progress.
	if !*flagCmp {
		t.Skip("skipping TestUnifiedCompare (use -cmp to enable)")
	}

	targets, err := exec.Command("go", "tool", "dist", "list").Output()
	if err != nil {
		t.Fatal(err)
	}

	for _, target := range strings.Fields(string(targets)) {
		t.Run(target, func(t *testing.T) {
			parts := strings.Split(target, "/")
			goos, goarch := parts[0], parts[1]

			if !(*flagAll || goos == runtime.GOOS && goarch == runtime.GOARCH) {
				t.Skip("skipping non-native target (use -all to enable)")
			}
			if *flagParallel {
				t.Parallel()
			}

			pkgs1 := loadPackages(t, goos, goarch, "-d=unified=0 -d=inlfuncswithclosures=0 -d=unifiedquirks=1 -G=0")
			pkgs2 := loadPackages(t, goos, goarch, "-d=unified=1 -d=inlfuncswithclosures=0 -d=unifiedquirks=1 -G=0")

			if len(pkgs1) != len(pkgs2) {
				t.Fatalf("length mismatch: %v != %v", len(pkgs1), len(pkgs2))
			}

			for i := range pkgs1 {
				pkg1 := pkgs1[i]
				pkg2 := pkgs2[i]

				path := pkg1.ImportPath
				if path != pkg2.ImportPath {
					t.Fatalf("mismatched paths: %q != %q", path, pkg2.ImportPath)
				}

				// Packages that don't have any source files (e.g., packages
				// unsafe, embed/internal/embedtest, and cmd/internal/moddeps).
				if pkg1.Export == "" && pkg2.Export == "" {
					continue
				}

				if pkg1.BuildID == pkg2.BuildID {
					t.Errorf("package %q: build IDs unexpectedly matched", path)
				}

				// Unlike toolstash -cmp, we're comparing the same compiler
				// binary against itself, just with different flags. So we
				// don't need to worry about skipping over mismatched version
				// strings, but we do need to account for differing build IDs.
				//
				// Fortunately, build IDs are cryptographic 256-bit hashes,
				// and cmd/go provides us with them up front. So we can just
				// use them as delimeters to split the files, and then check
				// that the substrings are all equal.
				file1 := strings.Split(readFile(t, pkg1.Export), pkg1.BuildID)
				file2 := strings.Split(readFile(t, pkg2.Export), pkg2.BuildID)
				if !reflect.DeepEqual(file1, file2) {
					t.Errorf("package %q: compile output differs", path)
				}
			}
		})
	}
}

type pkg struct {
	ImportPath string
	Export     string
	BuildID    string
	Incomplete bool
}

func loadPackages(t *testing.T, goos, goarch, gcflags string) []pkg {
	args := []string{"list", "-e", "-export", "-json", "-gcflags=all=" + gcflags, "--"}
	if testing.Short() {
		t.Log("short testing mode; only testing package runtime")
		args = append(args, "runtime")
	} else {
		args = append(args, strings.Fields(*flagPkgs)...)
	}

	cmd := exec.Command("go", args...)
	cmd.Env = append(os.Environ(), "GOOS="+goos, "GOARCH="+goarch)
	cmd.Stderr = os.Stderr
	t.Logf("running %v", cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	var res []pkg
	for dec := json.NewDecoder(stdout); dec.More(); {
		var pkg pkg
		if err := dec.Decode(&pkg); err != nil {
			t.Fatal(err)
		}
		if pkg.Incomplete {
			t.Fatalf("incomplete package: %q", pkg.ImportPath)
		}
		res = append(res, pkg)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	return res
}

func readFile(t *testing.T, name string) string {
	buf, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	return string(buf)
}
