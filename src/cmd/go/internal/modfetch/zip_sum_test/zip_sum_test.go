// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package zip_sum_test tests that the module zip files produced by modfetch
// have consistent content sums. Ideally the zip files themselves are also
// stable over time, though this is not strictly necessary.
//
// This test loads a table from testdata/zip_sums.csv. The table has columns
// for module path, version, content sum, and zip file hash. The table
// includes a large number of real modules. The test downloads these modules
// in direct mode and verifies the zip files.
//
// This test is very slow, and it depends on outside modules that change
// frequently, so this is a manual test. To enable it, pass the -zipsum flag.
package zip_sum_test

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"flag"
	"fmt"
	"internal/testenv"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"

	"golang.org/x/mod/module"
)

var (
	updateTestData = flag.Bool("u", false, "when set, tests may update files in testdata instead of failing")
	enableZipSum   = flag.Bool("zipsum", false, "enable TestZipSums")
	debugZipSum    = flag.Bool("testwork", false, "when set, TestZipSums will preserve its test directory")
	modCacheDir    = flag.String("zipsumcache", "", "module cache to use instead of temp directory")
	shardCount     = flag.Int("zipsumshardcount", 1, "number of shards to divide TestZipSums into")
	shardIndex     = flag.Int("zipsumshard", 0, "index of TestZipSums shard to test (0 <= zipsumshard < zipsumshardcount)")
)

const zipSumsPath = "testdata/zip_sums.csv"

type zipSumTest struct {
	m                     module.Version
	wantSum, wantFileHash string
}

func TestZipSums(t *testing.T) {
	if !*enableZipSum {
		// This test is very slow and heavily dependent on external repositories.
		// Only run it explicitly.
		t.Skip("TestZipSum not enabled with -zipsum")
	}
	if *shardCount < 1 {
		t.Fatal("-zipsumshardcount must be a positive integer")
	}
	if *shardIndex < 0 || *shardCount <= *shardIndex {
		t.Fatal("-zipsumshard must be between 0 and -zipsumshardcount")
	}

	testenv.MustHaveGoBuild(t)
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "bzr")
	testenv.MustHaveExecPath(t, "git")
	// TODO(jayconrod): add hg, svn, and fossil modules to testdata.
	// Could not find any for now.

	tests, err := readZipSumTests()
	if err != nil {
		t.Fatal(err)
	}

	if *modCacheDir != "" {
		cfg.BuildContext.GOPATH = *modCacheDir
	} else {
		tmpDir, err := ioutil.TempDir("", "TestZipSums")
		if err != nil {
			t.Fatal(err)
		}
		if *debugZipSum {
			fmt.Fprintf(os.Stderr, "TestZipSums: modCacheDir: %s\n", tmpDir)
		} else {
			defer os.RemoveAll(tmpDir)
		}
		cfg.BuildContext.GOPATH = tmpDir
	}

	cfg.GOPROXY = "direct"
	cfg.GOSUMDB = "off"
	modload.Init()

	// Shard tests by downloading only every nth module when shard flags are set.
	// This makes it easier to test small groups of modules quickly. We avoid
	// testing similarly named modules together (the list is sorted by module
	// path and version).
	if *shardCount > 1 {
		r := *shardIndex
		w := 0
		for r < len(tests) {
			tests[w] = tests[r]
			w++
			r += *shardCount
		}
		tests = tests[:w]
	}

	// Download modules with a rate limit. We may run out of file descriptors
	// or cause timeouts without a limit.
	needUpdate := false
	for i := range tests {
		test := &tests[i]
		name := fmt.Sprintf("%s@%s", strings.ReplaceAll(test.m.Path, "/", "_"), test.m.Version)
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			zipPath, err := modfetch.DownloadZip(test.m)
			if err != nil {
				if *updateTestData {
					t.Logf("%s: could not download module: %s (will remove from testdata)", test.m, err)
					test.m.Path = "" // mark for deletion
					needUpdate = true
				} else {
					t.Errorf("%s: could not download mdoule: %s", test.m, err)
				}
				return
			}

			sum := modfetch.Sum(test.m)
			if sum != test.wantSum {
				if *updateTestData {
					t.Logf("%s: updating content sum to %s", test.m, sum)
					test.wantSum = sum
					needUpdate = true
				} else {
					t.Errorf("%s: got content sum %s; want sum %s", test.m, sum, test.wantSum)
					return
				}
			}

			h := sha256.New()
			f, err := os.Open(zipPath)
			if err != nil {
				t.Errorf("%s: %v", test.m, err)
			}
			defer f.Close()
			if _, err := io.Copy(h, f); err != nil {
				t.Errorf("%s: %v", test.m, err)
			}
			zipHash := hex.EncodeToString(h.Sum(nil))
			if zipHash != test.wantFileHash {
				if *updateTestData {
					t.Logf("%s: updating zip file hash to %s", test.m, zipHash)
					test.wantFileHash = zipHash
					needUpdate = true
				} else {
					t.Errorf("%s: got zip file hash %s; want hash %s (but content sum matches)", test.m, zipHash, test.wantFileHash)
				}
			}
		})
	}

	if needUpdate {
		// Remove tests marked for deletion
		r, w := 0, 0
		for r < len(tests) {
			if tests[r].m.Path != "" {
				tests[w] = tests[r]
				w++
			}
			r++
		}
		tests = tests[:w]

		if err := writeZipSumTests(tests); err != nil {
			t.Error(err)
		}
	}
}

func readZipSumTests() ([]zipSumTest, error) {
	f, err := os.Open(filepath.FromSlash(zipSumsPath))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)

	var tests []zipSumTest
	for {
		line, err := r.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		} else if len(line) != 4 {
			return nil, fmt.Errorf("%s:%d: malformed line", f.Name(), len(tests)+1)
		}
		test := zipSumTest{m: module.Version{Path: line[0], Version: line[1]}, wantSum: line[2], wantFileHash: line[3]}
		tests = append(tests, test)
	}
	return tests, nil
}

func writeZipSumTests(tests []zipSumTest) (err error) {
	f, err := os.Create(filepath.FromSlash(zipSumsPath))
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); err == nil && cerr != nil {
			err = cerr
		}
	}()
	w := csv.NewWriter(f)
	line := make([]string, 0, 4)
	for _, test := range tests {
		line = append(line[:0], test.m.Path, test.m.Version, test.wantSum, test.wantFileHash)
		if err := w.Write(line); err != nil {
			return err
		}
	}
	w.Flush()
	return nil
}
