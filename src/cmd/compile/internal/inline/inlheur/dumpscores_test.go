// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDumpCallSiteScoreDump(t *testing.T) {
	td := t.TempDir()
	testenv.MustHaveGoBuild(t)

	scenarios := []struct {
		name               string
		promoted           int
		indirectlyPromoted int
		demoted            int
		unchanged          int
	}{
		{
			name:               "dumpscores",
			promoted:           1,
			indirectlyPromoted: 1,
			demoted:            1,
			unchanged:          5,
		},
	}

	for _, scen := range scenarios {
		dumpfile, err := gatherInlCallSitesScoresForFile(t, scen.name, td)
		if err != nil {
			t.Fatalf("dumping callsite scores for %q: error %v", scen.name, err)
		}
		var lines []string
		if content, err := os.ReadFile(dumpfile); err != nil {
			t.Fatalf("reading dump %q: error %v", dumpfile, err)
		} else {
			lines = strings.Split(string(content), "\n")
		}
		prom, indprom, dem, unch := 0, 0, 0, 0
		for _, line := range lines {
			switch {
			case strings.TrimSpace(line) == "":
			case !strings.Contains(line, "|"):
			case strings.HasPrefix(line, "#"):
			case strings.Contains(line, "PROMOTED"):
				prom++
			case strings.Contains(line, "INDPROM"):
				indprom++
			case strings.Contains(line, "DEMOTED"):
				dem++
			default:
				unch++
			}
		}
		showout := false
		if prom != scen.promoted {
			t.Errorf("testcase %q, got %d promoted want %d promoted",
				scen.name, prom, scen.promoted)
			showout = true
		}
		if indprom != scen.indirectlyPromoted {
			t.Errorf("testcase %q, got %d indirectly promoted want %d",
				scen.name, indprom, scen.indirectlyPromoted)
			showout = true
		}
		if dem != scen.demoted {
			t.Errorf("testcase %q, got %d demoted want %d demoted",
				scen.name, dem, scen.demoted)
			showout = true
		}
		if unch != scen.unchanged {
			t.Errorf("testcase %q, got %d unchanged want %d unchanged",
				scen.name, unch, scen.unchanged)
			showout = true
		}
		if showout {
			t.Logf(">> dump output: %s", strings.Join(lines, "\n"))
		}
	}
}

// gatherInlCallSitesScoresForFile builds the specified testcase 'testcase'
// from testdata/props passing the "-d=dumpinlcallsitescores=1"
// compiler option, to produce a dump, then returns the path of the
// newly created file.
func gatherInlCallSitesScoresForFile(t *testing.T, testcase string, td string) (string, error) {
	t.Helper()
	gopath := "testdata/" + testcase + ".go"
	outpath := filepath.Join(td, testcase+".a")
	dumpfile := filepath.Join(td, testcase+".callsites.txt")
	run := []string{testenv.GoToolPath(t), "build",
		"-gcflags=-d=dumpinlcallsitescores=1", "-o", outpath, gopath}
	out, err := testenv.Command(t, run[0], run[1:]...).CombinedOutput()
	t.Logf("run: %+v\n", run)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(dumpfile, out, 0666); err != nil {
		return "", err
	}
	return dumpfile, err
}
