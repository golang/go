// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/liveness"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"internal/testenv"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func mkiv(name string) *ir.Name {
	i32 := types.Types[types.TINT32]
	s := typecheck.Lookup(name)
	v := ir.NewNameAt(src.NoXPos, s, i32)
	return v
}

func TestMergeLocalState(t *testing.T) {
	v1 := mkiv("v1")
	v2 := mkiv("v2")
	v3 := mkiv("v3")

	testcases := []struct {
		vars      []*ir.Name
		partition map[*ir.Name][]int
		experr    bool
	}{
		{
			vars: []*ir.Name{v1, v2, v3},
			partition: map[*ir.Name][]int{
				v1: {0, 1, 2},
				v2: {0, 1, 2},
				v3: {0, 1, 2},
			},
			experr: false,
		},
		{
			// invalid mls.v slot -1
			vars: []*ir.Name{v1, v2, v3},
			partition: map[*ir.Name][]int{
				v1: {-1, 0},
				v2: {0, 1, 2},
				v3: {0, 1, 2},
			},
			experr: true,
		},
		{
			// duplicate var in v
			vars: []*ir.Name{v1, v2, v2},
			partition: map[*ir.Name][]int{
				v1: {0, 1, 2},
				v2: {0, 1, 2},
				v3: {0, 1, 2},
			},
			experr: true,
		},
		{
			// single element in partition
			vars: []*ir.Name{v1, v2, v3},
			partition: map[*ir.Name][]int{
				v1: {0},
				v2: {0, 1, 2},
				v3: {0, 1, 2},
			},
			experr: true,
		},
		{
			// missing element 2
			vars: []*ir.Name{v1, v2, v3},
			partition: map[*ir.Name][]int{
				v1: {0, 1},
				v2: {0, 1},
				v3: {0, 1},
			},
			experr: true,
		},
		{
			// partitions disagree for v1 vs v2
			vars: []*ir.Name{v1, v2, v3},
			partition: map[*ir.Name][]int{
				v1: {0, 1, 2},
				v2: {1, 0, 2},
				v3: {0, 1, 2},
			},
			experr: true,
		},
	}

	for k, testcase := range testcases {
		mls, err := liveness.MakeMergeLocalsState(testcase.partition, testcase.vars)
		t.Logf("tc %d err is %v\n", k, err)
		if testcase.experr && err == nil {
			t.Fatalf("tc:%d missing error mls %v", k, mls)
		} else if !testcase.experr && err != nil {
			t.Fatalf("tc:%d unexpected error mls %v", k, err)
		}
		if mls != nil {
			t.Logf("tc %d: mls: %v\n", k, mls.String())
		}
	}
}

func TestMergeLocalsIntegration(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This test does a build of a specific canned package to
	// check whether merging of stack slots is taking place.
	// The idea is to do the compile with a trace option turned
	// on and then pick up on the frame offsets of specific
	// variables.
	//
	// Stack slot merging is a greedy algorithm, and there can
	// be many possible ways to overlap a given set of candidate
	// variables, all of them legal. Rather than locking down
	// a specific set of overlappings or frame offsets, this
	// tests just verifies that there is a decent-sized clump of 4+ vars that
	// get overlapped.
	//
	// The expected output blob we're interested might look like
	// this (for amd64):
	//
	// =-= stack layout for ABC:
	// 2: "p1" frameoff -8200 ...
	// 3: "s" frameoff -8200 ...
	// 4: "v2" frameoff -8200 ...
	// 5: "v3" frameoff -8200 ...
	// 6: "xp3" frameoff -8200 ...
	// 7: "xp4" frameoff -8200 ...
	// 8: "p2" frameoff -16400 ...
	// 9: "r" frameoff -16408 ...
	//
	tmpdir := t.TempDir()
	src := filepath.Join("testdata", "mergelocals", "integration.go")
	obj := filepath.Join(tmpdir, "p.a")
	out, err := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile",
		"-p=p", "-c", "1", "-o", obj, "-d=mergelocalstrace=2,mergelocals=1",
		src).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to compile: %v\n%s", err, out)
	}
	vars := make(map[string]string)
	lines := strings.Split(string(out), "\n")
	prolog := true
	varsAtFrameOffset := make(map[string]int)
	for _, line := range lines {
		if line == "=-= stack layout for ABC:" {
			prolog = false
			continue
		} else if prolog || line == "" {
			continue
		}
		fields := strings.Fields(line)
		wantFields := 9
		if len(fields) != wantFields {
			t.Logf(string(out))
			t.Fatalf("bad trace output line, wanted %d fields got %d: %s",
				wantFields, len(fields), line)
		}
		vname := fields[1]
		frameoff := fields[3]
		varsAtFrameOffset[frameoff] = varsAtFrameOffset[frameoff] + 1
		vars[vname] = frameoff
	}
	wantvnum := 8
	gotvnum := len(vars)
	if wantvnum != gotvnum {
		t.Logf(string(out))
		t.Fatalf("expected trace output on %d vars got %d\n", wantvnum, gotvnum)
	}

	// Expect at least one clump of at least 3.
	n3 := 0
	got := []int{}
	for _, v := range varsAtFrameOffset {
		if v > 2 {
			n3++
		}
		got = append(got, v)
	}
	sort.Ints(got)
	if n3 == 0 {
		t.Logf("%s\n", string(out))
		t.Fatalf("expected at least one clump of 3, got: %+v", got)
	}
}
