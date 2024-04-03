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
	"slices"
	"sort"
	"strings"
	"testing"
)

func TestMergeLocalState(t *testing.T) {
	mkiv := func(name string) *ir.Name {
		i32 := types.Types[types.TINT32]
		s := typecheck.Lookup(name)
		v := ir.NewNameAt(src.NoXPos, s, i32)
		return v
	}
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
				v1: []int{0, 1, 2},
				v2: []int{0, 1, 2},
				v3: []int{0, 1, 2},
			},
			experr: false,
		},
		{
			// invalid mls.v slot -1
			vars: []*ir.Name{v1, v2, v3},
			partition: map[*ir.Name][]int{
				v1: []int{-1, 0},
				v2: []int{0, 1, 2},
				v3: []int{0, 1, 2},
			},
			experr: true,
		},
		{
			// duplicate var in v
			vars: []*ir.Name{v1, v2, v2},
			partition: map[*ir.Name][]int{
				v1: []int{0, 1, 2},
				v2: []int{0, 1, 2},
				v3: []int{0, 1, 2},
			},
			experr: true,
		},
		{
			// single element in partition
			vars: []*ir.Name{v1, v2, v3},
			partition: map[*ir.Name][]int{
				v1: []int{0},
				v2: []int{0, 1, 2},
				v3: []int{0, 1, 2},
			},
			experr: true,
		},
		{
			// missing element 2
			vars: []*ir.Name{v1, v2, v3},
			partition: map[*ir.Name][]int{
				v1: []int{0, 1},
				v2: []int{0, 1},
				v3: []int{0, 1},
			},
			experr: true,
		},
		{
			// partitions disagree for v1 vs v2
			vars: []*ir.Name{v1, v2, v3},
			partition: map[*ir.Name][]int{
				v1: []int{0, 1, 2},
				v2: []int{1, 0, 2},
				v3: []int{0, 1, 2},
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
	// tests just verifies that there is one clump of 3 vars that
	// get overlapped, then another clump of 2 that share the same
	// frame offset.
	//
	// The expected output blob we're interested in looks like this:
	//
	// =-= stack layout for ABC:
	//  2: "p1" frameoff -8200 used=true
	//  3: "xp3" frameoff -8200 used=true
	//  4: "xp4" frameoff -8200 used=true
	//  5: "p2" frameoff -16400 used=true
	//  6: "s" frameoff -24592 used=true
	//  7: "v1" frameoff -32792 used=true
	//  8: "v3" frameoff -32792 used=true
	//  9: "v2" frameoff -40992 used=true
	//
	tmpdir := t.TempDir()
	src := filepath.Join("testdata", "mergelocals", "integration.go")
	obj := filepath.Join(tmpdir, "p.a")
	out, err := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-p=p", "-c", "1", "-o", obj, "-d=mergelocalstrace=2,mergelocals=1", src).CombinedOutput()
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
		if len(fields) != 5 {
			t.Fatalf("bad trace output line: %s", line)
		}
		vname := fields[1]
		frameoff := fields[3]
		varsAtFrameOffset[frameoff] = varsAtFrameOffset[frameoff] + 1
		vars[vname] = frameoff
	}
	wantvnum := 8
	gotvnum := len(vars)
	if wantvnum != gotvnum {
		t.Fatalf("expected trace output on %d vars got %d\n", wantvnum, gotvnum)
	}

	// We expect one clump of 3, another clump of 2, and the rest singletons.
	expected := []int{1, 1, 1, 2, 3}
	got := []int{}
	for _, v := range varsAtFrameOffset {
		got = append(got, v)
	}
	sort.Ints(got)
	if !slices.Equal(got, expected) {
		t.Fatalf("expected variable clumps %+v not equal to what we got: %+v", expected, got)
	}
}
