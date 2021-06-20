// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysis

import (
	"strings"
	"testing"
)

func TestValidate(t *testing.T) {
	var (
		run = func(p *Pass) (interface{}, error) {
			return nil, nil
		}
		dependsOnSelf = &Analyzer{
			Name: "dependsOnSelf",
			Doc:  "this analyzer depends on itself",
			Run:  run,
		}
		inCycleA = &Analyzer{
			Name: "inCycleA",
			Doc:  "this analyzer depends on inCycleB",
			Run:  run,
		}
		inCycleB = &Analyzer{
			Name: "inCycleB",
			Doc:  "this analyzer depends on inCycleA and notInCycleA",
			Run:  run,
		}
		pointsToCycle = &Analyzer{
			Name: "pointsToCycle",
			Doc:  "this analyzer depends on inCycleA",
			Run:  run,
		}
		notInCycleA = &Analyzer{
			Name: "notInCycleA",
			Doc:  "this analyzer depends on notInCycleB and notInCycleC",
			Run:  run,
		}
		notInCycleB = &Analyzer{
			Name: "notInCycleB",
			Doc:  "this analyzer depends on notInCycleC",
			Run:  run,
		}
		notInCycleC = &Analyzer{
			Name: "notInCycleC",
			Doc:  "this analyzer has no dependencies",
			Run:  run,
		}
	)

	dependsOnSelf.Requires = append(dependsOnSelf.Requires, dependsOnSelf)
	inCycleA.Requires = append(inCycleA.Requires, inCycleB)
	inCycleB.Requires = append(inCycleB.Requires, inCycleA, notInCycleA)
	pointsToCycle.Requires = append(pointsToCycle.Requires, inCycleA)
	notInCycleA.Requires = append(notInCycleA.Requires, notInCycleB, notInCycleC)
	notInCycleB.Requires = append(notInCycleB.Requires, notInCycleC)
	notInCycleC.Requires = []*Analyzer{}

	cases := []struct {
		analyzers        []*Analyzer
		wantErr          bool
		analyzersInCycle map[string]bool
	}{
		{
			[]*Analyzer{dependsOnSelf},
			true,
			map[string]bool{"dependsOnSelf": true},
		},
		{
			[]*Analyzer{inCycleA, inCycleB},
			true,
			map[string]bool{"inCycleA": true, "inCycleB": true},
		},
		{
			[]*Analyzer{pointsToCycle},
			true,
			map[string]bool{"inCycleA": true, "inCycleB": true},
		},
		{
			[]*Analyzer{notInCycleA},
			false,
			map[string]bool{},
		},
	}

	for _, c := range cases {
		got := Validate(c.analyzers)

		if !c.wantErr {
			if got == nil {
				continue
			}
			t.Errorf("got unexpected error while validating analyzers %v: %v", c.analyzers, got)
		}

		if got == nil {
			t.Errorf("expected error while validating analyzers %v, but got nil", c.analyzers)
		}

		err, ok := got.(*CycleInRequiresGraphError)
		if !ok {
			t.Errorf("want CycleInRequiresGraphError, got %T", err)
		}

		for a := range c.analyzersInCycle {
			if !err.AnalyzerNames[a] {
				t.Errorf("analyzer %s should be in cycle", a)
			}
		}
		for a := range err.AnalyzerNames {
			if !c.analyzersInCycle[a] {
				t.Errorf("analyzer %s should not be in cycle", a)
			}
		}
	}
}

func TestCycleInRequiresGraphErrorMessage(t *testing.T) {
	err := CycleInRequiresGraphError{}
	errMsg := err.Error()
	wantSubstring := "cycle detected"
	if !strings.Contains(errMsg, wantSubstring) {
		t.Errorf("error string %s does not contain expected substring %q", errMsg, wantSubstring)
	}
}

func TestValidateEmptyDoc(t *testing.T) {
	withoutDoc := &Analyzer{
		Name: "withoutDoc",
		Run: func(p *Pass) (interface{}, error) {
			return nil, nil
		},
	}
	err := Validate([]*Analyzer{withoutDoc})
	if err == nil || !strings.Contains(err.Error(), "is undocumented") {
		t.Errorf("got unexpected error while validating analyzers withoutDoc: %v", err)
	}
}

func TestValidateNoRun(t *testing.T) {
	withoutRun := &Analyzer{
		Name: "withoutRun",
		Doc:  "this analyzer has no Run",
	}
	err := Validate([]*Analyzer{withoutRun})
	if err == nil || !strings.Contains(err.Error(), "has nil Run") {
		t.Errorf("got unexpected error while validating analyzers withoutRun: %v", err)
	}
}
