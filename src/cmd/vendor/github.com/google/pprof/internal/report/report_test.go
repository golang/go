// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package report

import (
	"bytes"
	"io/ioutil"
	"regexp"
	"runtime"
	"testing"

	"github.com/google/pprof/internal/binutils"
	"github.com/google/pprof/internal/graph"
	"github.com/google/pprof/internal/proftest"
	"github.com/google/pprof/profile"
)

type testcase struct {
	rpt  *Report
	want string
}

func TestSource(t *testing.T) {
	const path = "testdata/"

	sampleValue1 := func(v []int64) int64 {
		return v[1]
	}

	for _, tc := range []testcase{
		{
			rpt: New(
				testProfile.Copy(),
				&Options{
					OutputFormat: List,
					Symbol:       regexp.MustCompile(`.`),
					TrimPath:     "/some/path",

					SampleValue: sampleValue1,
					SampleUnit:  testProfile.SampleType[1].Unit,
				},
			),
			want: path + "source.rpt",
		},
		{
			rpt: New(
				testProfile.Copy(),
				&Options{
					OutputFormat: Dot,
					CallTree:     true,
					Symbol:       regexp.MustCompile(`.`),
					TrimPath:     "/some/path",

					SampleValue: sampleValue1,
					SampleUnit:  testProfile.SampleType[1].Unit,
				},
			),
			want: path + "source.dot",
		},
	} {
		var b bytes.Buffer
		if err := Generate(&b, tc.rpt, &binutils.Binutils{}); err != nil {
			t.Fatalf("%s: %v", tc.want, err)
		}

		gold, err := ioutil.ReadFile(tc.want)
		if err != nil {
			t.Fatalf("%s: %v", tc.want, err)
		}
		if runtime.GOOS == "windows" {
			gold = bytes.Replace(gold, []byte("testdata/"), []byte("testdata\\"), -1)
		}
		if string(b.String()) != string(gold) {
			d, err := proftest.Diff(gold, b.Bytes())
			if err != nil {
				t.Fatalf("%s: %v", "source", err)
			}
			t.Error("source" + "\n" + string(d) + "\n" + "gold:\n" + tc.want)
		}
	}
}

var testM = []*profile.Mapping{
	{
		ID:              1,
		HasFunctions:    true,
		HasFilenames:    true,
		HasLineNumbers:  true,
		HasInlineFrames: true,
	},
}

var testF = []*profile.Function{
	{
		ID:       1,
		Name:     "main",
		Filename: "testdata/source1",
	},
	{
		ID:       2,
		Name:     "foo",
		Filename: "testdata/source1",
	},
	{
		ID:       3,
		Name:     "bar",
		Filename: "testdata/source1",
	},
	{
		ID:       4,
		Name:     "tee",
		Filename: "/some/path/testdata/source2",
	},
}

var testL = []*profile.Location{
	{
		ID:      1,
		Mapping: testM[0],
		Line: []profile.Line{
			{
				Function: testF[0],
				Line:     2,
			},
		},
	},
	{
		ID:      2,
		Mapping: testM[0],
		Line: []profile.Line{
			{
				Function: testF[1],
				Line:     4,
			},
		},
	},
	{
		ID:      3,
		Mapping: testM[0],
		Line: []profile.Line{
			{
				Function: testF[2],
				Line:     10,
			},
		},
	},
	{
		ID:      4,
		Mapping: testM[0],
		Line: []profile.Line{
			{
				Function: testF[3],
				Line:     2,
			},
		},
	},
	{
		ID:      5,
		Mapping: testM[0],
		Line: []profile.Line{
			{
				Function: testF[3],
				Line:     8,
			},
		},
	},
}

var testProfile = &profile.Profile{
	PeriodType:    &profile.ValueType{Type: "cpu", Unit: "millisecond"},
	Period:        10,
	DurationNanos: 10e9,
	SampleType: []*profile.ValueType{
		{Type: "samples", Unit: "count"},
		{Type: "cpu", Unit: "cycles"},
	},
	Sample: []*profile.Sample{
		{
			Location: []*profile.Location{testL[0]},
			Value:    []int64{1, 1},
		},
		{
			Location: []*profile.Location{testL[2], testL[1], testL[0]},
			Value:    []int64{1, 10},
		},
		{
			Location: []*profile.Location{testL[4], testL[2], testL[0]},
			Value:    []int64{1, 100},
		},
		{
			Location: []*profile.Location{testL[3], testL[0]},
			Value:    []int64{1, 1000},
		},
		{
			Location: []*profile.Location{testL[4], testL[3], testL[0]},
			Value:    []int64{1, 10000},
		},
	},
	Location: testL,
	Function: testF,
	Mapping:  testM,
}

func TestDisambiguation(t *testing.T) {
	parent1 := &graph.Node{Info: graph.NodeInfo{Name: "parent1"}}
	parent2 := &graph.Node{Info: graph.NodeInfo{Name: "parent2"}}
	child1 := &graph.Node{Info: graph.NodeInfo{Name: "child"}, Function: parent1}
	child2 := &graph.Node{Info: graph.NodeInfo{Name: "child"}, Function: parent2}
	child3 := &graph.Node{Info: graph.NodeInfo{Name: "child"}, Function: parent1}
	sibling := &graph.Node{Info: graph.NodeInfo{Name: "sibling"}, Function: parent1}

	n := []*graph.Node{parent1, parent2, child1, child2, child3, sibling}

	wanted := map[*graph.Node]string{
		parent1: "parent1",
		parent2: "parent2",
		child1:  "child [1/2]",
		child2:  "child [2/2]",
		child3:  "child [1/2]",
		sibling: "sibling",
	}

	g := &graph.Graph{Nodes: n}

	names := getDisambiguatedNames(g)

	for node, want := range wanted {
		if got := names[node]; got != want {
			t.Errorf("name %s, got %s, want %s", node.Info.Name, got, want)
		}
	}
}

func TestFunctionMap(t *testing.T) {

	fm := make(functionMap)
	nodes := []graph.NodeInfo{
		{Name: "fun1"},
		{Name: "fun2", File: "filename"},
		{Name: "fun1"},
		{Name: "fun2", File: "filename2"},
	}

	want := []profile.Function{
		{ID: 1, Name: "fun1"},
		{ID: 2, Name: "fun2", Filename: "filename"},
		{ID: 1, Name: "fun1"},
		{ID: 3, Name: "fun2", Filename: "filename2"},
	}

	for i, tc := range nodes {
		if got, want := fm.FindOrAdd(tc), want[i]; *got != want {
			t.Errorf("%d: want %v, got %v", i, want, got)
		}
	}
}

func TestLegendActiveFilters(t *testing.T) {
	activeFilterInput := []string{
		"focus=123|456|789|101112|131415|161718|192021|222324|252627|282930|313233|343536|363738|acbdefghijklmnop",
		"show=short filter",
	}
	expectedLegendActiveFilter := []string{
		"Active filters:",
		"   focus=123|456|789|101112|131415|161718|192021|222324|252627|282930|313233|343536…",
		"   show=short filter",
	}
	legendActiveFilter := legendActiveFilters(activeFilterInput)
	if len(legendActiveFilter) != len(expectedLegendActiveFilter) {
		t.Errorf("wanted length %v got length %v", len(expectedLegendActiveFilter), len(legendActiveFilter))
	}
	for i := range legendActiveFilter {
		if legendActiveFilter[i] != expectedLegendActiveFilter[i] {
			t.Errorf("%d: want \"%v\", got \"%v\"", i, expectedLegendActiveFilter[i], legendActiveFilter[i])
		}
	}
}

func TestComputeTotal(t *testing.T) {
	p1 := testProfile.Copy()
	p1.Sample = []*profile.Sample{
		{
			Location: []*profile.Location{testL[0]},
			Value:    []int64{1, 1},
		},
		{
			Location: []*profile.Location{testL[2], testL[1], testL[0]},
			Value:    []int64{1, 10},
		},
		{
			Location: []*profile.Location{testL[4], testL[2], testL[0]},
			Value:    []int64{1, 100},
		},
	}

	p2 := testProfile.Copy()
	p2.Sample = []*profile.Sample{
		{
			Location: []*profile.Location{testL[0]},
			Value:    []int64{1, 1},
		},
		{
			Location: []*profile.Location{testL[2], testL[1], testL[0]},
			Value:    []int64{1, -10},
		},
		{
			Location: []*profile.Location{testL[4], testL[2], testL[0]},
			Value:    []int64{1, 100},
		},
	}

	p3 := testProfile.Copy()
	p3.Sample = []*profile.Sample{
		{
			Location: []*profile.Location{testL[0]},
			Value:    []int64{10000, 1},
		},
		{
			Location: []*profile.Location{testL[2], testL[1], testL[0]},
			Value:    []int64{-10, 3},
			Label:    map[string][]string{"pprof::base": {"true"}},
		},
		{
			Location: []*profile.Location{testL[2], testL[1], testL[0]},
			Value:    []int64{1000, -10},
		},
		{
			Location: []*profile.Location{testL[2], testL[1], testL[0]},
			Value:    []int64{-9000, 3},
			Label:    map[string][]string{"pprof::base": {"true"}},
		},
		{
			Location: []*profile.Location{testL[2], testL[1], testL[0]},
			Value:    []int64{-1, 3},
			Label:    map[string][]string{"pprof::base": {"true"}},
		},
		{
			Location: []*profile.Location{testL[4], testL[2], testL[0]},
			Value:    []int64{100, 100},
		},
		{
			Location: []*profile.Location{testL[2], testL[1], testL[0]},
			Value:    []int64{100, 3},
			Label:    map[string][]string{"pprof::base": {"true"}},
		},
	}

	testcases := []struct {
		desc           string
		prof           *profile.Profile
		value, meanDiv func(v []int64) int64
		wantTotal      int64
	}{
		{
			desc: "no diff base, all positive values, index 1",
			prof: p1,
			value: func(v []int64) int64 {
				return v[0]
			},
			wantTotal: 3,
		},
		{
			desc: "no diff base, all positive values, index 2",
			prof: p1,
			value: func(v []int64) int64 {
				return v[1]
			},
			wantTotal: 111,
		},
		{
			desc: "no diff base, some negative values",
			prof: p2,
			value: func(v []int64) int64 {
				return v[1]
			},
			wantTotal: 111,
		},
		{
			desc: "diff base, some negative values",
			prof: p3,
			value: func(v []int64) int64 {
				return v[0]
			},
			wantTotal: 9111,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			if gotTotal := computeTotal(tc.prof, tc.value, tc.meanDiv); gotTotal != tc.wantTotal {
				t.Errorf("got total %d, want %v", gotTotal, tc.wantTotal)
			}
		})
	}
}
