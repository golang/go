// Copyright 2018 Google Inc. All Rights Reserved.
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

package profile

import (
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/google/pprof/internal/proftest"
)

var mappings = []*Mapping{
	{ID: 1, Start: 0x10000, Limit: 0x40000, File: "map0", HasFunctions: true, HasFilenames: true, HasLineNumbers: true, HasInlineFrames: true},
	{ID: 2, Start: 0x50000, Limit: 0x70000, File: "map1", HasFunctions: true, HasFilenames: true, HasLineNumbers: true, HasInlineFrames: true},
}

var functions = []*Function{
	{ID: 1, Name: "fun0", SystemName: "fun0", Filename: "file0"},
	{ID: 2, Name: "fun1", SystemName: "fun1", Filename: "file1"},
	{ID: 3, Name: "fun2", SystemName: "fun2", Filename: "file2"},
	{ID: 4, Name: "fun3", SystemName: "fun3", Filename: "file3"},
	{ID: 5, Name: "fun4", SystemName: "fun4", Filename: "file4"},
	{ID: 6, Name: "fun5", SystemName: "fun5", Filename: "file5"},
	{ID: 7, Name: "fun6", SystemName: "fun6", Filename: "file6"},
	{ID: 8, Name: "fun7", SystemName: "fun7", Filename: "file7"},
	{ID: 9, Name: "fun8", SystemName: "fun8", Filename: "file8"},
	{ID: 10, Name: "fun9", SystemName: "fun9", Filename: "file9"},
	{ID: 11, Name: "fun10", SystemName: "fun10", Filename: "file10"},
}

var noInlinesLocs = []*Location{
	{ID: 1, Mapping: mappings[0], Address: 0x1000, Line: []Line{{Function: functions[0], Line: 1}}},
	{ID: 2, Mapping: mappings[0], Address: 0x2000, Line: []Line{{Function: functions[1], Line: 1}}},
	{ID: 3, Mapping: mappings[0], Address: 0x3000, Line: []Line{{Function: functions[2], Line: 1}}},
	{ID: 4, Mapping: mappings[0], Address: 0x4000, Line: []Line{{Function: functions[3], Line: 1}}},
	{ID: 5, Mapping: mappings[0], Address: 0x5000, Line: []Line{{Function: functions[4], Line: 1}}},
	{ID: 6, Mapping: mappings[0], Address: 0x6000, Line: []Line{{Function: functions[5], Line: 1}}},
	{ID: 7, Mapping: mappings[0], Address: 0x7000, Line: []Line{{Function: functions[6], Line: 1}}},
	{ID: 8, Mapping: mappings[0], Address: 0x8000, Line: []Line{{Function: functions[7], Line: 1}}},
	{ID: 9, Mapping: mappings[0], Address: 0x9000, Line: []Line{{Function: functions[8], Line: 1}}},
	{ID: 10, Mapping: mappings[0], Address: 0x10000, Line: []Line{{Function: functions[9], Line: 1}}},
	{ID: 11, Mapping: mappings[1], Address: 0x11000, Line: []Line{{Function: functions[10], Line: 1}}},
}

var noInlinesProfile = &Profile{
	TimeNanos:     10000,
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType:    []*ValueType{{Type: "samples", Unit: "count"}},
	Mapping:       mappings,
	Function:      functions,
	Location:      noInlinesLocs,
	Sample: []*Sample{
		{Value: []int64{1}, Location: []*Location{noInlinesLocs[0], noInlinesLocs[1], noInlinesLocs[2], noInlinesLocs[3]}},
		{Value: []int64{2}, Location: []*Location{noInlinesLocs[4], noInlinesLocs[5], noInlinesLocs[1], noInlinesLocs[6]}},
		{Value: []int64{3}, Location: []*Location{noInlinesLocs[7], noInlinesLocs[8]}},
		{Value: []int64{4}, Location: []*Location{noInlinesLocs[9], noInlinesLocs[4], noInlinesLocs[10], noInlinesLocs[7]}},
	},
}

var allNoInlinesSampleFuncs = []string{
	"fun0 fun1 fun2 fun3: 1",
	"fun4 fun5 fun1 fun6: 2",
	"fun7 fun8: 3",
	"fun9 fun4 fun10 fun7: 4",
}

var inlinesLocs = []*Location{
	{ID: 1, Mapping: mappings[0], Address: 0x1000, Line: []Line{{Function: functions[0], Line: 1}, {Function: functions[1], Line: 1}}},
	{ID: 2, Mapping: mappings[0], Address: 0x2000, Line: []Line{{Function: functions[2], Line: 1}, {Function: functions[3], Line: 1}}},
	{ID: 3, Mapping: mappings[0], Address: 0x3000, Line: []Line{{Function: functions[4], Line: 1}, {Function: functions[5], Line: 1}, {Function: functions[6], Line: 1}}},
}

var inlinesProfile = &Profile{
	TimeNanos:     10000,
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType:    []*ValueType{{Type: "samples", Unit: "count"}},
	Mapping:       mappings,
	Function:      functions,
	Location:      inlinesLocs,
	Sample: []*Sample{
		{Value: []int64{1}, Location: []*Location{inlinesLocs[0], inlinesLocs[1]}},
		{Value: []int64{2}, Location: []*Location{inlinesLocs[2]}},
	},
}

var emptyLinesLocs = []*Location{
	{ID: 1, Mapping: mappings[0], Address: 0x1000, Line: []Line{{Function: functions[0], Line: 1}, {Function: functions[1], Line: 1}}},
	{ID: 2, Mapping: mappings[0], Address: 0x2000, Line: []Line{}},
	{ID: 3, Mapping: mappings[1], Address: 0x2000, Line: []Line{}},
}

var emptyLinesProfile = &Profile{
	TimeNanos:     10000,
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType:    []*ValueType{{Type: "samples", Unit: "count"}},
	Mapping:       mappings,
	Function:      functions,
	Location:      emptyLinesLocs,
	Sample: []*Sample{
		{Value: []int64{1}, Location: []*Location{emptyLinesLocs[0], emptyLinesLocs[1]}},
		{Value: []int64{2}, Location: []*Location{emptyLinesLocs[2]}},
		{Value: []int64{3}, Location: []*Location{}},
	},
}

func TestFilterSamplesByName(t *testing.T) {
	for _, tc := range []struct {
		// name is the name of the test case.
		name string
		// profile is the profile that gets filtered.
		profile *Profile
		// These are the inputs to FilterSamplesByName().
		focus, ignore, hide, show *regexp.Regexp
		// want{F,I,S,H}m are expected return values from FilterSamplesByName.
		wantFm, wantIm, wantSm, wantHm bool
		// wantSampleFuncs contains expected stack functions and sample value after
		// filtering, in the same order as in the profile. The format is as
		// returned by sampleFuncs function below, which is "callee caller: <num>".
		wantSampleFuncs []string
	}{
		// No Filters
		{
			name:            "empty filters keep all frames",
			profile:         noInlinesProfile,
			wantFm:          true,
			wantSampleFuncs: allNoInlinesSampleFuncs,
		},
		// Focus
		{
			name:    "focus with no matches",
			profile: noInlinesProfile,
			focus:   regexp.MustCompile("unknown"),
		},
		{
			name:    "focus matches function names",
			profile: noInlinesProfile,
			focus:   regexp.MustCompile("fun1"),
			wantFm:  true,
			wantSampleFuncs: []string{
				"fun0 fun1 fun2 fun3: 1",
				"fun4 fun5 fun1 fun6: 2",
				"fun9 fun4 fun10 fun7: 4",
			},
		},
		{
			name:    "focus matches file names",
			profile: noInlinesProfile,
			focus:   regexp.MustCompile("file1"),
			wantFm:  true,
			wantSampleFuncs: []string{
				"fun0 fun1 fun2 fun3: 1",
				"fun4 fun5 fun1 fun6: 2",
				"fun9 fun4 fun10 fun7: 4",
			},
		},
		{
			name:    "focus matches mapping names",
			profile: noInlinesProfile,
			focus:   regexp.MustCompile("map1"),
			wantFm:  true,
			wantSampleFuncs: []string{
				"fun9 fun4 fun10 fun7: 4",
			},
		},
		{
			name:    "focus matches inline functions",
			profile: inlinesProfile,
			focus:   regexp.MustCompile("fun5"),
			wantFm:  true,
			wantSampleFuncs: []string{
				"fun4 fun5 fun6: 2",
			},
		},
		// Ignore
		{
			name:            "ignore with no matches matches all samples",
			profile:         noInlinesProfile,
			ignore:          regexp.MustCompile("unknown"),
			wantFm:          true,
			wantSampleFuncs: allNoInlinesSampleFuncs,
		},
		{
			name:    "ignore matches function names",
			profile: noInlinesProfile,
			ignore:  regexp.MustCompile("fun1"),
			wantFm:  true,
			wantIm:  true,
			wantSampleFuncs: []string{
				"fun7 fun8: 3",
			},
		},
		{
			name:    "ignore matches file names",
			profile: noInlinesProfile,
			ignore:  regexp.MustCompile("file1"),
			wantFm:  true,
			wantIm:  true,
			wantSampleFuncs: []string{
				"fun7 fun8: 3",
			},
		},
		{
			name:    "ignore matches mapping names",
			profile: noInlinesProfile,
			ignore:  regexp.MustCompile("map1"),
			wantFm:  true,
			wantIm:  true,
			wantSampleFuncs: []string{
				"fun0 fun1 fun2 fun3: 1",
				"fun4 fun5 fun1 fun6: 2",
				"fun7 fun8: 3",
			},
		},
		{
			name:    "ignore matches inline functions",
			profile: inlinesProfile,
			ignore:  regexp.MustCompile("fun5"),
			wantFm:  true,
			wantIm:  true,
			wantSampleFuncs: []string{
				"fun0 fun1 fun2 fun3: 1",
			},
		},
		// Show
		{
			name:    "show with no matches",
			profile: noInlinesProfile,
			show:    regexp.MustCompile("unknown"),
			wantFm:  true,
		},
		{
			name:    "show matches function names",
			profile: noInlinesProfile,
			show:    regexp.MustCompile("fun1|fun2"),
			wantFm:  true,
			wantSm:  true,
			wantSampleFuncs: []string{
				"fun1 fun2: 1",
				"fun1: 2",
				"fun10: 4",
			},
		},
		{
			name:    "show matches file names",
			profile: noInlinesProfile,
			show:    regexp.MustCompile("file1|file3"),
			wantFm:  true,
			wantSm:  true,
			wantSampleFuncs: []string{
				"fun1 fun3: 1",
				"fun1: 2",
				"fun10: 4",
			},
		},
		{
			name:    "show matches mapping names",
			profile: noInlinesProfile,
			show:    regexp.MustCompile("map1"),
			wantFm:  true,
			wantSm:  true,
			wantSampleFuncs: []string{
				"fun10: 4",
			},
		},
		{
			name:    "show matches inline functions",
			profile: inlinesProfile,
			show:    regexp.MustCompile("fun[03]"),
			wantFm:  true,
			wantSm:  true,
			wantSampleFuncs: []string{
				"fun0 fun3: 1",
			},
		},
		{
			name:    "show keeps all lines when matching both mapping and function",
			profile: inlinesProfile,
			show:    regexp.MustCompile("map0|fun5"),
			wantFm:  true,
			wantSm:  true,
			wantSampleFuncs: []string{
				"fun0 fun1 fun2 fun3: 1",
				"fun4 fun5 fun6: 2",
			},
		},
		// Hide
		{
			name:            "hide with no matches",
			profile:         noInlinesProfile,
			hide:            regexp.MustCompile("unknown"),
			wantFm:          true,
			wantSampleFuncs: allNoInlinesSampleFuncs,
		},
		{
			name:    "hide matches function names",
			profile: noInlinesProfile,
			hide:    regexp.MustCompile("fun1|fun2"),
			wantFm:  true,
			wantHm:  true,
			wantSampleFuncs: []string{
				"fun0 fun3: 1",
				"fun4 fun5 fun6: 2",
				"fun7 fun8: 3",
				"fun9 fun4 fun7: 4",
			},
		},
		{
			name:    "hide matches file names",
			profile: noInlinesProfile,
			hide:    regexp.MustCompile("file1|file3"),
			wantFm:  true,
			wantHm:  true,
			wantSampleFuncs: []string{
				"fun0 fun2: 1",
				"fun4 fun5 fun6: 2",
				"fun7 fun8: 3",
				"fun9 fun4 fun7: 4",
			},
		},
		{
			name:    "hide matches mapping names",
			profile: noInlinesProfile,
			hide:    regexp.MustCompile("map1"),
			wantFm:  true,
			wantHm:  true,
			wantSampleFuncs: []string{
				"fun0 fun1 fun2 fun3: 1",
				"fun4 fun5 fun1 fun6: 2",
				"fun7 fun8: 3",
				"fun9 fun4 fun7: 4",
			},
		},
		{
			name:    "hide matches inline functions",
			profile: inlinesProfile,
			hide:    regexp.MustCompile("fun[125]"),
			wantFm:  true,
			wantHm:  true,
			wantSampleFuncs: []string{
				"fun0 fun3: 1",
				"fun4 fun6: 2",
			},
		},
		{
			name:    "hide drops all lines when matching both mapping and function",
			profile: inlinesProfile,
			hide:    regexp.MustCompile("map0|fun5"),
			wantFm:  true,
			wantHm:  true,
		},
		// Compound filters
		{
			name:    "hides a stack matched by both focus and ignore",
			profile: noInlinesProfile,
			focus:   regexp.MustCompile("fun1|fun7"),
			ignore:  regexp.MustCompile("fun1"),
			wantFm:  true,
			wantIm:  true,
			wantSampleFuncs: []string{
				"fun7 fun8: 3",
			},
		},
		{
			name:    "hides a function if both show and hide match it",
			profile: noInlinesProfile,
			show:    regexp.MustCompile("fun1"),
			hide:    regexp.MustCompile("fun10"),
			wantFm:  true,
			wantSm:  true,
			wantHm:  true,
			wantSampleFuncs: []string{
				"fun1: 1",
				"fun1: 2",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.profile.Copy()
			fm, im, hm, sm := p.FilterSamplesByName(tc.focus, tc.ignore, tc.hide, tc.show)

			type match struct{ fm, im, hm, sm bool }
			if got, want := (match{fm: fm, im: im, hm: hm, sm: sm}), (match{fm: tc.wantFm, im: tc.wantIm, hm: tc.wantHm, sm: tc.wantSm}); got != want {
				t.Errorf("match got %+v want %+v", got, want)
			}

			if got, want := strings.Join(sampleFuncs(p), "\n")+"\n", strings.Join(tc.wantSampleFuncs, "\n")+"\n"; got != want {
				diff, err := proftest.Diff([]byte(want), []byte(got))
				if err != nil {
					t.Fatalf("failed to get diff: %v", err)
				}
				t.Errorf("FilterSamplesByName: got diff(want->got):\n%s", diff)
			}
		})
	}
}

func TestShowFrom(t *testing.T) {
	for _, tc := range []struct {
		name     string
		profile  *Profile
		showFrom *regexp.Regexp
		// wantMatch is the expected return value.
		wantMatch bool
		// wantSampleFuncs contains expected stack functions and sample value after
		// filtering, in the same order as in the profile. The format is as
		// returned by sampleFuncs function below, which is "callee caller: <num>".
		wantSampleFuncs []string
	}{
		{
			name:            "nil showFrom keeps all frames",
			profile:         noInlinesProfile,
			wantMatch:       false,
			wantSampleFuncs: allNoInlinesSampleFuncs,
		},
		{
			name:      "showFrom with no matches drops all samples",
			profile:   noInlinesProfile,
			showFrom:  regexp.MustCompile("unknown"),
			wantMatch: false,
		},
		{
			name:      "showFrom matches function names",
			profile:   noInlinesProfile,
			showFrom:  regexp.MustCompile("fun1"),
			wantMatch: true,
			wantSampleFuncs: []string{
				"fun0 fun1: 1",
				"fun4 fun5 fun1: 2",
				"fun9 fun4 fun10: 4",
			},
		},
		{
			name:      "showFrom matches file names",
			profile:   noInlinesProfile,
			showFrom:  regexp.MustCompile("file1"),
			wantMatch: true,
			wantSampleFuncs: []string{
				"fun0 fun1: 1",
				"fun4 fun5 fun1: 2",
				"fun9 fun4 fun10: 4",
			},
		},
		{
			name:      "showFrom matches mapping names",
			profile:   noInlinesProfile,
			showFrom:  regexp.MustCompile("map1"),
			wantMatch: true,
			wantSampleFuncs: []string{
				"fun9 fun4 fun10: 4",
			},
		},
		{
			name:      "showFrom drops frames above highest of multiple matches",
			profile:   noInlinesProfile,
			showFrom:  regexp.MustCompile("fun[12]"),
			wantMatch: true,
			wantSampleFuncs: []string{
				"fun0 fun1 fun2: 1",
				"fun4 fun5 fun1: 2",
				"fun9 fun4 fun10: 4",
			},
		},
		{
			name:      "showFrom matches inline functions",
			profile:   inlinesProfile,
			showFrom:  regexp.MustCompile("fun0|fun5"),
			wantMatch: true,
			wantSampleFuncs: []string{
				"fun0: 1",
				"fun4 fun5: 2",
			},
		},
		{
			name:      "showFrom drops frames above highest of multiple inline matches",
			profile:   inlinesProfile,
			showFrom:  regexp.MustCompile("fun[1245]"),
			wantMatch: true,
			wantSampleFuncs: []string{
				"fun0 fun1 fun2: 1",
				"fun4 fun5: 2",
			},
		},
		{
			name:      "showFrom keeps all lines when matching mapping and function",
			profile:   inlinesProfile,
			showFrom:  regexp.MustCompile("map0|fun5"),
			wantMatch: true,
			wantSampleFuncs: []string{
				"fun0 fun1 fun2 fun3: 1",
				"fun4 fun5 fun6: 2",
			},
		},
		{
			name:      "showFrom matches location with empty lines",
			profile:   emptyLinesProfile,
			showFrom:  regexp.MustCompile("map1"),
			wantMatch: true,
			wantSampleFuncs: []string{
				": 2",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.profile.Copy()

			if gotMatch := p.ShowFrom(tc.showFrom); gotMatch != tc.wantMatch {
				t.Errorf("match got %+v, want %+v", gotMatch, tc.wantMatch)
			}

			if got, want := strings.Join(sampleFuncs(p), "\n")+"\n", strings.Join(tc.wantSampleFuncs, "\n")+"\n"; got != want {
				diff, err := proftest.Diff([]byte(want), []byte(got))
				if err != nil {
					t.Fatalf("failed to get diff: %v", err)
				}
				t.Errorf("profile samples got diff(want->got):\n%s", diff)
			}
		})
	}
}

// sampleFuncs returns a slice of strings where each string represents one
// profile sample in the format "<fun1> <fun2> <fun3>: <value>". This allows
// the expected values for test cases to be specifed in human-readable strings.
func sampleFuncs(p *Profile) []string {
	var ret []string
	for _, s := range p.Sample {
		var funcs []string
		for _, loc := range s.Location {
			for _, line := range loc.Line {
				funcs = append(funcs, line.Function.Name)
			}
		}
		ret = append(ret, fmt.Sprintf("%s: %d", strings.Join(funcs, " "), s.Value[0]))
	}
	return ret
}

func TestTagFilter(t *testing.T) {
	// Perform several forms of tag filtering on the test profile.

	type filterTestcase struct {
		include, exclude *regexp.Regexp
		im, em           bool
		count            int
	}

	countTags := func(p *Profile) map[string]bool {
		tags := make(map[string]bool)

		for _, s := range p.Sample {
			for l := range s.Label {
				tags[l] = true
			}
			for l := range s.NumLabel {
				tags[l] = true
			}
		}
		return tags
	}

	for tx, tc := range []filterTestcase{
		{nil, nil, true, false, 3},
		{regexp.MustCompile("notfound"), nil, false, false, 0},
		{regexp.MustCompile("key1"), nil, true, false, 1},
		{nil, regexp.MustCompile("key[12]"), true, true, 1},
	} {
		prof := testProfile1.Copy()
		gim, gem := prof.FilterTagsByName(tc.include, tc.exclude)
		if gim != tc.im {
			t.Errorf("Filter #%d, got include match=%v, want %v", tx, gim, tc.im)
		}
		if gem != tc.em {
			t.Errorf("Filter #%d, got exclude match=%v, want %v", tx, gem, tc.em)
		}
		if tags := countTags(prof); len(tags) != tc.count {
			t.Errorf("Filter #%d, got %d tags[%v], want %d", tx, len(tags), tags, tc.count)
		}
	}
}
