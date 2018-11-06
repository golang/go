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

package profile

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/google/pprof/internal/proftest"
)

func TestParse(t *testing.T) {
	const path = "testdata/"

	for _, source := range []string{
		"go.crc32.cpu",
		"go.godoc.thread",
		"gobench.cpu",
		"gobench.heap",
		"cppbench.cpu",
		"cppbench.heap",
		"cppbench.contention",
		"cppbench.growth",
		"cppbench.thread",
		"cppbench.thread.all",
		"cppbench.thread.none",
		"java.cpu",
		"java.heap",
		"java.contention",
	} {
		inbytes, err := ioutil.ReadFile(filepath.Join(path, source))
		if err != nil {
			t.Fatal(err)
		}
		p, err := Parse(bytes.NewBuffer(inbytes))
		if err != nil {
			t.Fatalf("%s: %s", source, err)
		}

		js := p.String()
		goldFilename := path + source + ".string"
		gold, err := ioutil.ReadFile(goldFilename)
		if err != nil {
			t.Fatalf("%s: %v", source, err)
		}

		if js != string(gold) {
			t.Errorf("diff %s %s", source, goldFilename)
			d, err := proftest.Diff(gold, []byte(js))
			if err != nil {
				t.Fatalf("%s: %v", source, err)
			}
			t.Error(source + "\n" + string(d) + "\n" + "new profile at:\n" + leaveTempfile([]byte(js)))
		}

		// Reencode and decode.
		var bw bytes.Buffer
		if err := p.Write(&bw); err != nil {
			t.Fatalf("%s: %v", source, err)
		}
		if p, err = Parse(&bw); err != nil {
			t.Fatalf("%s: %v", source, err)
		}
		js2 := p.String()
		if js2 != string(gold) {
			d, err := proftest.Diff(gold, []byte(js2))
			if err != nil {
				t.Fatalf("%s: %v", source, err)
			}
			t.Error(source + "\n" + string(d) + "\n" + "gold:\n" + goldFilename +
				"\nnew profile at:\n" + leaveTempfile([]byte(js)))
		}
	}
}

func TestParseError(t *testing.T) {
	testcases := []string{
		"",
		"garbage text",
		"\x1f\x8b", // truncated gzip header
		"\x1f\x8b\x08\x08\xbe\xe9\x20\x58\x00\x03\x65\x6d\x70\x74\x79\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", // empty gzipped file
	}

	for i, input := range testcases {
		_, err := Parse(strings.NewReader(input))
		if err == nil {
			t.Errorf("got nil, want error for input #%d", i)
		}
	}
}

func TestParseConcatentated(t *testing.T) {
	prof := testProfile1.Copy()
	// Write the profile twice to buffer to create concatented profile.
	var buf bytes.Buffer
	prof.Write(&buf)
	prof.Write(&buf)
	_, err := Parse(&buf)
	if err == nil {
		t.Fatalf("got nil, want error")
	}
	if got, want := err.Error(), "parsing profile: concatenated profiles detected"; want != got {
		t.Fatalf("got error %q, want error %q", got, want)
	}
}

func TestCheckValid(t *testing.T) {
	const path = "testdata/java.cpu"

	inbytes, err := ioutil.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read profile file %q: %v", path, err)
	}
	p, err := Parse(bytes.NewBuffer(inbytes))
	if err != nil {
		t.Fatalf("failed to parse profile %q: %s", path, err)
	}

	for _, tc := range []struct {
		mutateFn func(*Profile)
		wantErr  string
	}{
		{
			mutateFn: func(p *Profile) { p.SampleType = nil },
			wantErr:  "missing sample type information",
		},
		{
			mutateFn: func(p *Profile) { p.Sample[0] = nil },
			wantErr:  "profile has nil sample",
		},
		{
			mutateFn: func(p *Profile) { p.Sample[0].Value = append(p.Sample[0].Value, 0) },
			wantErr:  "sample has 3 values vs. 2 types",
		},
		{
			mutateFn: func(p *Profile) { p.Sample[0].Location[0] = nil },
			wantErr:  "sample has nil location",
		},
		{
			mutateFn: func(p *Profile) { p.Location[0] = nil },
			wantErr:  "profile has nil location",
		},
		{
			mutateFn: func(p *Profile) { p.Mapping = append(p.Mapping, nil) },
			wantErr:  "profile has nil mapping",
		},
		{
			mutateFn: func(p *Profile) { p.Function[0] = nil },
			wantErr:  "profile has nil function",
		},
	} {
		t.Run(tc.wantErr, func(t *testing.T) {
			p := p.Copy()
			tc.mutateFn(p)
			if err := p.CheckValid(); err == nil {
				t.Errorf("CheckValid(): got no error, want error %q", tc.wantErr)
			} else if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("CheckValid(): got error %v, want error %q", err, tc.wantErr)
			}
		})
	}
}

// leaveTempfile leaves |b| in a temporary file on disk and returns the
// temp filename. This is useful to recover a profile when the test
// fails.
func leaveTempfile(b []byte) string {
	f1, err := ioutil.TempFile("", "profile_test")
	if err != nil {
		panic(err)
	}
	if _, err := f1.Write(b); err != nil {
		panic(err)
	}
	return f1.Name()
}

const mainBinary = "/bin/main"

var cpuM = []*Mapping{
	{
		ID:              1,
		Start:           0x10000,
		Limit:           0x40000,
		File:            mainBinary,
		HasFunctions:    true,
		HasFilenames:    true,
		HasLineNumbers:  true,
		HasInlineFrames: true,
	},
	{
		ID:              2,
		Start:           0x1000,
		Limit:           0x4000,
		File:            "/lib/lib.so",
		HasFunctions:    true,
		HasFilenames:    true,
		HasLineNumbers:  true,
		HasInlineFrames: true,
	},
	{
		ID:              3,
		Start:           0x4000,
		Limit:           0x5000,
		File:            "/lib/lib2_c.so.6",
		HasFunctions:    true,
		HasFilenames:    true,
		HasLineNumbers:  true,
		HasInlineFrames: true,
	},
	{
		ID:              4,
		Start:           0x5000,
		Limit:           0x9000,
		File:            "/lib/lib.so_6 (deleted)",
		HasFunctions:    true,
		HasFilenames:    true,
		HasLineNumbers:  true,
		HasInlineFrames: true,
	},
}

var cpuF = []*Function{
	{ID: 1, Name: "main", SystemName: "main", Filename: "main.c"},
	{ID: 2, Name: "foo", SystemName: "foo", Filename: "foo.c"},
	{ID: 3, Name: "foo_caller", SystemName: "foo_caller", Filename: "foo.c"},
}

var cpuL = []*Location{
	{
		ID:      1000,
		Mapping: cpuM[1],
		Address: 0x1000,
		Line: []Line{
			{Function: cpuF[0], Line: 1},
		},
	},
	{
		ID:      2000,
		Mapping: cpuM[0],
		Address: 0x2000,
		Line: []Line{
			{Function: cpuF[1], Line: 2},
			{Function: cpuF[2], Line: 1},
		},
	},
	{
		ID:      3000,
		Mapping: cpuM[0],
		Address: 0x3000,
		Line: []Line{
			{Function: cpuF[1], Line: 2},
			{Function: cpuF[2], Line: 1},
		},
	},
	{
		ID:      3001,
		Mapping: cpuM[0],
		Address: 0x3001,
		Line: []Line{
			{Function: cpuF[2], Line: 2},
		},
	},
	{
		ID:      3002,
		Mapping: cpuM[0],
		Address: 0x3002,
		Line: []Line{
			{Function: cpuF[2], Line: 3},
		},
	},
}

var testProfile1 = &Profile{
	TimeNanos:     10000,
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType: []*ValueType{
		{Type: "samples", Unit: "count"},
		{Type: "cpu", Unit: "milliseconds"},
	},
	Sample: []*Sample{
		{
			Location: []*Location{cpuL[0]},
			Value:    []int64{1000, 1000},
			Label: map[string][]string{
				"key1": {"tag1"},
				"key2": {"tag1"},
			},
		},
		{
			Location: []*Location{cpuL[1], cpuL[0]},
			Value:    []int64{100, 100},
			Label: map[string][]string{
				"key1": {"tag2"},
				"key3": {"tag2"},
			},
		},
		{
			Location: []*Location{cpuL[2], cpuL[0]},
			Value:    []int64{10, 10},
			Label: map[string][]string{
				"key1": {"tag3"},
				"key2": {"tag2"},
			},
		},
		{
			Location: []*Location{cpuL[3], cpuL[0]},
			Value:    []int64{10000, 10000},
			Label: map[string][]string{
				"key1": {"tag4"},
				"key2": {"tag1"},
			},
		},
		{
			Location: []*Location{cpuL[4], cpuL[0]},
			Value:    []int64{1, 1},
			Label: map[string][]string{
				"key1": {"tag4"},
				"key2": {"tag1"},
			},
		},
	},
	Location: cpuL,
	Function: cpuF,
	Mapping:  cpuM,
}

var testProfile1NoMapping = &Profile{
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType: []*ValueType{
		{Type: "samples", Unit: "count"},
		{Type: "cpu", Unit: "milliseconds"},
	},
	Sample: []*Sample{
		{
			Location: []*Location{cpuL[0]},
			Value:    []int64{1000, 1000},
			Label: map[string][]string{
				"key1": {"tag1"},
				"key2": {"tag1"},
			},
		},
		{
			Location: []*Location{cpuL[1], cpuL[0]},
			Value:    []int64{100, 100},
			Label: map[string][]string{
				"key1": {"tag2"},
				"key3": {"tag2"},
			},
		},
		{
			Location: []*Location{cpuL[2], cpuL[0]},
			Value:    []int64{10, 10},
			Label: map[string][]string{
				"key1": {"tag3"},
				"key2": {"tag2"},
			},
		},
		{
			Location: []*Location{cpuL[3], cpuL[0]},
			Value:    []int64{10000, 10000},
			Label: map[string][]string{
				"key1": {"tag4"},
				"key2": {"tag1"},
			},
		},
		{
			Location: []*Location{cpuL[4], cpuL[0]},
			Value:    []int64{1, 1},
			Label: map[string][]string{
				"key1": {"tag4"},
				"key2": {"tag1"},
			},
		},
	},
	Location: cpuL,
	Function: cpuF,
}

var testProfile2 = &Profile{
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType: []*ValueType{
		{Type: "samples", Unit: "count"},
		{Type: "cpu", Unit: "milliseconds"},
	},
	Sample: []*Sample{
		{
			Location: []*Location{cpuL[0]},
			Value:    []int64{70, 1000},
			Label: map[string][]string{
				"key1": {"tag1"},
				"key2": {"tag1"},
			},
		},
		{
			Location: []*Location{cpuL[1], cpuL[0]},
			Value:    []int64{60, 100},
			Label: map[string][]string{
				"key1": {"tag2"},
				"key3": {"tag2"},
			},
		},
		{
			Location: []*Location{cpuL[2], cpuL[0]},
			Value:    []int64{50, 10},
			Label: map[string][]string{
				"key1": {"tag3"},
				"key2": {"tag2"},
			},
		},
		{
			Location: []*Location{cpuL[3], cpuL[0]},
			Value:    []int64{40, 10000},
			Label: map[string][]string{
				"key1": {"tag4"},
				"key2": {"tag1"},
			},
		},
		{
			Location: []*Location{cpuL[4], cpuL[0]},
			Value:    []int64{1, 1},
			Label: map[string][]string{
				"key1": {"tag4"},
				"key2": {"tag1"},
			},
		},
	},
	Location: cpuL,
	Function: cpuF,
	Mapping:  cpuM,
}

var testProfile3 = &Profile{
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType: []*ValueType{
		{Type: "samples", Unit: "count"},
	},
	Sample: []*Sample{
		{
			Location: []*Location{cpuL[0]},
			Value:    []int64{1000},
			Label: map[string][]string{
				"key1": {"tag1"},
				"key2": {"tag1"},
			},
		},
	},
	Location: cpuL,
	Function: cpuF,
	Mapping:  cpuM,
}

var testProfile4 = &Profile{
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType: []*ValueType{
		{Type: "samples", Unit: "count"},
	},
	Sample: []*Sample{
		{
			Location: []*Location{cpuL[0]},
			Value:    []int64{1000},
			NumLabel: map[string][]int64{
				"key1": {10},
				"key2": {30},
			},
			NumUnit: map[string][]string{
				"key1": {"bytes"},
				"key2": {"bytes"},
			},
		},
	},
	Location: cpuL,
	Function: cpuF,
	Mapping:  cpuM,
}

var testProfile5 = &Profile{
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType: []*ValueType{
		{Type: "samples", Unit: "count"},
	},
	Sample: []*Sample{
		{
			Location: []*Location{cpuL[0]},
			Value:    []int64{1000},
			NumLabel: map[string][]int64{
				"key1": {10},
				"key2": {30},
			},
			NumUnit: map[string][]string{
				"key1": {"bytes"},
				"key2": {"bytes"},
			},
		},
		{
			Location: []*Location{cpuL[0]},
			Value:    []int64{1000},
			NumLabel: map[string][]int64{
				"key1": {10},
				"key2": {30},
			},
			NumUnit: map[string][]string{
				"key1": {"kilobytes"},
				"key2": {"kilobytes"},
			},
		},
	},
	Location: cpuL,
	Function: cpuF,
	Mapping:  cpuM,
}

var aggTests = map[string]aggTest{
	"precise":         {true, true, true, true, 5},
	"fileline":        {false, true, true, true, 4},
	"inline_function": {false, true, false, true, 3},
	"function":        {false, true, false, false, 2},
}

type aggTest struct {
	precise, function, fileline, inlineFrame bool
	rows                                     int
}

const totalSamples = int64(11111)

func TestAggregation(t *testing.T) {
	prof := testProfile1.Copy()
	for _, resolution := range []string{"precise", "fileline", "inline_function", "function"} {
		a := aggTests[resolution]
		if !a.precise {
			if err := prof.Aggregate(a.inlineFrame, a.function, a.fileline, a.fileline, false); err != nil {
				t.Error("aggregating to " + resolution + ":" + err.Error())
			}
		}
		if err := checkAggregation(prof, &a); err != nil {
			t.Error("failed aggregation to " + resolution + ": " + err.Error())
		}
	}
}

// checkAggregation verifies that the profile remained consistent
// with its aggregation.
func checkAggregation(prof *Profile, a *aggTest) error {
	// Check that the total number of samples for the rows was preserved.
	total := int64(0)

	samples := make(map[string]bool)
	for _, sample := range prof.Sample {
		tb := locationHash(sample)
		samples[tb] = true
		total += sample.Value[0]
	}

	if total != totalSamples {
		return fmt.Errorf("sample total %d, want %d", total, totalSamples)
	}

	// Check the number of unique sample locations
	if a.rows != len(samples) {
		return fmt.Errorf("number of samples %d, want %d", len(samples), a.rows)
	}

	// Check that all mappings have the right detail flags.
	for _, m := range prof.Mapping {
		if m.HasFunctions != a.function {
			return fmt.Errorf("unexpected mapping.HasFunctions %v, want %v", m.HasFunctions, a.function)
		}
		if m.HasFilenames != a.fileline {
			return fmt.Errorf("unexpected mapping.HasFilenames %v, want %v", m.HasFilenames, a.fileline)
		}
		if m.HasLineNumbers != a.fileline {
			return fmt.Errorf("unexpected mapping.HasLineNumbers %v, want %v", m.HasLineNumbers, a.fileline)
		}
		if m.HasInlineFrames != a.inlineFrame {
			return fmt.Errorf("unexpected mapping.HasInlineFrames %v, want %v", m.HasInlineFrames, a.inlineFrame)
		}
	}

	// Check that aggregation has removed finer resolution data.
	for _, l := range prof.Location {
		if !a.inlineFrame && len(l.Line) > 1 {
			return fmt.Errorf("found %d lines on location %d, want 1", len(l.Line), l.ID)
		}

		for _, ln := range l.Line {
			if !a.fileline && (ln.Function.Filename != "" || ln.Line != 0) {
				return fmt.Errorf("found line %s:%d on location %d, want :0",
					ln.Function.Filename, ln.Line, l.ID)
			}
			if !a.function && (ln.Function.Name != "") {
				return fmt.Errorf(`found file %s location %d, want ""`,
					ln.Function.Name, l.ID)
			}
		}
	}

	return nil
}

// Test merge leaves the main binary in place.
func TestMergeMain(t *testing.T) {
	prof := testProfile1.Copy()
	p1, err := Merge([]*Profile{prof})
	if err != nil {
		t.Fatalf("merge error: %v", err)
	}
	if cpuM[0].File != p1.Mapping[0].File {
		t.Errorf("want Mapping[0]=%s got %s", cpuM[0].File, p1.Mapping[0].File)
	}
}

func TestMerge(t *testing.T) {
	// Aggregate a profile with itself and once again with a factor of
	// -2. Should end up with an empty profile (all samples for a
	// location should add up to 0).

	prof := testProfile1.Copy()
	prof.Comments = []string{"comment1"}
	p1, err := Merge([]*Profile{prof, prof})
	if err != nil {
		t.Errorf("merge error: %v", err)
	}
	prof.Scale(-2)
	prof, err = Merge([]*Profile{p1, prof})
	if err != nil {
		t.Errorf("merge error: %v", err)
	}
	if got, want := len(prof.Comments), 1; got != want {
		t.Errorf("len(prof.Comments) = %d, want %d", got, want)
	}

	// Use aggregation to merge locations at function granularity.
	if err := prof.Aggregate(false, true, false, false, false); err != nil {
		t.Errorf("aggregating after merge: %v", err)
	}

	samples := make(map[string]int64)
	for _, s := range prof.Sample {
		tb := locationHash(s)
		samples[tb] = samples[tb] + s.Value[0]
	}
	for s, v := range samples {
		if v != 0 {
			t.Errorf("nonzero value for sample %s: %d", s, v)
		}
	}
}

func TestMergeAll(t *testing.T) {
	// Aggregate 10 copies of the profile.
	profs := make([]*Profile, 10)
	for i := 0; i < 10; i++ {
		profs[i] = testProfile1.Copy()
	}
	prof, err := Merge(profs)
	if err != nil {
		t.Errorf("merge error: %v", err)
	}
	samples := make(map[string]int64)
	for _, s := range prof.Sample {
		tb := locationHash(s)
		samples[tb] = samples[tb] + s.Value[0]
	}
	for _, s := range testProfile1.Sample {
		tb := locationHash(s)
		if samples[tb] != s.Value[0]*10 {
			t.Errorf("merge got wrong value at %s : %d instead of %d", tb, samples[tb], s.Value[0]*10)
		}
	}
}

func TestIsFoldedMerge(t *testing.T) {
	testProfile1Folded := testProfile1.Copy()
	testProfile1Folded.Location[0].IsFolded = true
	testProfile1Folded.Location[1].IsFolded = true

	for _, tc := range []struct {
		name            string
		profs           []*Profile
		wantLocationLen int
	}{
		{
			name:            "folded and non-folded locations not merged",
			profs:           []*Profile{testProfile1.Copy(), testProfile1Folded.Copy()},
			wantLocationLen: 7,
		},
		{
			name:            "identical folded locations are merged",
			profs:           []*Profile{testProfile1Folded.Copy(), testProfile1Folded.Copy()},
			wantLocationLen: 5,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prof, err := Merge(tc.profs)
			if err != nil {
				t.Fatalf("merge error: %v", err)
			}
			if got, want := len(prof.Location), tc.wantLocationLen; got != want {
				t.Fatalf("got %d locations, want %d locations", got, want)
			}
		})
	}
}

func TestNumLabelMerge(t *testing.T) {
	for _, tc := range []struct {
		name          string
		profs         []*Profile
		wantNumLabels []map[string][]int64
		wantNumUnits  []map[string][]string
	}{
		{
			name:  "different label units not merged",
			profs: []*Profile{testProfile4.Copy(), testProfile5.Copy()},
			wantNumLabels: []map[string][]int64{
				{
					"key1": {10},
					"key2": {30},
				},
				{
					"key1": {10},
					"key2": {30},
				},
			},
			wantNumUnits: []map[string][]string{
				{
					"key1": {"bytes"},
					"key2": {"bytes"},
				},
				{
					"key1": {"kilobytes"},
					"key2": {"kilobytes"},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prof, err := Merge(tc.profs)
			if err != nil {
				t.Errorf("merge error: %v", err)
			}

			if want, got := len(tc.wantNumLabels), len(prof.Sample); want != got {
				t.Fatalf("got %d samples, want %d samples", got, want)
			}
			for i, wantLabels := range tc.wantNumLabels {
				numLabels := prof.Sample[i].NumLabel
				if !reflect.DeepEqual(wantLabels, numLabels) {
					t.Errorf("got numeric labels %v, want %v", numLabels, wantLabels)
				}

				wantUnits := tc.wantNumUnits[i]
				numUnits := prof.Sample[i].NumUnit
				if !reflect.DeepEqual(wantUnits, numUnits) {
					t.Errorf("got numeric labels %v, want %v", numUnits, wantUnits)
				}
			}
		})
	}
}

func TestEmptyMappingMerge(t *testing.T) {
	// Aggregate a profile with itself and once again with a factor of
	// -2. Should end up with an empty profile (all samples for a
	// location should add up to 0).

	prof1 := testProfile1.Copy()
	prof2 := testProfile1NoMapping.Copy()
	p1, err := Merge([]*Profile{prof2, prof1})
	if err != nil {
		t.Errorf("merge error: %v", err)
	}
	prof2.Scale(-2)
	prof, err := Merge([]*Profile{p1, prof2})
	if err != nil {
		t.Errorf("merge error: %v", err)
	}

	// Use aggregation to merge locations at function granularity.
	if err := prof.Aggregate(false, true, false, false, false); err != nil {
		t.Errorf("aggregating after merge: %v", err)
	}

	samples := make(map[string]int64)
	for _, s := range prof.Sample {
		tb := locationHash(s)
		samples[tb] = samples[tb] + s.Value[0]
	}
	for s, v := range samples {
		if v != 0 {
			t.Errorf("nonzero value for sample %s: %d", s, v)
		}
	}
}

func TestNormalizeBySameProfile(t *testing.T) {
	pb := testProfile1.Copy()
	p := testProfile1.Copy()

	if err := p.Normalize(pb); err != nil {
		t.Fatal(err)
	}

	for i, s := range p.Sample {
		for j, v := range s.Value {
			expectedSampleValue := testProfile1.Sample[i].Value[j]
			if v != expectedSampleValue {
				t.Errorf("For sample %d, value %d want %d got %d", i, j, expectedSampleValue, v)
			}
		}
	}
}

func TestNormalizeByDifferentProfile(t *testing.T) {
	p := testProfile1.Copy()
	pb := testProfile2.Copy()

	if err := p.Normalize(pb); err != nil {
		t.Fatal(err)
	}

	expectedSampleValues := [][]int64{
		{19, 1000},
		{1, 100},
		{0, 10},
		{198, 10000},
		{0, 1},
	}

	for i, s := range p.Sample {
		for j, v := range s.Value {
			if v != expectedSampleValues[i][j] {
				t.Errorf("For sample %d, value %d want %d got %d", i, j, expectedSampleValues[i][j], v)
			}
		}
	}
}

func TestNormalizeByMultipleOfSameProfile(t *testing.T) {
	pb := testProfile1.Copy()
	for i, s := range pb.Sample {
		for j, v := range s.Value {
			pb.Sample[i].Value[j] = 10 * v
		}
	}

	p := testProfile1.Copy()

	err := p.Normalize(pb)
	if err != nil {
		t.Fatal(err)
	}

	for i, s := range p.Sample {
		for j, v := range s.Value {
			expectedSampleValue := 10 * testProfile1.Sample[i].Value[j]
			if v != expectedSampleValue {
				t.Errorf("For sample %d, value %d, want %d got %d", i, j, expectedSampleValue, v)
			}
		}
	}
}

func TestNormalizeIncompatibleProfiles(t *testing.T) {
	p := testProfile1.Copy()
	pb := testProfile3.Copy()

	if err := p.Normalize(pb); err == nil {
		t.Errorf("Expected an error")
	}
}

// locationHash constructs a string to use as a hashkey for a sample, based on its locations
func locationHash(s *Sample) string {
	var tb string
	for _, l := range s.Location {
		for _, ln := range l.Line {
			tb = tb + fmt.Sprintf("%s:%d@%d ", ln.Function.Name, ln.Line, l.Address)
		}
	}
	return tb
}

func TestHasLabel(t *testing.T) {
	var testcases = []struct {
		desc         string
		labels       map[string][]string
		key          string
		value        string
		wantHasLabel bool
	}{
		{
			desc:         "empty label does not have label",
			labels:       map[string][]string{},
			key:          "key",
			value:        "value",
			wantHasLabel: false,
		},
		{
			desc:         "label with one key and value has label",
			labels:       map[string][]string{"key": {"value"}},
			key:          "key",
			value:        "value",
			wantHasLabel: true,
		},
		{
			desc:         "label with one key and value does not have label",
			labels:       map[string][]string{"key": {"value"}},
			key:          "key1",
			value:        "value1",
			wantHasLabel: false,
		},
		{
			desc: "label with many keys and values has label",
			labels: map[string][]string{
				"key1": {"value2", "value1"},
				"key2": {"value1", "value2", "value2"},
				"key3": {"value1", "value2", "value2"},
			},
			key:          "key1",
			value:        "value1",
			wantHasLabel: true,
		},
		{
			desc: "label with many keys and values does not have label",
			labels: map[string][]string{
				"key1": {"value2", "value1"},
				"key2": {"value1", "value2", "value2"},
				"key3": {"value1", "value2", "value2"},
			},
			key:          "key5",
			value:        "value5",
			wantHasLabel: false,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			sample := &Sample{
				Label: tc.labels,
			}
			if gotHasLabel := sample.HasLabel(tc.key, tc.value); gotHasLabel != tc.wantHasLabel {
				t.Errorf("sample.HasLabel(%q, %q) got %v, want %v", tc.key, tc.value, gotHasLabel, tc.wantHasLabel)
			}
		})
	}
}

func TestDiffBaseSample(t *testing.T) {
	var testcases = []struct {
		desc               string
		labels             map[string][]string
		wantDiffBaseSample bool
	}{
		{
			desc:               "empty label does not have label",
			labels:             map[string][]string{},
			wantDiffBaseSample: false,
		},
		{
			desc:               "label with one key and value, including diff base label",
			labels:             map[string][]string{"pprof::base": {"true"}},
			wantDiffBaseSample: true,
		},
		{
			desc:               "label with one key and value, not including diff base label",
			labels:             map[string][]string{"key": {"value"}},
			wantDiffBaseSample: false,
		},
		{
			desc: "label with many keys and values, including diff base label",
			labels: map[string][]string{
				"pprof::base": {"value2", "true"},
				"key2":        {"true", "value2", "value2"},
				"key3":        {"true", "value2", "value2"},
			},
			wantDiffBaseSample: true,
		},
		{
			desc: "label with many keys and values, not including diff base label",
			labels: map[string][]string{
				"key1": {"value2", "value1"},
				"key2": {"value1", "value2", "value2"},
				"key3": {"value1", "value2", "value2"},
			},
			wantDiffBaseSample: false,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			sample := &Sample{
				Label: tc.labels,
			}
			if gotHasLabel := sample.DiffBaseSample(); gotHasLabel != tc.wantDiffBaseSample {
				t.Errorf("sample.DiffBaseSample() got %v, want %v", gotHasLabel, tc.wantDiffBaseSample)
			}
		})
	}
}

func TestRemove(t *testing.T) {
	var testcases = []struct {
		desc       string
		samples    []*Sample
		removeKey  string
		wantLabels []map[string][]string
	}{
		{
			desc: "some samples have label already",
			samples: []*Sample{
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
				},
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
					Label: map[string][]string{
						"key1": {"value1", "value2", "value3"},
						"key2": {"value1"},
					},
				},
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
					Label: map[string][]string{
						"key1": {"value2"},
					},
				},
			},
			removeKey: "key1",
			wantLabels: []map[string][]string{
				{},
				{"key2": {"value1"}},
				{},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			profile := testProfile1.Copy()
			profile.Sample = tc.samples
			profile.RemoveLabel(tc.removeKey)
			if got, want := len(profile.Sample), len(tc.wantLabels); got != want {
				t.Fatalf("got %v samples, want %v samples", got, want)
			}
			for i, sample := range profile.Sample {
				wantLabels := tc.wantLabels[i]
				if got, want := len(sample.Label), len(wantLabels); got != want {
					t.Errorf("got %v label keys for sample %v, want %v", got, i, want)
					continue
				}
				for wantKey, wantValues := range wantLabels {
					if gotValues, ok := sample.Label[wantKey]; ok {
						if !reflect.DeepEqual(gotValues, wantValues) {
							t.Errorf("for key %s, got values %v, want values %v", wantKey, gotValues, wantValues)
						}
					} else {
						t.Errorf("for key %s got no values, want %v", wantKey, wantValues)
					}
				}
			}
		})
	}
}

func TestSetLabel(t *testing.T) {
	var testcases = []struct {
		desc       string
		samples    []*Sample
		setKey     string
		setVal     []string
		wantLabels []map[string][]string
	}{
		{
			desc: "some samples have label already",
			samples: []*Sample{
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
				},
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
					Label: map[string][]string{
						"key1": {"value1", "value2", "value3"},
						"key2": {"value1"},
					},
				},
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
					Label: map[string][]string{
						"key1": {"value2"},
					},
				},
			},
			setKey: "key1",
			setVal: []string{"value1"},
			wantLabels: []map[string][]string{
				{"key1": {"value1"}},
				{"key1": {"value1"}, "key2": {"value1"}},
				{"key1": {"value1"}},
			},
		},
		{
			desc: "no samples have labels",
			samples: []*Sample{
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
				},
			},
			setKey: "key1",
			setVal: []string{"value1"},
			wantLabels: []map[string][]string{
				{"key1": {"value1"}},
			},
		},
		{
			desc: "all samples have some labels, but not key being added",
			samples: []*Sample{
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
					Label: map[string][]string{
						"key2": {"value2"},
					},
				},
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
					Label: map[string][]string{
						"key3": {"value3"},
					},
				},
			},
			setKey: "key1",
			setVal: []string{"value1"},
			wantLabels: []map[string][]string{
				{"key1": {"value1"}, "key2": {"value2"}},
				{"key1": {"value1"}, "key3": {"value3"}},
			},
		},
		{
			desc: "all samples have key being added",
			samples: []*Sample{
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
					Label: map[string][]string{
						"key1": {"value1"},
					},
				},
				{
					Location: []*Location{cpuL[0]},
					Value:    []int64{1000},
					Label: map[string][]string{
						"key1": {"value1"},
					},
				},
			},
			setKey: "key1",
			setVal: []string{"value1"},
			wantLabels: []map[string][]string{
				{"key1": {"value1"}},
				{"key1": {"value1"}},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			profile := testProfile1.Copy()
			profile.Sample = tc.samples
			profile.SetLabel(tc.setKey, tc.setVal)
			if got, want := len(profile.Sample), len(tc.wantLabels); got != want {
				t.Fatalf("got %v samples, want %v samples", got, want)
			}
			for i, sample := range profile.Sample {
				wantLabels := tc.wantLabels[i]
				if got, want := len(sample.Label), len(wantLabels); got != want {
					t.Errorf("got %v label keys for sample %v, want %v", got, i, want)
					continue
				}
				for wantKey, wantValues := range wantLabels {
					if gotValues, ok := sample.Label[wantKey]; ok {
						if !reflect.DeepEqual(gotValues, wantValues) {
							t.Errorf("for key %s, got values %v, want values %v", wantKey, gotValues, wantValues)
						}
					} else {
						t.Errorf("for key %s got no values, want %v", wantKey, wantValues)
					}
				}
			}
		})
	}
}

func TestNumLabelUnits(t *testing.T) {
	var tagFilterTests = []struct {
		desc             string
		tagVals          []map[string][]int64
		tagUnits         []map[string][]string
		wantUnits        map[string]string
		wantIgnoredUnits map[string][]string
	}{
		{
			"One sample, multiple keys, different specified units",
			[]map[string][]int64{{"key1": {131072}, "key2": {128}}},
			[]map[string][]string{{"key1": {"bytes"}, "key2": {"kilobytes"}}},
			map[string]string{"key1": "bytes", "key2": "kilobytes"},
			map[string][]string{},
		},
		{
			"One sample, one key with one value, unit specified",
			[]map[string][]int64{{"key1": {8}}},
			[]map[string][]string{{"key1": {"bytes"}}},
			map[string]string{"key1": "bytes"},
			map[string][]string{},
		},
		{
			"One sample, one key with one value, empty unit specified",
			[]map[string][]int64{{"key1": {8}}},
			[]map[string][]string{{"key1": {""}}},
			map[string]string{"key1": "key1"},
			map[string][]string{},
		},
		{
			"Key bytes, unit not specified",
			[]map[string][]int64{{"bytes": {8}}},
			[]map[string][]string{nil},
			map[string]string{"bytes": "bytes"},
			map[string][]string{},
		},
		{
			"One sample, one key with one value, unit not specified",
			[]map[string][]int64{{"kilobytes": {8}}},
			[]map[string][]string{nil},
			map[string]string{"kilobytes": "kilobytes"},
			map[string][]string{},
		},
		{
			"Key request, unit not specified",
			[]map[string][]int64{{"request": {8}}},
			[]map[string][]string{nil},
			map[string]string{"request": "bytes"},
			map[string][]string{},
		},
		{
			"Key alignment, unit not specified",
			[]map[string][]int64{{"alignment": {8}}},
			[]map[string][]string{nil},
			map[string]string{"alignment": "bytes"},
			map[string][]string{},
		},
		{
			"One sample, one key with multiple values and two different units",
			[]map[string][]int64{{"key1": {8, 8}}},
			[]map[string][]string{{"key1": {"bytes", "kilobytes"}}},
			map[string]string{"key1": "bytes"},
			map[string][]string{"key1": {"kilobytes"}},
		},
		{
			"One sample, one key with multiple values and three different units",
			[]map[string][]int64{{"key1": {8, 8}}},
			[]map[string][]string{{"key1": {"bytes", "megabytes", "kilobytes"}}},
			map[string]string{"key1": "bytes"},
			map[string][]string{"key1": {"kilobytes", "megabytes"}},
		},
		{
			"Two samples, one key, different units specified",
			[]map[string][]int64{{"key1": {8}}, {"key1": {8}}},
			[]map[string][]string{{"key1": {"bytes"}}, {"key1": {"kilobytes"}}},
			map[string]string{"key1": "bytes"},
			map[string][]string{"key1": {"kilobytes"}},
		},
		{
			"Keys alignment, request, and bytes have units specified",
			[]map[string][]int64{{
				"alignment": {8},
				"request":   {8},
				"bytes":     {8},
			}},
			[]map[string][]string{{
				"alignment": {"seconds"},
				"request":   {"minutes"},
				"bytes":     {"hours"},
			}},
			map[string]string{
				"alignment": "seconds",
				"request":   "minutes",
				"bytes":     "hours",
			},
			map[string][]string{},
		},
	}
	for _, test := range tagFilterTests {
		p := &Profile{Sample: make([]*Sample, len(test.tagVals))}
		for i, numLabel := range test.tagVals {
			s := Sample{
				NumLabel: numLabel,
				NumUnit:  test.tagUnits[i],
			}
			p.Sample[i] = &s
		}
		units, ignoredUnits := p.NumLabelUnits()
		if !reflect.DeepEqual(test.wantUnits, units) {
			t.Errorf("%s: got %v units, want %v", test.desc, units, test.wantUnits)
		}
		if !reflect.DeepEqual(test.wantIgnoredUnits, ignoredUnits) {
			t.Errorf("%s: got %v ignored units, want %v", test.desc, ignoredUnits, test.wantIgnoredUnits)
		}
	}
}

func TestSetMain(t *testing.T) {
	testProfile1.massageMappings()
	if testProfile1.Mapping[0].File != mainBinary {
		t.Errorf("got %s for main", testProfile1.Mapping[0].File)
	}
}

// parallel runs n copies of fn in parallel.
func parallel(n int, fn func()) {
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			fn()
			wg.Done()
		}()
	}
	wg.Wait()
}

func TestThreadSafety(t *testing.T) {
	src := testProfile1.Copy()
	parallel(4, func() { src.Copy() })
	parallel(4, func() {
		var b bytes.Buffer
		src.WriteUncompressed(&b)
	})
	parallel(4, func() {
		var b bytes.Buffer
		src.Write(&b)
	})
}
