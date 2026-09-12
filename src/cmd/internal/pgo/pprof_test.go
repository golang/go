// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pgo

import (
	"bytes"
	"internal/profile"
	"testing"
)

func testProfile(sampleTypes []*profile.ValueType, defaultSampleType string, values []int64) *profile.Profile {
	main := &profile.Function{ID: 1, Name: "main.main", StartLine: 10}
	foo := &profile.Function{ID: 2, Name: "main.foo", StartLine: 20}
	mainLoc := &profile.Location{ID: 1, Line: []profile.Line{{Function: main, Line: 12}}}
	fooLoc := &profile.Location{ID: 2, Line: []profile.Line{{Function: foo, Line: 22}}}
	return &profile.Profile{
		SampleType:        sampleTypes,
		DefaultSampleType: defaultSampleType,
		Function:          []*profile.Function{main, foo},
		Location:          []*profile.Location{mainLoc, fooLoc},
		Sample: []*profile.Sample{
			// Stacks are leaf-first: foo called from main.
			{Location: []*profile.Location{fooLoc, mainLoc}, Value: values},
		},
	}
}

func serializeProfile(t *testing.T, p *profile.Profile) *bytes.Reader {
	t.Helper()
	var buf bytes.Buffer
	if err := p.Write(&buf); err != nil {
		t.Fatalf("writing profile: %v", err)
	}
	return bytes.NewReader(buf.Bytes())
}

func TestFromPProfSampleType(t *testing.T) {
	perf := []*profile.ValueType{
		{Type: "cycles_sample", Unit: "count"},
		{Type: "cycles_event", Unit: "count"},
	}

	tests := []struct {
		name            string
		sampleTypes     []*profile.ValueType
		defaultType     string
		values          []int64
		wantErr         bool
		wantTotalWeight int64
	}{
		{
			name:            "native samples/count",
			sampleTypes:     []*profile.ValueType{{Type: "samples", Unit: "count"}, {Type: "cpu", Unit: "nanoseconds"}},
			values:          []int64{5, 500},
			wantTotalWeight: 5,
		},
		{
			name:            "native cpu/nanoseconds",
			sampleTypes:     []*profile.ValueType{{Type: "cpu", Unit: "nanoseconds"}},
			values:          []int64{500},
			wantTotalWeight: 500,
		},
		{
			name:            "perf via default sample type",
			sampleTypes:     perf,
			defaultType:     "cycles_event",
			values:          []int64{5, 500},
			wantTotalWeight: 500,
		},
		{
			name:            "perf without default uses last value type",
			sampleTypes:     perf,
			values:          []int64{5, 500},
			wantTotalWeight: 500,
		},
		{
			name:        "heap profile rejected",
			sampleTypes: []*profile.ValueType{{Type: "alloc_objects", Unit: "count"}, {Type: "inuse_space", Unit: "bytes"}},
			defaultType: "inuse_space",
			values:      []int64{1, 2},
			wantErr:     true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := FromPProf(serializeProfile(t, testProfile(tc.sampleTypes, tc.defaultType, tc.values)))
			if tc.wantErr {
				if err == nil {
					t.Fatal("FromPProf succeeded, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("FromPProf: %v", err)
			}
			if len(got.NamedEdgeMap.ByWeight) == 0 {
				t.Error("empty edge map, want a call edge")
			}
			if got.TotalWeight != tc.wantTotalWeight {
				t.Errorf("TotalWeight = %d, want %d", got.TotalWeight, tc.wantTotalWeight)
			}
		})
	}
}
