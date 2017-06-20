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
	"testing"

	"github.com/google/pprof/internal/proftest"
)

var testM = []*Mapping{
	{
		ID:              1,
		Start:           1,
		Limit:           10,
		Offset:          0,
		File:            "file1",
		BuildID:         "buildid1",
		HasFunctions:    true,
		HasFilenames:    true,
		HasLineNumbers:  true,
		HasInlineFrames: true,
	},
	{
		ID:              2,
		Start:           10,
		Limit:           30,
		Offset:          9,
		File:            "file1",
		BuildID:         "buildid2",
		HasFunctions:    true,
		HasFilenames:    true,
		HasLineNumbers:  true,
		HasInlineFrames: true,
	},
}

var testF = []*Function{
	{ID: 1, Name: "func1", SystemName: "func1", Filename: "file1"},
	{ID: 2, Name: "func2", SystemName: "func2", Filename: "file1"},
	{ID: 3, Name: "func3", SystemName: "func3", Filename: "file2"},
}

var testL = []*Location{
	{
		ID:      1,
		Address: 1,
		Mapping: testM[0],
		Line: []Line{
			{
				Function: testF[0],
				Line:     2,
			},
			{
				Function: testF[1],
				Line:     2222222,
			},
		},
	},
	{
		ID:      2,
		Mapping: testM[1],
		Address: 11,
		Line: []Line{
			{
				Function: testF[2],
				Line:     2,
			},
		},
	},
	{
		ID:      3,
		Mapping: testM[1],
		Address: 12,
	},
}

var all = &Profile{
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        10,
	DurationNanos: 10e9,
	SampleType: []*ValueType{
		{Type: "cpu", Unit: "cycles"},
		{Type: "object", Unit: "count"},
	},
	Sample: []*Sample{
		{
			Location: []*Location{testL[0], testL[1], testL[2], testL[1], testL[1]},
			Label: map[string][]string{
				"key1": []string{"value1"},
				"key2": []string{"value2"},
			},
			Value: []int64{10, 20},
		},
		{
			Location: []*Location{testL[1], testL[2], testL[0], testL[1]},
			Value:    []int64{30, 40},
			Label: map[string][]string{
				"key1": []string{"value1"},
				"key2": []string{"value2"},
			},
			NumLabel: map[string][]int64{
				"key1": []int64{1, 2},
				"key2": []int64{3, 4},
			},
		},
	},
	Function: testF,
	Mapping:  testM,
	Location: testL,
	Comments: []string{"Comment 1", "Comment 2"},
}

func TestMarshalUnmarshal(t *testing.T) {
	// Write the profile, parse it, and ensure they're equal.
	buf := bytes.NewBuffer(nil)
	all.Write(buf)
	all2, err := Parse(buf)
	if err != nil {
		t.Fatal(err)
	}

	js1 := proftest.EncodeJSON(&all)
	js2 := proftest.EncodeJSON(&all2)
	if string(js1) != string(js2) {
		t.Errorf("profiles differ")
		d, err := proftest.Diff(js1, js2)
		if err != nil {
			t.Fatal(err)
		}
		t.Error("\n" + string(d))
	}
}
