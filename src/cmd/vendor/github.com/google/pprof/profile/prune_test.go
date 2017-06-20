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
	"strings"
	"testing"
)

func TestPrune(t *testing.T) {
	for _, test := range []struct {
		in   *Profile
		want string
	}{
		{in1, out1},
	} {
		in := test.in.Copy()
		in.RemoveUninteresting()
		if err := in.CheckValid(); err != nil {
			t.Error(err)
		}
		w := strings.Split(test.want, "\n")
		for i, g := range strings.Split(in.String(), "\n") {
			if i >= len(w) {
				t.Fatalf("got trailing %s", g)
			}
			if strings.TrimSpace(g) != strings.TrimSpace(w[i]) {
				t.Fatalf(`%d: got: "%s"  want:"%s"`, i, g, w[i])
			}
		}
	}
}

var funs = []*Function{
	{ID: 1, Name: "main", SystemName: "main", Filename: "main.c"},
	{ID: 2, Name: "fun1", SystemName: "fun1", Filename: "fun.c"},
	{ID: 3, Name: "fun2", SystemName: "fun2", Filename: "fun.c"},
	{ID: 4, Name: "fun3", SystemName: "fun3", Filename: "fun.c"},
	{ID: 5, Name: "fun4", SystemName: "fun4", Filename: "fun.c"},
	{ID: 6, Name: "fun5", SystemName: "fun5", Filename: "fun.c"},
}

var locs1 = []*Location{
	{
		ID: 1,
		Line: []Line{
			{Function: funs[0], Line: 1},
		},
	},
	{
		ID: 2,
		Line: []Line{
			{Function: funs[1], Line: 2},
			{Function: funs[2], Line: 1},
		},
	},
	{
		ID: 3,
		Line: []Line{
			{Function: funs[3], Line: 2},
			{Function: funs[1], Line: 1},
		},
	},
	{
		ID: 4,
		Line: []Line{
			{Function: funs[3], Line: 2},
			{Function: funs[1], Line: 2},
			{Function: funs[5], Line: 2},
		},
	},
}

var in1 = &Profile{
	PeriodType:    &ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:        1,
	DurationNanos: 10e9,
	SampleType: []*ValueType{
		{Type: "samples", Unit: "count"},
		{Type: "cpu", Unit: "milliseconds"},
	},
	Sample: []*Sample{
		{
			Location: []*Location{locs1[0]},
			Value:    []int64{1, 1},
		},
		{
			Location: []*Location{locs1[1], locs1[0]},
			Value:    []int64{1, 1},
		},
		{
			Location: []*Location{locs1[2], locs1[0]},
			Value:    []int64{1, 1},
		},
		{
			Location: []*Location{locs1[3], locs1[0]},
			Value:    []int64{1, 1},
		},
		{
			Location: []*Location{locs1[3], locs1[2], locs1[1], locs1[0]},
			Value:    []int64{1, 1},
		},
	},
	Location:   locs1,
	Function:   funs,
	DropFrames: "fu.*[12]|banana",
	KeepFrames: ".*[n2][n2]",
}

const out1 = `PeriodType: cpu milliseconds
Period: 1
Duration: 10s
Samples:
samples/count cpu/milliseconds
          1          1: 1
          1          1: 2 1
          1          1: 1
          1          1: 4 1
          1          1: 2 1
Locations
     1: 0x0 main main.c:1 s=0
     2: 0x0 fun2 fun.c:1 s=0
     3: 0x0 fun3 fun.c:2 s=0
             fun1 fun.c:1 s=0
     4: 0x0 fun5 fun.c:2 s=0
Mappings
`
