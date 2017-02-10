// Copyright 2016 Google Inc. All Rights Reserved.
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
	"testing"
)

func TestSampleIndexByName(t *testing.T) {
	for _, c := range []struct {
		desc              string
		sampleTypes       []string
		defaultSampleType string
		index             string
		want              int
		wantError         bool
	}{
		{
			desc:        "use last by default",
			index:       "",
			want:        1,
			sampleTypes: []string{"zero", "default"},
		},
		{
			desc:              "honour specified default",
			index:             "",
			want:              1,
			defaultSampleType: "default",
			sampleTypes:       []string{"zero", "default", "two"},
		},
		{
			desc:              "invalid default is ignored",
			index:             "",
			want:              2,
			defaultSampleType: "non-existent",
			sampleTypes:       []string{"zero", "one", "default"},
		},
		{
			desc:        "index by int",
			index:       "0",
			want:        0,
			sampleTypes: []string{"zero", "one", "two"},
		},
		{
			desc:              "index by int ignores default",
			index:             "0",
			want:              0,
			defaultSampleType: "default",
			sampleTypes:       []string{"zero", "default", "two"},
		},
		{
			desc:        "index by name",
			index:       "two",
			want:        2,
			sampleTypes: []string{"zero", "one", "two", "three"},
		},
		{
			desc:              "index by name ignores default",
			index:             "zero",
			want:              0,
			defaultSampleType: "default",
			sampleTypes:       []string{"zero", "default", "two"},
		},
		{
			desc:        "out of bound int causes error",
			index:       "100",
			wantError:   true,
			sampleTypes: []string{"zero", "default"},
		},
		{
			desc:        "unknown name causes error",
			index:       "does not exist",
			wantError:   true,
			sampleTypes: []string{"zero", "default"},
		},
		{
			desc:        "'inused_{x}' recognized for legacy '{x}'",
			index:       "inuse_zero",
			want:        0,
			sampleTypes: []string{"zero", "default"},
		},
	} {
		p := &Profile{
			DefaultSampleType: c.defaultSampleType,
			SampleType:        []*ValueType{},
		}
		for _, st := range c.sampleTypes {
			p.SampleType = append(p.SampleType, &ValueType{Type: st, Unit: "milliseconds"})
		}

		got, err := p.SampleIndexByName(c.index)

		switch {
		case c.wantError && err == nil:
			t.Errorf("%s: error should have been returned not index=%d, err=%v", c.desc, got, err)
		case !c.wantError && err != nil:
			t.Errorf("%s: unexpected got index=%d, err=%v; wanted index=%d, err=nil", c.desc, got, err, c.want)
		case !c.wantError && got != c.want:
			t.Errorf("%s: got index=%d, want index=%d", c.desc, got, c.want)
		}
	}
}
