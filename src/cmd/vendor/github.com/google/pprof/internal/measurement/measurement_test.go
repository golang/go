// Copyright 2017 Google Inc. All Rights Reserved.
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

package measurement

import (
	"testing"
)

func TestScale(t *testing.T) {
	for _, tc := range []struct {
		value            int64
		fromUnit, toUnit string
		wantValue        float64
		wantUnit         string
	}{
		{1, "s", "ms", 1000, "ms"},
		{1, "kb", "b", 1024, "B"},
		{1, "kbyte", "b", 1024, "B"},
		{1, "kilobyte", "b", 1024, "B"},
		{1, "mb", "kb", 1024, "kB"},
		{1, "gb", "mb", 1024, "MB"},
		{1024, "gb", "tb", 1, "TB"},
		{1024, "tb", "pb", 1, "PB"},
		{2048, "mb", "auto", 2, "GB"},
		{3.1536e7, "s", "auto", 1, "yrs"},
		{-1, "s", "ms", -1000, "ms"},
		{1, "foo", "count", 1, ""},
		{1, "foo", "bar", 1, "bar"},
	} {
		if gotValue, gotUnit := Scale(tc.value, tc.fromUnit, tc.toUnit); gotValue != tc.wantValue || gotUnit != tc.wantUnit {
			t.Errorf("Scale(%d, %q, %q) = (%f, %q), want (%f, %q)",
				tc.value, tc.fromUnit, tc.toUnit, gotValue, gotUnit, tc.wantValue, tc.wantUnit)
		}
	}
}
