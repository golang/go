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
	"fmt"
	"strconv"
	"strings"
)

// SampleIndexByName returns the appropriate index for a value of sample index.
// If numeric, it returns the number, otherwise it looks up the text in the
// profile sample types.
func (p *Profile) SampleIndexByName(sampleIndex string) (int, error) {
	if sampleIndex == "" {
		if dst := p.DefaultSampleType; dst != "" {
			for i, t := range sampleTypes(p) {
				if t == dst {
					return i, nil
				}
			}
		}
		// By default select the last sample value
		return len(p.SampleType) - 1, nil
	}
	if i, err := strconv.Atoi(sampleIndex); err == nil {
		if i < 0 || i >= len(p.SampleType) {
			return 0, fmt.Errorf("sample_index %s is outside the range [0..%d]", sampleIndex, len(p.SampleType)-1)
		}
		return i, nil
	}

	// Remove the inuse_ prefix to support legacy pprof options
	// "inuse_space" and "inuse_objects" for profiles containing types
	// "space" and "objects".
	noInuse := strings.TrimPrefix(sampleIndex, "inuse_")
	for i, t := range p.SampleType {
		if t.Type == sampleIndex || t.Type == noInuse {
			return i, nil
		}
	}

	return 0, fmt.Errorf("sample_index %q must be one of: %v", sampleIndex, sampleTypes(p))
}

func sampleTypes(p *Profile) []string {
	types := make([]string, len(p.SampleType))
	for i, t := range p.SampleType {
		types[i] = t.Type
	}
	return types
}
