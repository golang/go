// Copyright 2022 Google Inc. All Rights Reserved.
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
	"path/filepath"
	"regexp"

	"github.com/google/pprof/internal/graph"
)

var (
	sepRE     = regexp.MustCompile(`::|\.`)
	fileSepRE = regexp.MustCompile(`/`)
)

// fileNameSuffixes returns a non-empty sequence of shortened file names
// (in decreasing preference) that can be used to represent name.
func fileNameSuffixes(name string) []string {
	if name == "" {
		// Avoid returning "." when symbol info is missing
		return []string{""}
	}
	return allSuffixes(filepath.ToSlash(filepath.Clean(name)), fileSepRE)
}

// shortNameList returns a non-empty sequence of shortened names
// (in decreasing preference) that can be used to represent name.
func shortNameList(name string) []string {
	name = graph.ShortenFunctionName(name)
	return allSuffixes(name, sepRE)
}

// allSuffixes returns a list of suffixes (in order of decreasing length)
// found by splitting at re.
func allSuffixes(name string, re *regexp.Regexp) []string {
	seps := re.FindAllStringIndex(name, -1)
	result := make([]string, 0, len(seps)+1)
	result = append(result, name)
	for _, sep := range seps {
		// Suffix starting just after sep
		if sep[1] < len(name) {
			result = append(result, name[sep[1]:])
		}
	}
	return result
}
