// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package genflags

import (
	"bytes"
	"cmd/go/internal/base"
	"encoding/json"
	"fmt"
	exec "internal/execabs"
	"regexp"
	"sort"
)

// VetAnalyzers computes analyzers and their aliases supported by vet.
func VetAnalyzers() ([]string, error) {
	// get supported vet flag information
	tool := base.Tool("vet")
	vetcmd := exec.Command(tool, "-flags")
	out := new(bytes.Buffer)
	vetcmd.Stdout = out
	if err := vetcmd.Run(); err != nil {
		return nil, fmt.Errorf("go vet: can't execute %s -flags: %v\n", tool, err)
	}
	var analysisFlags []struct {
		Name  string
		Bool  bool
		Usage string
	}
	if err := json.Unmarshal(out.Bytes(), &analysisFlags); err != nil {
		return nil, fmt.Errorf("go vet: can't unmarshal JSON from %s -flags: %v", tool, err)
	}

	// parse the flags to figure out which ones stand for analyses
	analyzerSet := make(map[string]bool)
	rEnable := regexp.MustCompile("^enable .+ analysis$")
	for _, flag := range analysisFlags {
		if rEnable.MatchString(flag.Usage) {
			analyzerSet[flag.Name] = true
		}
	}

	rDeprecated := regexp.MustCompile("^deprecated alias for -(?P<analyzer>(.+))$")
	// Returns the original value matched by rDeprecated on input value.
	// If there is no match, "" is returned.
	originalValue := func(value string) string {
		match := rDeprecated.FindStringSubmatch(value)
		if len(match) < 2 {
			return ""
		}
		return match[1]
	}
	// extract deprecated aliases for existing analyses
	for _, flag := range analysisFlags {
		if o := originalValue(flag.Usage); analyzerSet[o] {
			analyzerSet[flag.Name] = true
		}
	}

	var analyzers []string
	for a := range analyzerSet {
		analyzers = append(analyzers, a)
	}
	sort.Strings(analyzers)
	return analyzers, nil
}
