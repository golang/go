// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu_test

import (
	"errors"
	. "internal/cpu"
	"os"
	"regexp"
	"testing"
)

func getFeatureList() ([]string, error) {
	cpuinfo, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return nil, err
	}
	r := regexp.MustCompile("features\\s*:\\s*(.*)")
	b := r.FindSubmatch(cpuinfo)
	if len(b) < 2 {
		return nil, errors.New("no feature list in /proc/cpuinfo")
	}
	return regexp.MustCompile("\\s+").Split(string(b[1]), -1), nil
}

func TestS390XAgainstCPUInfo(t *testing.T) {
	// mapping of linux feature strings to S390X fields
	mapping := make(map[string]*bool)
	for _, option := range Options {
		mapping[option.Name] = option.Feature
	}

	// these must be true on the machines Go supports
	mandatory := make(map[string]bool)
	mandatory["zarch"] = false
	mandatory["eimm"] = false
	mandatory["ldisp"] = false
	mandatory["stfle"] = false

	features, err := getFeatureList()
	if err != nil {
		t.Error(err)
	}
	for _, feature := range features {
		if _, ok := mandatory[feature]; ok {
			mandatory[feature] = true
		}
		if flag, ok := mapping[feature]; ok {
			if !*flag {
				t.Errorf("feature '%v' not detected", feature)
			}
		} else {
			t.Logf("no entry for '%v'", feature)
		}
	}
	for k, v := range mandatory {
		if !v {
			t.Errorf("mandatory feature '%v' not detected", k)
		}
	}
}
