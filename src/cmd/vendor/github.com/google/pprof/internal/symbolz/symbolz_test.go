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

package symbolz

import (
	"fmt"
	"strings"
	"testing"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/proftest"
	"github.com/google/pprof/profile"
)

func TestSymbolzURL(t *testing.T) {
	for try, want := range map[string]string{
		"http://host:8000/profilez":                        "http://host:8000/symbolz",
		"http://host:8000/profilez?seconds=5":              "http://host:8000/symbolz",
		"http://host:8000/profilez?seconds=5&format=proto": "http://host:8000/symbolz",
		"http://host:8000/heapz?format=legacy":             "http://host:8000/symbolz",
		"http://host:8000/debug/pprof/profile":             "http://host:8000/debug/pprof/symbol",
		"http://host:8000/debug/pprof/profile?seconds=10":  "http://host:8000/debug/pprof/symbol",
		"http://host:8000/debug/pprof/heap":                "http://host:8000/debug/pprof/symbol",
	} {
		if got := symbolz(try); got != want {
			t.Errorf(`symbolz(%s)=%s, want "%s"`, try, got, want)
		}
	}
}

func TestSymbolize(t *testing.T) {
	m := []*profile.Mapping{
		{
			ID:      1,
			Start:   0x1000,
			Limit:   0x5000,
			BuildID: "buildid",
		},
	}
	p := &profile.Profile{
		Location: []*profile.Location{
			{ID: 1, Mapping: m[0], Address: 0x1000},
			{ID: 2, Mapping: m[0], Address: 0x2000},
			{ID: 3, Mapping: m[0], Address: 0x3000},
			{ID: 4, Mapping: m[0], Address: 0x4000},
		},
		Mapping: m,
	}

	s := plugin.MappingSources{
		"buildid": []struct {
			Source string
			Start  uint64
		}{
			{Source: "http://localhost:80/profilez"},
		},
	}

	if err := Symbolize(s, fetchSymbols, p, &proftest.TestUI{T: t}); err != nil {
		t.Errorf("symbolz: %v", err)
	}

	if l := p.Location[0]; len(l.Line) != 0 {
		t.Errorf("unexpected symbolization for %#x: %v", l.Address, l.Line)
	}

	for _, l := range p.Location[1:] {
		if len(l.Line) != 1 {
			t.Errorf("failed to symbolize %#x", l.Address)
			continue
		}
		address := l.Address - l.Mapping.Start
		if got, want := l.Line[0].Function.Name, fmt.Sprintf("%#x", address); got != want {
			t.Errorf("symbolz %#x, got %s, want %s", address, got, want)
		}
	}
}

func fetchSymbols(source, post string) ([]byte, error) {
	var symbolz string

	addresses := strings.Split(post, "+")
	// Do not symbolize the first symbol.
	for _, address := range addresses[1:] {
		symbolz += fmt.Sprintf("%s\t%s\n", address, address)
	}
	return []byte(symbolz), nil
}
