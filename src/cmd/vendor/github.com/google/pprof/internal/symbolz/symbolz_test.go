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
	"math"
	"strings"
	"testing"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/proftest"
	"github.com/google/pprof/profile"
)

func TestSymbolzURL(t *testing.T) {
	for try, want := range map[string]string{
		"http://host:8000/profilez":                                               "http://host:8000/symbolz",
		"http://host:8000/profilez?seconds=5":                                     "http://host:8000/symbolz",
		"http://host:8000/profilez?seconds=5&format=proto":                        "http://host:8000/symbolz",
		"http://host:8000/heapz?format=legacy":                                    "http://host:8000/symbolz",
		"http://host:8000/debug/pprof/profile":                                    "http://host:8000/debug/pprof/symbol",
		"http://host:8000/debug/pprof/profile?seconds=10":                         "http://host:8000/debug/pprof/symbol",
		"http://host:8000/debug/pprof/heap":                                       "http://host:8000/debug/pprof/symbol",
		"http://some.host:8080/some/deeper/path/debug/pprof/endpoint?param=value": "http://some.host:8080/some/deeper/path/debug/pprof/symbol",
		"http://host:8000/pprof/profile":                                          "http://host:8000/pprof/symbol",
		"http://host:8000/pprof/profile?seconds=15":                               "http://host:8000/pprof/symbol",
		"http://host:8000/pprof/heap":                                             "http://host:8000/pprof/symbol",
		"http://host:8000/debug/pprof/block":                                      "http://host:8000/debug/pprof/symbol",
		"http://host:8000/debug/pprof/trace?seconds=5":                            "http://host:8000/debug/pprof/symbol",
		"http://host:8000/debug/pprof/mutex":                                      "http://host:8000/debug/pprof/symbol",
		"http://host/whatever/pprof/heap":                                         "http://host/whatever/pprof/symbol",
		"http://host/whatever/pprof/growth":                                       "http://host/whatever/pprof/symbol",
		"http://host/whatever/pprof/profile":                                      "http://host/whatever/pprof/symbol",
		"http://host/whatever/pprof/pmuprofile":                                   "http://host/whatever/pprof/symbol",
		"http://host/whatever/pprof/contention":                                   "http://host/whatever/pprof/symbol",
	} {
		if got := symbolz(try); got != want {
			t.Errorf(`symbolz(%s)=%s, want "%s"`, try, got, want)
		}
	}
}

func TestSymbolize(t *testing.T) {
	s := plugin.MappingSources{
		"buildid": []struct {
			Source string
			Start  uint64
		}{
			{Source: "http://localhost:80/profilez"},
		},
	}

	for _, hasFunctions := range []bool{false, true} {
		for _, force := range []bool{false, true} {
			p := testProfile(hasFunctions)

			if err := Symbolize(p, force, s, fetchSymbols, &proftest.TestUI{T: t}); err != nil {
				t.Errorf("symbolz: %v", err)
				continue
			}
			var wantSym, wantNoSym []*profile.Location
			if force || !hasFunctions {
				wantNoSym = p.Location[:1]
				wantSym = p.Location[1:]
			} else {
				wantNoSym = p.Location
			}

			if err := checkSymbolized(wantSym, true); err != nil {
				t.Errorf("symbolz hasFns=%v force=%v: %v", hasFunctions, force, err)
			}
			if err := checkSymbolized(wantNoSym, false); err != nil {
				t.Errorf("symbolz hasFns=%v force=%v: %v", hasFunctions, force, err)
			}
		}
	}
}

func testProfile(hasFunctions bool) *profile.Profile {
	m := []*profile.Mapping{
		{
			ID:           1,
			Start:        0x1000,
			Limit:        0x5000,
			BuildID:      "buildid",
			HasFunctions: hasFunctions,
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

	return p
}

func checkSymbolized(locs []*profile.Location, wantSymbolized bool) error {
	for _, loc := range locs {
		if !wantSymbolized && len(loc.Line) != 0 {
			return fmt.Errorf("unexpected symbolization for %#x: %v", loc.Address, loc.Line)
		}
		if wantSymbolized {
			if len(loc.Line) != 1 {
				return fmt.Errorf("expected symbolization for %#x: %v", loc.Address, loc.Line)
			}
			address := loc.Address - loc.Mapping.Start
			if got, want := loc.Line[0].Function.Name, fmt.Sprintf("%#x", address); got != want {
				return fmt.Errorf("symbolz %#x, got %s, want %s", address, got, want)
			}
		}
	}
	return nil
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

func TestAdjust(t *testing.T) {
	for _, tc := range []struct {
		addr         uint64
		offset       int64
		wantAdj      uint64
		wantOverflow bool
	}{{math.MaxUint64, 0, math.MaxUint64, false},
		{math.MaxUint64, 1, 0, true},
		{math.MaxUint64 - 1, 1, math.MaxUint64, false},
		{math.MaxUint64 - 1, 2, 0, true},
		{math.MaxInt64 + 1, math.MaxInt64, math.MaxUint64, false},
		{0, 0, 0, false},
		{0, -1, 0, true},
		{1, -1, 0, false},
		{2, -1, 1, false},
		{2, -2, 0, false},
		{2, -3, 0, true},
		{-math.MinInt64, math.MinInt64, 0, false},
		{-math.MinInt64 + 1, math.MinInt64, 1, false},
		{-math.MinInt64 - 1, math.MinInt64, 0, true},
	} {
		if adj, overflow := adjust(tc.addr, tc.offset); adj != tc.wantAdj || overflow != tc.wantOverflow {
			t.Errorf("adjust(%d, %d) = (%d, %t), want (%d, %t)", tc.addr, tc.offset, adj, overflow, tc.wantAdj, tc.wantOverflow)
		}
	}
}
