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

package symbolizer

import (
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/proftest"
	"github.com/google/pprof/profile"
)

var testM = []*profile.Mapping{
	{
		ID:    1,
		Start: 0x1000,
		Limit: 0x5000,
		File:  "mapping",
	},
}

var testL = []*profile.Location{
	{
		ID:      1,
		Mapping: testM[0],
		Address: 1000,
	},
	{
		ID:      2,
		Mapping: testM[0],
		Address: 2000,
	},
	{
		ID:      3,
		Mapping: testM[0],
		Address: 3000,
	},
	{
		ID:      4,
		Mapping: testM[0],
		Address: 4000,
	},
	{
		ID:      5,
		Mapping: testM[0],
		Address: 5000,
	},
}

var testProfile = profile.Profile{
	DurationNanos: 10e9,
	SampleType: []*profile.ValueType{
		{Type: "cpu", Unit: "cycles"},
	},
	Sample: []*profile.Sample{
		{
			Location: []*profile.Location{testL[0]},
			Value:    []int64{1},
		},
		{
			Location: []*profile.Location{testL[1], testL[0]},
			Value:    []int64{10},
		},
		{
			Location: []*profile.Location{testL[2], testL[0]},
			Value:    []int64{100},
		},
		{
			Location: []*profile.Location{testL[3], testL[0]},
			Value:    []int64{1},
		},
		{
			Location: []*profile.Location{testL[4], testL[3], testL[0]},
			Value:    []int64{10000},
		},
	},
	Location:   testL,
	Mapping:    testM,
	PeriodType: &profile.ValueType{Type: "cpu", Unit: "milliseconds"},
	Period:     10,
}

func TestSymbolization(t *testing.T) {
	sSym := symbolzSymbolize
	lSym := localSymbolize
	defer func() {
		symbolzSymbolize = sSym
		localSymbolize = lSym
	}()
	symbolzSymbolize = symbolzMock
	localSymbolize = localMock

	type testcase struct {
		mode        string
		wantComment string
	}

	s := Symbolizer{
		mockObjTool{},
		&proftest.TestUI{T: t},
	}
	for i, tc := range []testcase{
		{
			"local",
			"local=local",
		},
		{
			"fastlocal",
			"local=fastlocal",
		},
		{
			"remote",
			"symbolz",
		},
		{
			"",
			"local=:symbolz",
		},
	} {
		prof := testProfile.Copy()
		if err := s.Symbolize(tc.mode, nil, prof); err != nil {
			t.Errorf("symbolize #%d: %v", i, err)
			continue
		}
		if got, want := strings.Join(prof.Comments, ":"), tc.wantComment; got != want {
			t.Errorf("got %s, want %s", got, want)
			continue
		}
	}
}

func symbolzMock(sources plugin.MappingSources, syms func(string, string) ([]byte, error), p *profile.Profile, ui plugin.UI) error {
	p.Comments = append(p.Comments, "symbolz")
	return nil
}

func localMock(mode string, p *profile.Profile, obj plugin.ObjTool, ui plugin.UI) error {
	p.Comments = append(p.Comments, "local="+mode)
	return nil
}

func TestLocalSymbolization(t *testing.T) {
	prof := testProfile.Copy()

	if prof.HasFunctions() {
		t.Error("unexpected function names")
	}
	if prof.HasFileLines() {
		t.Error("unexpected filenames or line numbers")
	}

	b := mockObjTool{}
	if err := localSymbolize("", prof, b, &proftest.TestUI{T: t}); err != nil {
		t.Fatalf("localSymbolize(): %v", err)
	}

	for _, loc := range prof.Location {
		if err := checkSymbolizedLocation(loc.Address, loc.Line); err != nil {
			t.Errorf("location %d: %v", loc.Address, err)
		}
	}
	if !prof.HasFunctions() {
		t.Error("missing function names")
	}
	if !prof.HasFileLines() {
		t.Error("missing filenames or line numbers")
	}
}

func checkSymbolizedLocation(a uint64, got []profile.Line) error {
	want, ok := mockAddresses[a]
	if !ok {
		return fmt.Errorf("unexpected address")
	}
	if len(want) != len(got) {
		return fmt.Errorf("want len %d, got %d", len(want), len(got))
	}

	for i, w := range want {
		g := got[i]
		if g.Function.Name != w.Func {
			return fmt.Errorf("want function: %q, got %q", w.Func, g.Function.Name)
		}
		if g.Function.Filename != w.File {
			return fmt.Errorf("want filename: %q, got %q", w.File, g.Function.Filename)
		}
		if g.Line != int64(w.Line) {
			return fmt.Errorf("want lineno: %d, got %d", w.Line, g.Line)
		}
	}
	return nil
}

var mockAddresses = map[uint64][]plugin.Frame{
	1000: []plugin.Frame{frame("fun11", "file11.src", 10)},
	2000: []plugin.Frame{frame("fun21", "file21.src", 20), frame("fun22", "file22.src", 20)},
	3000: []plugin.Frame{frame("fun31", "file31.src", 30), frame("fun32", "file32.src", 30), frame("fun33", "file33.src", 30)},
	4000: []plugin.Frame{frame("fun41", "file41.src", 40), frame("fun42", "file42.src", 40), frame("fun43", "file43.src", 40), frame("fun44", "file44.src", 40)},
	5000: []plugin.Frame{frame("fun51", "file51.src", 50), frame("fun52", "file52.src", 50), frame("fun53", "file53.src", 50), frame("fun54", "file54.src", 50), frame("fun55", "file55.src", 50)},
}

func frame(fname, file string, line int) plugin.Frame {
	return plugin.Frame{
		Func: fname,
		File: file,
		Line: line}
}

type mockObjTool struct{}

func (mockObjTool) Open(file string, start, limit, offset uint64) (plugin.ObjFile, error) {
	return mockObjFile{frames: mockAddresses}, nil
}

func (mockObjTool) Disasm(file string, start, end uint64) ([]plugin.Inst, error) {
	return nil, fmt.Errorf("disassembly not supported")
}

type mockObjFile struct {
	frames map[uint64][]plugin.Frame
}

func (mockObjFile) Name() string {
	return ""
}

func (mockObjFile) Base() uint64 {
	return 0
}

func (mockObjFile) BuildID() string {
	return ""
}

func (mf mockObjFile) SourceLine(addr uint64) ([]plugin.Frame, error) {
	return mf.frames[addr], nil
}

func (mockObjFile) Symbols(r *regexp.Regexp, addr uint64) ([]*plugin.Sym, error) {
	return []*plugin.Sym{}, nil
}

func (mockObjFile) Close() error {
	return nil
}
