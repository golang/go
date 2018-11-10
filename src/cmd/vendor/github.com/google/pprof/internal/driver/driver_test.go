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

package driver

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/proftest"
	"github.com/google/pprof/internal/symbolz"
	"github.com/google/pprof/profile"
)

func TestParse(t *testing.T) {
	// Override weblist command to collect output in buffer
	pprofCommands["weblist"].postProcess = nil

	// Our mockObjTool.Open will always return success, causing
	// driver.locateBinaries to "find" the binaries below in a non-existant
	// directory. As a workaround, point the search path to the fake
	// directory containing out fake binaries.
	savePath := os.Getenv("PPROF_BINARY_PATH")
	os.Setenv("PPROF_BINARY_PATH", "/path/to")
	defer os.Setenv("PPROF_BINARY_PATH", savePath)

	testcase := []struct {
		flags, source string
	}{
		{"text,functions,flat", "cpu"},
		{"tree,addresses,flat,nodecount=4", "cpusmall"},
		{"text,functions,flat", "unknown"},
		{"text,alloc_objects,flat", "heap_alloc"},
		{"text,files,flat", "heap"},
		{"text,inuse_objects,flat", "heap"},
		{"text,lines,cum,hide=line[X3]0", "cpu"},
		{"text,lines,cum,show=[12]00", "cpu"},
		{"topproto,lines,cum,hide=mangled[X3]0", "cpu"},
		{"tree,lines,cum,focus=[24]00", "heap"},
		{"tree,relative_percentages,cum,focus=[24]00", "heap"},
		{"callgrind", "cpu"},
		{"callgrind", "heap"},
		{"dot,functions,flat", "cpu"},
		{"dot,lines,flat,focus=[12]00", "heap"},
		{"dot,addresses,flat,ignore=[X3]002,focus=[X1]000", "contention"},
		{"dot,files,cum", "contention"},
		{"comments", "cpu"},
		{"comments", "heap"},
		{"tags", "cpu"},
		{"tags,tagignore=tag[13],tagfocus=key[12]", "cpu"},
		{"tags", "heap"},
		{"tags,unit=bytes", "heap"},
		{"traces", "cpu"},
		{"dot,alloc_space,flat,focus=[234]00", "heap_alloc"},
		{"dot,alloc_space,flat,hide=line.*1?23?", "heap_alloc"},
		{"dot,inuse_space,flat,tagfocus=1mb:2gb", "heap"},
		{"dot,inuse_space,flat,tagfocus=30kb:,tagignore=1mb:2mb", "heap"},
		{"disasm=line[13],addresses,flat", "cpu"},
		{"peek=line.*01", "cpu"},
		{"weblist=line[13],addresses,flat", "cpu"},
	}

	baseVars := pprofVariables
	defer func() { pprofVariables = baseVars }()
	for _, tc := range testcase {
		// Reset the pprof variables before processing
		pprofVariables = baseVars.makeCopy()

		f := baseFlags()
		f.args = []string{tc.source}

		flags := strings.Split(tc.flags, ",")

		// Skip the output format in the first flag, to output to a proto
		addFlags(&f, flags[1:])

		// Encode profile into a protobuf and decode it again.
		protoTempFile, err := ioutil.TempFile("", "profile_proto")
		if err != nil {
			t.Errorf("cannot create tempfile: %v", err)
		}
		defer protoTempFile.Close()
		f.strings["output"] = protoTempFile.Name()

		if flags[0] == "topproto" {
			f.bools["proto"] = false
			f.bools["topproto"] = true
		}

		// First pprof invocation to save the profile into a profile.proto.
		o1 := setDefaults(nil)
		o1.Flagset = f
		o1.Fetch = testFetcher{}
		o1.Sym = testSymbolizer{}
		if err := PProf(o1); err != nil {
			t.Errorf("%s %q:  %v", tc.source, tc.flags, err)
			continue
		}
		// Reset the pprof variables after the proto invocation
		pprofVariables = baseVars.makeCopy()

		// Read the profile from the encoded protobuf
		outputTempFile, err := ioutil.TempFile("", "profile_output")
		if err != nil {
			t.Errorf("cannot create tempfile: %v", err)
		}
		defer outputTempFile.Close()
		f.strings["output"] = outputTempFile.Name()
		f.args = []string{protoTempFile.Name()}

		var solution string
		// Apply the flags for the second pprof run, and identify name of
		// the file containing expected results
		if flags[0] == "topproto" {
			solution = solutionFilename(tc.source, &f)
			delete(f.bools, "topproto")
			f.bools["text"] = true
		} else {
			delete(f.bools, "proto")
			addFlags(&f, flags[:1])
			solution = solutionFilename(tc.source, &f)
		}

		// Second pprof invocation to read the profile from profile.proto
		// and generate a report.
		o2 := setDefaults(nil)
		o2.Flagset = f
		o2.Sym = testSymbolizeDemangler{}
		o2.Obj = new(mockObjTool)

		if err := PProf(o2); err != nil {
			t.Errorf("%s: %v", tc.source, err)
		}
		b, err := ioutil.ReadFile(outputTempFile.Name())
		if err != nil {
			t.Errorf("Failed to read profile %s: %v", outputTempFile.Name(), err)
		}

		// Read data file with expected solution
		solution = "testdata/" + solution
		sbuf, err := ioutil.ReadFile(solution)
		if err != nil {
			t.Errorf("reading solution file %s: %v", solution, err)
			continue
		}
		if runtime.GOOS == "windows" {
			sbuf = bytes.Replace(sbuf, []byte("testdata/"), []byte("testdata\\"), -1)
			sbuf = bytes.Replace(sbuf, []byte("/path/to/"), []byte("\\path\\to\\"), -1)
		}

		if flags[0] == "svg" {
			b = removeScripts(b)
			sbuf = removeScripts(sbuf)
		}

		if string(b) != string(sbuf) {
			t.Errorf("diff %s %s", solution, tc.source)
			d, err := proftest.Diff(sbuf, b)
			if err != nil {
				t.Fatalf("diff %s %v", solution, err)
			}
			t.Errorf("%s\n%s\n", solution, d)
		}
	}
}

// removeScripts removes <script > .. </script> pairs from its input
func removeScripts(in []byte) []byte {
	beginMarker := []byte("<script")
	endMarker := []byte("</script>")

	if begin := bytes.Index(in, beginMarker); begin > 0 {
		if end := bytes.Index(in[begin:], endMarker); end > 0 {
			in = append(in[:begin], removeScripts(in[begin+end+len(endMarker):])...)
		}
	}
	return in
}

// addFlags parses flag descriptions and adds them to the testFlags
func addFlags(f *testFlags, flags []string) {
	for _, flag := range flags {
		fields := strings.SplitN(flag, "=", 2)
		switch len(fields) {
		case 1:
			f.bools[fields[0]] = true
		case 2:
			if i, err := strconv.Atoi(fields[1]); err == nil {
				f.ints[fields[0]] = i
			} else {
				f.strings[fields[0]] = fields[1]
			}
		}
	}
}

// solutionFilename returns the name of the solution file for the test
func solutionFilename(source string, f *testFlags) string {
	name := []string{"pprof", strings.TrimPrefix(source, "http://host:8000/")}
	name = addString(name, f, []string{"flat", "cum"})
	name = addString(name, f, []string{"functions", "files", "lines", "addresses"})
	name = addString(name, f, []string{"inuse_space", "inuse_objects", "alloc_space", "alloc_objects"})
	name = addString(name, f, []string{"relative_percentages"})
	name = addString(name, f, []string{"seconds"})
	name = addString(name, f, []string{"text", "tree", "callgrind", "dot", "svg", "tags", "dot", "traces", "disasm", "peek", "weblist", "topproto", "comments"})
	if f.strings["focus"] != "" || f.strings["tagfocus"] != "" {
		name = append(name, "focus")
	}
	if f.strings["ignore"] != "" || f.strings["tagignore"] != "" {
		name = append(name, "ignore")
	}
	name = addString(name, f, []string{"hide", "show"})
	if f.strings["unit"] != "minimum" {
		name = addString(name, f, []string{"unit"})
	}
	return strings.Join(name, ".")
}

func addString(name []string, f *testFlags, components []string) []string {
	for _, c := range components {
		if f.bools[c] || f.strings[c] != "" || f.ints[c] != 0 {
			return append(name, c)
		}
	}
	return name
}

// testFlags implements the plugin.FlagSet interface.
type testFlags struct {
	bools   map[string]bool
	ints    map[string]int
	floats  map[string]float64
	strings map[string]string
	args    []string
}

func (testFlags) ExtraUsage() string { return "" }

func (f testFlags) Bool(s string, d bool, c string) *bool {
	if b, ok := f.bools[s]; ok {
		return &b
	}
	return &d
}

func (f testFlags) Int(s string, d int, c string) *int {
	if i, ok := f.ints[s]; ok {
		return &i
	}
	return &d
}

func (f testFlags) Float64(s string, d float64, c string) *float64 {
	if g, ok := f.floats[s]; ok {
		return &g
	}
	return &d
}

func (f testFlags) String(s, d, c string) *string {
	if t, ok := f.strings[s]; ok {
		return &t
	}
	return &d
}

func (f testFlags) BoolVar(p *bool, s string, d bool, c string) {
	if b, ok := f.bools[s]; ok {
		*p = b
	} else {
		*p = d
	}
}

func (f testFlags) IntVar(p *int, s string, d int, c string) {
	if i, ok := f.ints[s]; ok {
		*p = i
	} else {
		*p = d
	}
}

func (f testFlags) Float64Var(p *float64, s string, d float64, c string) {
	if g, ok := f.floats[s]; ok {
		*p = g
	} else {
		*p = d
	}
}

func (f testFlags) StringVar(p *string, s, d, c string) {
	if t, ok := f.strings[s]; ok {
		*p = t
	} else {
		*p = d
	}
}

func (f testFlags) StringList(s, d, c string) *[]*string {
	return &[]*string{}
}

func (f testFlags) Parse(func()) []string {
	return f.args
}

func baseFlags() testFlags {
	return testFlags{
		bools: map[string]bool{
			"proto":          true,
			"trim":           true,
			"compact_labels": true,
		},
		ints: map[string]int{
			"nodecount": 20,
		},
		floats: map[string]float64{
			"nodefraction": 0.05,
			"edgefraction": 0.01,
			"divide_by":    1.0,
		},
		strings: map[string]string{
			"unit": "minimum",
		},
	}
}

type testProfile struct {
}

const testStart = 0x1000
const testOffset = 0x5000

type testFetcher struct{}

func (testFetcher) Fetch(s string, d, t time.Duration) (*profile.Profile, string, error) {
	var p *profile.Profile
	s = strings.TrimPrefix(s, "http://host:8000/")
	switch s {
	case "cpu", "unknown":
		p = cpuProfile()
	case "cpusmall":
		p = cpuProfileSmall()
	case "heap":
		p = heapProfile()
	case "heap_alloc":
		p = heapProfile()
		p.SampleType = []*profile.ValueType{
			{Type: "alloc_objects", Unit: "count"},
			{Type: "alloc_space", Unit: "bytes"},
		}
	case "contention":
		p = contentionProfile()
	case "symbolz":
		p = symzProfile()
	case "http://host2/symbolz":
		p = symzProfile()
		p.Mapping[0].Start += testOffset
		p.Mapping[0].Limit += testOffset
		for i := range p.Location {
			p.Location[i].Address += testOffset
		}
	default:
		return nil, "", fmt.Errorf("unexpected source: %s", s)
	}
	return p, s, nil
}

type testSymbolizer struct{}

func (testSymbolizer) Symbolize(_ string, _ plugin.MappingSources, _ *profile.Profile) error {
	return nil
}

type testSymbolizeDemangler struct{}

func (testSymbolizeDemangler) Symbolize(_ string, _ plugin.MappingSources, p *profile.Profile) error {
	for _, fn := range p.Function {
		if fn.Name == "" || fn.SystemName == fn.Name {
			fn.Name = fakeDemangler(fn.SystemName)
		}
	}
	return nil
}

func testFetchSymbols(source, post string) ([]byte, error) {
	var buf bytes.Buffer

	if source == "http://host2/symbolz" {
		for _, address := range strings.Split(post, "+") {
			a, _ := strconv.ParseInt(address, 0, 64)
			fmt.Fprintf(&buf, "%v\t", address)
			if a-testStart < testOffset {
				fmt.Fprintf(&buf, "wrong_source_%v_", address)
				continue
			}
			fmt.Fprintf(&buf, "%#x\n", a-testStart-testOffset)
		}
		return buf.Bytes(), nil
	}
	for _, address := range strings.Split(post, "+") {
		a, _ := strconv.ParseInt(address, 0, 64)
		fmt.Fprintf(&buf, "%v\t", address)
		if a-testStart > testOffset {
			fmt.Fprintf(&buf, "wrong_source_%v_", address)
			continue
		}
		fmt.Fprintf(&buf, "%#x\n", a-testStart)
	}
	return buf.Bytes(), nil
}

type testSymbolzSymbolizer struct{}

func (testSymbolzSymbolizer) Symbolize(variables string, sources plugin.MappingSources, p *profile.Profile) error {
	return symbolz.Symbolize(sources, testFetchSymbols, p, nil)
}

func fakeDemangler(name string) string {
	switch name {
	case "mangled1000":
		return "line1000"
	case "mangled2000":
		return "line2000"
	case "mangled2001":
		return "line2001"
	case "mangled3000":
		return "line3000"
	case "mangled3001":
		return "line3001"
	case "mangled3002":
		return "line3002"
	case "mangledNEW":
		return "operator new"
	case "mangledMALLOC":
		return "malloc"
	default:
		return name
	}
}

func cpuProfile() *profile.Profile {
	var cpuM = []*profile.Mapping{
		{
			ID:              1,
			Start:           0x1000,
			Limit:           0x4000,
			File:            "/path/to/testbinary",
			HasFunctions:    true,
			HasFilenames:    true,
			HasLineNumbers:  true,
			HasInlineFrames: true,
		},
	}

	var cpuF = []*profile.Function{
		{ID: 1, Name: "mangled1000", SystemName: "mangled1000", Filename: "testdata/file1000.src"},
		{ID: 2, Name: "mangled2000", SystemName: "mangled2000", Filename: "testdata/file2000.src"},
		{ID: 3, Name: "mangled2001", SystemName: "mangled2001", Filename: "testdata/file2000.src"},
		{ID: 4, Name: "mangled3000", SystemName: "mangled3000", Filename: "testdata/file3000.src"},
		{ID: 5, Name: "mangled3001", SystemName: "mangled3001", Filename: "testdata/file3000.src"},
		{ID: 6, Name: "mangled3002", SystemName: "mangled3002", Filename: "testdata/file3000.src"},
	}

	var cpuL = []*profile.Location{
		{
			ID:      1000,
			Mapping: cpuM[0],
			Address: 0x1000,
			Line: []profile.Line{
				{Function: cpuF[0], Line: 1},
			},
		},
		{
			ID:      2000,
			Mapping: cpuM[0],
			Address: 0x2000,
			Line: []profile.Line{
				{Function: cpuF[2], Line: 9},
				{Function: cpuF[1], Line: 4},
			},
		},
		{
			ID:      3000,
			Mapping: cpuM[0],
			Address: 0x3000,
			Line: []profile.Line{
				{Function: cpuF[5], Line: 2},
				{Function: cpuF[4], Line: 5},
				{Function: cpuF[3], Line: 6},
			},
		},
		{
			ID:      3001,
			Mapping: cpuM[0],
			Address: 0x3001,
			Line: []profile.Line{
				{Function: cpuF[4], Line: 8},
				{Function: cpuF[3], Line: 9},
			},
		},
		{
			ID:      3002,
			Mapping: cpuM[0],
			Address: 0x3002,
			Line: []profile.Line{
				{Function: cpuF[5], Line: 5},
				{Function: cpuF[3], Line: 9},
			},
		},
	}

	return &profile.Profile{
		PeriodType:    &profile.ValueType{Type: "cpu", Unit: "milliseconds"},
		Period:        1,
		DurationNanos: 10e9,
		SampleType: []*profile.ValueType{
			{Type: "samples", Unit: "count"},
			{Type: "cpu", Unit: "milliseconds"},
		},
		Sample: []*profile.Sample{
			{
				Location: []*profile.Location{cpuL[0], cpuL[1], cpuL[2]},
				Value:    []int64{1000, 1000},
				Label: map[string][]string{
					"key1": []string{"tag1"},
					"key2": []string{"tag1"},
				},
			},
			{
				Location: []*profile.Location{cpuL[0], cpuL[3]},
				Value:    []int64{100, 100},
				Label: map[string][]string{
					"key1": []string{"tag2"},
					"key3": []string{"tag2"},
				},
			},
			{
				Location: []*profile.Location{cpuL[1], cpuL[4]},
				Value:    []int64{10, 10},
				Label: map[string][]string{
					"key1": []string{"tag3"},
					"key2": []string{"tag2"},
				},
			},
			{
				Location: []*profile.Location{cpuL[2]},
				Value:    []int64{10, 10},
				Label: map[string][]string{
					"key1": []string{"tag4"},
					"key2": []string{"tag1"},
				},
			},
		},
		Location: cpuL,
		Function: cpuF,
		Mapping:  cpuM,
	}
}

func cpuProfileSmall() *profile.Profile {
	var cpuM = []*profile.Mapping{
		{
			ID:              1,
			Start:           0x1000,
			Limit:           0x4000,
			File:            "/path/to/testbinary",
			HasFunctions:    true,
			HasFilenames:    true,
			HasLineNumbers:  true,
			HasInlineFrames: true,
		},
	}

	var cpuL = []*profile.Location{
		{
			ID:      1000,
			Mapping: cpuM[0],
			Address: 0x1000,
		},
		{
			ID:      2000,
			Mapping: cpuM[0],
			Address: 0x2000,
		},
		{
			ID:      3000,
			Mapping: cpuM[0],
			Address: 0x3000,
		},
		{
			ID:      4000,
			Mapping: cpuM[0],
			Address: 0x4000,
		},
		{
			ID:      5000,
			Mapping: cpuM[0],
			Address: 0x5000,
		},
	}

	return &profile.Profile{
		PeriodType:    &profile.ValueType{Type: "cpu", Unit: "milliseconds"},
		Period:        1,
		DurationNanos: 10e9,
		SampleType: []*profile.ValueType{
			{Type: "samples", Unit: "count"},
			{Type: "cpu", Unit: "milliseconds"},
		},
		Sample: []*profile.Sample{
			{
				Location: []*profile.Location{cpuL[0], cpuL[1], cpuL[2]},
				Value:    []int64{1000, 1000},
			},
			{
				Location: []*profile.Location{cpuL[3], cpuL[1], cpuL[4]},
				Value:    []int64{1000, 1000},
			},
			{
				Location: []*profile.Location{cpuL[2]},
				Value:    []int64{1000, 1000},
			},
			{
				Location: []*profile.Location{cpuL[4]},
				Value:    []int64{1000, 1000},
			},
		},
		Location: cpuL,
		Function: nil,
		Mapping:  cpuM,
	}
}

func heapProfile() *profile.Profile {
	var heapM = []*profile.Mapping{
		{
			ID:              1,
			BuildID:         "buildid",
			Start:           0x1000,
			Limit:           0x4000,
			HasFunctions:    true,
			HasFilenames:    true,
			HasLineNumbers:  true,
			HasInlineFrames: true,
		},
	}

	var heapF = []*profile.Function{
		{ID: 1, Name: "pruneme", SystemName: "pruneme", Filename: "prune.h"},
		{ID: 2, Name: "mangled1000", SystemName: "mangled1000", Filename: "testdata/file1000.src"},
		{ID: 3, Name: "mangled2000", SystemName: "mangled2000", Filename: "testdata/file2000.src"},
		{ID: 4, Name: "mangled2001", SystemName: "mangled2001", Filename: "testdata/file2000.src"},
		{ID: 5, Name: "mangled3000", SystemName: "mangled3000", Filename: "testdata/file3000.src"},
		{ID: 6, Name: "mangled3001", SystemName: "mangled3001", Filename: "testdata/file3000.src"},
		{ID: 7, Name: "mangled3002", SystemName: "mangled3002", Filename: "testdata/file3000.src"},
		{ID: 8, Name: "mangledMALLOC", SystemName: "mangledMALLOC", Filename: "malloc.h"},
		{ID: 9, Name: "mangledNEW", SystemName: "mangledNEW", Filename: "new.h"},
	}

	var heapL = []*profile.Location{
		{
			ID:      1000,
			Mapping: heapM[0],
			Address: 0x1000,
			Line: []profile.Line{
				{Function: heapF[0], Line: 100},
				{Function: heapF[7], Line: 100},
				{Function: heapF[1], Line: 1},
			},
		},
		{
			ID:      2000,
			Mapping: heapM[0],
			Address: 0x2000,
			Line: []profile.Line{
				{Function: heapF[8], Line: 100},
				{Function: heapF[3], Line: 2},
				{Function: heapF[2], Line: 3},
			},
		},
		{
			ID:      3000,
			Mapping: heapM[0],
			Address: 0x3000,
			Line: []profile.Line{
				{Function: heapF[8], Line: 100},
				{Function: heapF[6], Line: 3},
				{Function: heapF[5], Line: 2},
				{Function: heapF[4], Line: 4},
			},
		},
		{
			ID:      3001,
			Mapping: heapM[0],
			Address: 0x3001,
			Line: []profile.Line{
				{Function: heapF[0], Line: 100},
				{Function: heapF[8], Line: 100},
				{Function: heapF[5], Line: 2},
				{Function: heapF[4], Line: 4},
			},
		},
		{
			ID:      3002,
			Mapping: heapM[0],
			Address: 0x3002,
			Line: []profile.Line{
				{Function: heapF[6], Line: 3},
				{Function: heapF[4], Line: 4},
			},
		},
	}

	return &profile.Profile{
		Comments:   []string{"comment", "#hidden comment"},
		PeriodType: &profile.ValueType{Type: "allocations", Unit: "bytes"},
		Period:     524288,
		SampleType: []*profile.ValueType{
			{Type: "inuse_objects", Unit: "count"},
			{Type: "inuse_space", Unit: "bytes"},
		},
		Sample: []*profile.Sample{
			{
				Location: []*profile.Location{heapL[0], heapL[1], heapL[2]},
				Value:    []int64{10, 1024000},
				NumLabel: map[string][]int64{
					"bytes": []int64{102400},
				},
			},
			{
				Location: []*profile.Location{heapL[0], heapL[3]},
				Value:    []int64{20, 4096000},
				NumLabel: map[string][]int64{
					"bytes": []int64{204800},
				},
			},
			{
				Location: []*profile.Location{heapL[1], heapL[4]},
				Value:    []int64{40, 65536000},
				NumLabel: map[string][]int64{
					"bytes": []int64{1638400},
				},
			},
			{
				Location: []*profile.Location{heapL[2]},
				Value:    []int64{80, 32768000},
				NumLabel: map[string][]int64{
					"bytes": []int64{409600},
				},
			},
		},
		DropFrames: ".*operator new.*|malloc",
		Location:   heapL,
		Function:   heapF,
		Mapping:    heapM,
	}
}

func contentionProfile() *profile.Profile {
	var contentionM = []*profile.Mapping{
		{
			ID:              1,
			BuildID:         "buildid-contention",
			Start:           0x1000,
			Limit:           0x4000,
			HasFunctions:    true,
			HasFilenames:    true,
			HasLineNumbers:  true,
			HasInlineFrames: true,
		},
	}

	var contentionF = []*profile.Function{
		{ID: 1, Name: "mangled1000", SystemName: "mangled1000", Filename: "testdata/file1000.src"},
		{ID: 2, Name: "mangled2000", SystemName: "mangled2000", Filename: "testdata/file2000.src"},
		{ID: 3, Name: "mangled2001", SystemName: "mangled2001", Filename: "testdata/file2000.src"},
		{ID: 4, Name: "mangled3000", SystemName: "mangled3000", Filename: "testdata/file3000.src"},
		{ID: 5, Name: "mangled3001", SystemName: "mangled3001", Filename: "testdata/file3000.src"},
		{ID: 6, Name: "mangled3002", SystemName: "mangled3002", Filename: "testdata/file3000.src"},
	}

	var contentionL = []*profile.Location{
		{
			ID:      1000,
			Mapping: contentionM[0],
			Address: 0x1000,
			Line: []profile.Line{
				{Function: contentionF[0], Line: 1},
			},
		},
		{
			ID:      2000,
			Mapping: contentionM[0],
			Address: 0x2000,
			Line: []profile.Line{
				{Function: contentionF[2], Line: 2},
				{Function: contentionF[1], Line: 3},
			},
		},
		{
			ID:      3000,
			Mapping: contentionM[0],
			Address: 0x3000,
			Line: []profile.Line{
				{Function: contentionF[5], Line: 2},
				{Function: contentionF[4], Line: 3},
				{Function: contentionF[3], Line: 5},
			},
		},
		{
			ID:      3001,
			Mapping: contentionM[0],
			Address: 0x3001,
			Line: []profile.Line{
				{Function: contentionF[4], Line: 3},
				{Function: contentionF[3], Line: 5},
			},
		},
		{
			ID:      3002,
			Mapping: contentionM[0],
			Address: 0x3002,
			Line: []profile.Line{
				{Function: contentionF[5], Line: 4},
				{Function: contentionF[3], Line: 3},
			},
		},
	}

	return &profile.Profile{
		PeriodType: &profile.ValueType{Type: "contentions", Unit: "count"},
		Period:     524288,
		SampleType: []*profile.ValueType{
			{Type: "contentions", Unit: "count"},
			{Type: "delay", Unit: "nanoseconds"},
		},
		Sample: []*profile.Sample{
			{
				Location: []*profile.Location{contentionL[0], contentionL[1], contentionL[2]},
				Value:    []int64{10, 10240000},
			},
			{
				Location: []*profile.Location{contentionL[0], contentionL[3]},
				Value:    []int64{20, 40960000},
			},
			{
				Location: []*profile.Location{contentionL[1], contentionL[4]},
				Value:    []int64{40, 65536000},
			},
			{
				Location: []*profile.Location{contentionL[2]},
				Value:    []int64{80, 32768000},
			},
		},
		Location: contentionL,
		Function: contentionF,
		Mapping:  contentionM,
		Comments: []string{"Comment #1", "Comment #2"},
	}
}

func symzProfile() *profile.Profile {
	var symzM = []*profile.Mapping{
		{
			ID:    1,
			Start: testStart,
			Limit: 0x4000,
			File:  "/path/to/testbinary",
		},
	}

	var symzL = []*profile.Location{
		{ID: 1, Mapping: symzM[0], Address: testStart},
		{ID: 2, Mapping: symzM[0], Address: testStart + 0x1000},
		{ID: 3, Mapping: symzM[0], Address: testStart + 0x2000},
	}

	return &profile.Profile{
		PeriodType:    &profile.ValueType{Type: "cpu", Unit: "milliseconds"},
		Period:        1,
		DurationNanos: 10e9,
		SampleType: []*profile.ValueType{
			{Type: "samples", Unit: "count"},
			{Type: "cpu", Unit: "milliseconds"},
		},
		Sample: []*profile.Sample{
			{
				Location: []*profile.Location{symzL[0], symzL[1], symzL[2]},
				Value:    []int64{1, 1},
			},
		},
		Location: symzL,
		Mapping:  symzM,
	}
}

var autoCompleteTests = []struct {
	in  string
	out string
}{
	{"", ""},
	{"xyz", "xyz"},                        // no match
	{"dis", "disasm"},                     // single match
	{"t", "t"},                            // many matches
	{"top abc", "top abc"},                // no function name match
	{"top mangledM", "top mangledMALLOC"}, // single function name match
	{"top cmd cmd mangledM", "top cmd cmd mangledMALLOC"},
	{"top mangled", "top mangled"},                      // many function name matches
	{"cmd mangledM", "cmd mangledM"},                    // invalid command
	{"top mangledM cmd", "top mangledM cmd"},            // cursor misplaced
	{"top edMA", "top mangledMALLOC"},                   // single infix function name match
	{"top -mangledM", "top -mangledMALLOC"},             // ignore sign handled
	{"lin", "lines"},                                    // single variable match
	{"EdGeF", "edgefraction"},                           // single capitalized match
	{"help dis", "help disasm"},                         // help command match
	{"help relative_perc", "help relative_percentages"}, // help variable match
	{"help coMpa", "help compact_labels"},               // help variable capitalized match
}

func TestAutoComplete(t *testing.T) {
	complete := newCompleter(functionNames(heapProfile()))

	for _, test := range autoCompleteTests {
		if out := complete(test.in); out != test.out {
			t.Errorf("autoComplete(%s) = %s; want %s", test.in, out, test.out)
		}
	}
}

func TestTagFilter(t *testing.T) {
	var tagFilterTests = []struct {
		name, value string
		tags        map[string][]string
		want        bool
	}{
		{"test1", "tag2", map[string][]string{"value1": {"tag1", "tag2"}}, true},
		{"test2", "tag3", map[string][]string{"value1": {"tag1", "tag2"}}, false},
		{"test3", "tag1,tag3", map[string][]string{"value1": {"tag1", "tag2"}, "value2": {"tag3"}}, true},
		{"test4", "t..[12],t..3", map[string][]string{"value1": {"tag1", "tag2"}, "value2": {"tag3"}}, true},
		{"test5", "tag2,tag3", map[string][]string{"value1": {"tag1", "tag2"}}, false},
	}

	for _, test := range tagFilterTests {
		filter, err := compileTagFilter(test.name, test.value, &proftest.TestUI{T: t}, nil)
		if err != nil {
			t.Errorf("tagFilter %s:%v", test.name, err)
			continue
		}
		s := profile.Sample{
			Label: test.tags,
		}

		if got := filter(&s); got != test.want {
			t.Errorf("tagFilter %s: got %v, want %v", test.name, got, test.want)
		}
	}
}

func TestSymbolzAfterMerge(t *testing.T) {
	baseVars := pprofVariables
	pprofVariables = baseVars.makeCopy()
	defer func() { pprofVariables = baseVars }()

	f := baseFlags()
	f.args = []string{"symbolz", "http://host2/symbolz"}

	o := setDefaults(nil)
	o.Flagset = f
	o.Obj = new(mockObjTool)
	src, cmd, err := parseFlags(o)
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}

	if len(cmd) != 1 || cmd[0] != "proto" {
		t.Fatalf("parseFlags returned command %v, want [proto]", cmd)
	}

	o.Fetch = testFetcher{}
	o.Sym = testSymbolzSymbolizer{}
	p, err := fetchProfiles(src, o)
	if err != nil {
		t.Fatalf("fetchProfiles: %v", err)
	}
	if len(p.Location) != 3 {
		t.Errorf("Got %d locations after merge, want %d", len(p.Location), 3)
	}
	for i, l := range p.Location {
		if len(l.Line) != 1 {
			t.Errorf("Number of lines for symbolz %#x in iteration %d, got %d, want %d", l.Address, i, len(l.Line), 1)
			continue
		}
		address := l.Address - l.Mapping.Start
		if got, want := l.Line[0].Function.Name, fmt.Sprintf("%#x", address); got != want {
			t.Errorf("symbolz %#x, got %s, want %s", address, got, want)
		}
	}
}

type mockObjTool struct{}

func (*mockObjTool) Open(file string, start, limit, offset uint64) (plugin.ObjFile, error) {
	return &mockFile{file, "abcdef", 0}, nil
}

func (m *mockObjTool) Disasm(file string, start, end uint64) ([]plugin.Inst, error) {
	switch start {
	case 0x1000:
		return []plugin.Inst{
			{Addr: 0x1000, Text: "instruction one"},
			{Addr: 0x1001, Text: "instruction two"},
			{Addr: 0x1002, Text: "instruction three"},
			{Addr: 0x1003, Text: "instruction four"},
		}, nil
	case 0x3000:
		return []plugin.Inst{
			{Addr: 0x3000, Text: "instruction one"},
			{Addr: 0x3001, Text: "instruction two"},
			{Addr: 0x3002, Text: "instruction three"},
			{Addr: 0x3003, Text: "instruction four"},
			{Addr: 0x3004, Text: "instruction five"},
		}, nil
	}
	return nil, fmt.Errorf("unimplemented")
}

type mockFile struct {
	name, buildId string
	base          uint64
}

// Name returns the underlyinf file name, if available
func (m *mockFile) Name() string {
	return m.name
}

// Base returns the base address to use when looking up symbols in the file.
func (m *mockFile) Base() uint64 {
	return m.base
}

// BuildID returns the GNU build ID of the file, or an empty string.
func (m *mockFile) BuildID() string {
	return m.buildId
}

// SourceLine reports the source line information for a given
// address in the file. Due to inlining, the source line information
// is in general a list of positions representing a call stack,
// with the leaf function first.
func (*mockFile) SourceLine(addr uint64) ([]plugin.Frame, error) {
	return nil, fmt.Errorf("unimplemented")
}

// Symbols returns a list of symbols in the object file.
// If r is not nil, Symbols restricts the list to symbols
// with names matching the regular expression.
// If addr is not zero, Symbols restricts the list to symbols
// containing that address.
func (m *mockFile) Symbols(r *regexp.Regexp, addr uint64) ([]*plugin.Sym, error) {
	switch r.String() {
	case "line[13]":
		return []*plugin.Sym{
			{[]string{"line1000"}, m.name, 0x1000, 0x1003},
			{[]string{"line3000"}, m.name, 0x3000, 0x3004},
		}, nil
	}
	return nil, fmt.Errorf("unimplemented")
}

// Close closes the file, releasing associated resources.
func (*mockFile) Close() error {
	return nil
}
