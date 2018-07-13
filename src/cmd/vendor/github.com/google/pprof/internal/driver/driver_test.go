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
	"flag"
	"fmt"
	"io/ioutil"
	"net"
	_ "net/http/pprof"
	"os"
	"reflect"
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

var updateFlag = flag.Bool("update", false, "Update the golden files")

func TestParse(t *testing.T) {
	// Override weblist command to collect output in buffer
	pprofCommands["weblist"].postProcess = nil

	// Our mockObjTool.Open will always return success, causing
	// driver.locateBinaries to "find" the binaries below in a non-existent
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
		{"text,functions,flat,nodecount=5,call_tree", "unknown"},
		{"text,alloc_objects,flat", "heap_alloc"},
		{"text,files,flat", "heap"},
		{"text,files,flat,focus=[12]00,taghide=[X3]00", "heap"},
		{"text,inuse_objects,flat", "heap"},
		{"text,lines,cum,hide=line[X3]0", "cpu"},
		{"text,lines,cum,show=[12]00", "cpu"},
		{"text,lines,cum,hide=line[X3]0,focus=[12]00", "cpu"},
		{"topproto,lines,cum,hide=mangled[X3]0", "cpu"},
		{"tree,lines,cum,focus=[24]00", "heap"},
		{"tree,relative_percentages,cum,focus=[24]00", "heap"},
		{"tree,lines,cum,show_from=line2", "cpu"},
		{"callgrind", "cpu"},
		{"callgrind,call_tree", "cpu"},
		{"callgrind", "heap"},
		{"dot,functions,flat", "cpu"},
		{"dot,functions,flat,call_tree", "cpu"},
		{"dot,lines,flat,focus=[12]00", "heap"},
		{"dot,unit=minimum", "heap_sizetags"},
		{"dot,addresses,flat,ignore=[X3]002,focus=[X1]000", "contention"},
		{"dot,files,cum", "contention"},
		{"comments,add_comment=some-comment", "cpu"},
		{"comments", "heap"},
		{"tags", "cpu"},
		{"tags,tagignore=tag[13],tagfocus=key[12]", "cpu"},
		{"tags", "heap"},
		{"tags,unit=bytes", "heap"},
		{"traces", "cpu"},
		{"traces", "heap_tags"},
		{"dot,alloc_space,flat,focus=[234]00", "heap_alloc"},
		{"dot,alloc_space,flat,tagshow=[2]00", "heap_alloc"},
		{"dot,alloc_space,flat,hide=line.*1?23?", "heap_alloc"},
		{"dot,inuse_space,flat,tagfocus=1mb:2gb", "heap"},
		{"dot,inuse_space,flat,tagfocus=30kb:,tagignore=1mb:2mb", "heap"},
		{"disasm=line[13],addresses,flat", "cpu"},
		{"peek=line.*01", "cpu"},
		{"weblist=line[13],addresses,flat", "cpu"},
		{"tags,tagfocus=400kb:", "heap_request"},
	}

	baseVars := pprofVariables
	defer func() { pprofVariables = baseVars }()
	for _, tc := range testcase {
		t.Run(tc.flags+":"+tc.source, func(t *testing.T) {
			// Reset the pprof variables before processing
			pprofVariables = baseVars.makeCopy()

			testUI := &proftest.TestUI{T: t, AllowRx: "Generating report in|Ignoring local file|expression matched no samples|Interpreted .* as range, not regexp"}

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
			defer os.Remove(protoTempFile.Name())
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
			o1.UI = testUI
			if err := PProf(o1); err != nil {
				t.Fatalf("%s %q:  %v", tc.source, tc.flags, err)
			}
			// Reset the pprof variables after the proto invocation
			pprofVariables = baseVars.makeCopy()

			// Read the profile from the encoded protobuf
			outputTempFile, err := ioutil.TempFile("", "profile_output")
			if err != nil {
				t.Errorf("cannot create tempfile: %v", err)
			}
			defer os.Remove(outputTempFile.Name())
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
			// The add_comment flag is not idempotent so only apply it on the first run.
			delete(f.strings, "add_comment")

			// Second pprof invocation to read the profile from profile.proto
			// and generate a report.
			o2 := setDefaults(nil)
			o2.Flagset = f
			o2.Sym = testSymbolizeDemangler{}
			o2.Obj = new(mockObjTool)
			o2.UI = testUI

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
				t.Fatalf("reading solution file %s: %v", solution, err)
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
				if *updateFlag {
					err := ioutil.WriteFile(solution, b, 0644)
					if err != nil {
						t.Errorf("failed to update the solution file %q: %v", solution, err)
					}
				}
			}
		})
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

func testSourceURL(port int) string {
	return fmt.Sprintf("http://%s/", net.JoinHostPort(testSourceAddress, strconv.Itoa(port)))
}

// solutionFilename returns the name of the solution file for the test
func solutionFilename(source string, f *testFlags) string {
	name := []string{"pprof", strings.TrimPrefix(source, testSourceURL(8000))}
	name = addString(name, f, []string{"flat", "cum"})
	name = addString(name, f, []string{"functions", "files", "lines", "addresses"})
	name = addString(name, f, []string{"inuse_space", "inuse_objects", "alloc_space", "alloc_objects"})
	name = addString(name, f, []string{"relative_percentages"})
	name = addString(name, f, []string{"seconds"})
	name = addString(name, f, []string{"call_tree"})
	name = addString(name, f, []string{"text", "tree", "callgrind", "dot", "svg", "tags", "dot", "traces", "disasm", "peek", "weblist", "topproto", "comments"})
	if f.strings["focus"] != "" || f.strings["tagfocus"] != "" {
		name = append(name, "focus")
	}
	if f.strings["ignore"] != "" || f.strings["tagignore"] != "" {
		name = append(name, "ignore")
	}
	if f.strings["show_from"] != "" {
		name = append(name, "show_from")
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
	bools       map[string]bool
	ints        map[string]int
	floats      map[string]float64
	strings     map[string]string
	args        []string
	stringLists map[string][]string
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
	if t, ok := f.stringLists[s]; ok {
		// convert slice of strings to slice of string pointers before returning.
		tp := make([]*string, len(t))
		for i, v := range t {
			tp[i] = &v
		}
		return &tp
	}
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

const testStart = 0x1000
const testOffset = 0x5000

type testFetcher struct{}

func (testFetcher) Fetch(s string, d, t time.Duration) (*profile.Profile, string, error) {
	var p *profile.Profile
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
	case "heap_request":
		p = heapProfile()
		for _, s := range p.Sample {
			s.NumLabel["request"] = s.NumLabel["bytes"]
		}
	case "heap_sizetags":
		p = heapProfile()
		tags := []int64{2, 4, 8, 16, 32, 64, 128, 256}
		for _, s := range p.Sample {
			numValues := append(s.NumLabel["bytes"], tags...)
			s.NumLabel["bytes"] = numValues
		}
	case "heap_tags":
		p = heapProfile()
		for i := 0; i < len(p.Sample); i += 2 {
			s := p.Sample[i]
			if s.Label == nil {
				s.Label = make(map[string][]string)
			}
			s.NumLabel["request"] = s.NumLabel["bytes"]
			s.Label["key1"] = []string{"tag"}
		}
	case "contention":
		p = contentionProfile()
	case "symbolz":
		p = symzProfile()
	default:
		return nil, "", fmt.Errorf("unexpected source: %s", s)
	}
	return p, testSourceURL(8000) + s, nil
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

	switch source {
	case testSourceURL(8000) + "symbolz":
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
	case testSourceURL(8001) + "symbolz":
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
	default:
		return nil, fmt.Errorf("unexpected source: %s", source)
	}
}

type testSymbolzSymbolizer struct{}

func (testSymbolzSymbolizer) Symbolize(variables string, sources plugin.MappingSources, p *profile.Profile) error {
	return symbolz.Symbolize(p, false, sources, testFetchSymbols, nil)
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
					"key1": {"tag1"},
					"key2": {"tag1"},
				},
			},
			{
				Location: []*profile.Location{cpuL[0], cpuL[3]},
				Value:    []int64{100, 100},
				Label: map[string][]string{
					"key1": {"tag2"},
					"key3": {"tag2"},
				},
			},
			{
				Location: []*profile.Location{cpuL[1], cpuL[4]},
				Value:    []int64{10, 10},
				Label: map[string][]string{
					"key1": {"tag3"},
					"key2": {"tag2"},
				},
			},
			{
				Location: []*profile.Location{cpuL[2]},
				Value:    []int64{10, 10},
				Label: map[string][]string{
					"key1": {"tag4"},
					"key2": {"tag1"},
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
				NumLabel: map[string][]int64{"bytes": {102400}},
			},
			{
				Location: []*profile.Location{heapL[0], heapL[3]},
				Value:    []int64{20, 4096000},
				NumLabel: map[string][]int64{"bytes": {204800}},
			},
			{
				Location: []*profile.Location{heapL[1], heapL[4]},
				Value:    []int64{40, 65536000},
				NumLabel: map[string][]int64{"bytes": {1638400}},
			},
			{
				Location: []*profile.Location{heapL[2]},
				Value:    []int64{80, 32768000},
				NumLabel: map[string][]int64{"bytes": {409600}},
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
		desc, value string
		tags        map[string][]string
		want        bool
	}{
		{
			"1 key with 1 matching value",
			"tag2",
			map[string][]string{"value1": {"tag1", "tag2"}},
			true,
		},
		{
			"1 key with no matching values",
			"tag3",
			map[string][]string{"value1": {"tag1", "tag2"}},
			false,
		},
		{
			"two keys, each with value matching different one value in list",
			"tag1,tag3",
			map[string][]string{"value1": {"tag1", "tag2"}, "value2": {"tag3"}},
			true,
		},
		{"two keys, all value matching different regex value in list",
			"t..[12],t..3",
			map[string][]string{"value1": {"tag1", "tag2"}, "value2": {"tag3"}},
			true,
		},
		{
			"one key, not all values in list matched",
			"tag2,tag3",
			map[string][]string{"value1": {"tag1", "tag2"}},
			false,
		},
		{
			"key specified, list of tags where all tags in list matched",
			"key1=tag1,tag2",
			map[string][]string{"key1": {"tag1", "tag2"}},
			true,
		},
		{"key specified, list of tag values where not all are matched",
			"key1=tag1,tag2",
			map[string][]string{"key1": {"tag1"}},
			true,
		},
		{
			"key included for regex matching, list of values where all values in list matched",
			"key1:tag1,tag2",
			map[string][]string{"key1": {"tag1", "tag2"}},
			true,
		},
		{
			"key included for regex matching, list of values where not only second value matched",
			"key1:tag1,tag2",
			map[string][]string{"key1": {"tag2"}},
			false,
		},
		{
			"key included for regex matching, list of values where not only first value matched",
			"key1:tag1,tag2",
			map[string][]string{"key1": {"tag1"}},
			false,
		},
	}
	for _, test := range tagFilterTests {
		t.Run(test.desc, func(*testing.T) {
			filter, err := compileTagFilter(test.desc, test.value, nil, &proftest.TestUI{T: t}, nil)
			if err != nil {
				t.Fatalf("tagFilter %s:%v", test.desc, err)
			}
			s := profile.Sample{
				Label: test.tags,
			}
			if got := filter(&s); got != test.want {
				t.Errorf("tagFilter %s: got %v, want %v", test.desc, got, test.want)
			}
		})
	}
}

func TestIdentifyNumLabelUnits(t *testing.T) {
	var tagFilterTests = []struct {
		desc               string
		tagVals            []map[string][]int64
		tagUnits           []map[string][]string
		wantUnits          map[string]string
		allowedRx          string
		wantIgnoreErrCount int
	}{
		{
			"Multiple keys, no units for all keys",
			[]map[string][]int64{{"keyA": {131072}, "keyB": {128}}},
			[]map[string][]string{{"keyA": {}, "keyB": {""}}},
			map[string]string{"keyA": "keyA", "keyB": "keyB"},
			"",
			0,
		},
		{
			"Multiple keys, different units for each key",
			[]map[string][]int64{{"keyA": {131072}, "keyB": {128}}},
			[]map[string][]string{{"keyA": {"bytes"}, "keyB": {"kilobytes"}}},
			map[string]string{"keyA": "bytes", "keyB": "kilobytes"},
			"",
			0,
		},
		{
			"Multiple keys with multiple values, different units for each key",
			[]map[string][]int64{{"keyC": {131072, 1}, "keyD": {128, 252}}},
			[]map[string][]string{{"keyC": {"bytes", "bytes"}, "keyD": {"kilobytes", "kilobytes"}}},
			map[string]string{"keyC": "bytes", "keyD": "kilobytes"},
			"",
			0,
		},
		{
			"Multiple keys with multiple values, some units missing",
			[]map[string][]int64{{"key1": {131072, 1}, "A": {128, 252}, "key3": {128}, "key4": {1}}, {"key3": {128}, "key4": {1}}},
			[]map[string][]string{{"key1": {"", "bytes"}, "A": {"kilobytes", ""}, "key3": {""}, "key4": {"hour"}}, {"key3": {"seconds"}, "key4": {""}}},
			map[string]string{"key1": "bytes", "A": "kilobytes", "key3": "seconds", "key4": "hour"},
			"",
			0,
		},
		{
			"One key with three units in same sample",
			[]map[string][]int64{{"key": {8, 8, 16}}},
			[]map[string][]string{{"key": {"bytes", "megabytes", "kilobytes"}}},
			map[string]string{"key": "bytes"},
			`(For tag key used unit bytes, also encountered unit\(s\) kilobytes, megabytes)`,
			1,
		},
		{
			"One key with four units in same sample",
			[]map[string][]int64{{"key": {8, 8, 16, 32}}},
			[]map[string][]string{{"key": {"bytes", "kilobytes", "a", "megabytes"}}},
			map[string]string{"key": "bytes"},
			`(For tag key used unit bytes, also encountered unit\(s\) a, kilobytes, megabytes)`,
			1,
		},
		{
			"One key with two units in same sample",
			[]map[string][]int64{{"key": {8, 8}}},
			[]map[string][]string{{"key": {"bytes", "seconds"}}},
			map[string]string{"key": "bytes"},
			`(For tag key used unit bytes, also encountered unit\(s\) seconds)`,
			1,
		},
		{
			"One key with different units in different samples",
			[]map[string][]int64{{"key1": {8}}, {"key1": {8}}, {"key1": {8}}},
			[]map[string][]string{{"key1": {"bytes"}}, {"key1": {"kilobytes"}}, {"key1": {"megabytes"}}},
			map[string]string{"key1": "bytes"},
			`(For tag key1 used unit bytes, also encountered unit\(s\) kilobytes, megabytes)`,
			1,
		},
		{
			"Key alignment, unit not specified",
			[]map[string][]int64{{"alignment": {8}}},
			[]map[string][]string{nil},
			map[string]string{"alignment": "bytes"},
			"",
			0,
		},
		{
			"Key request, unit not specified",
			[]map[string][]int64{{"request": {8}}, {"request": {8, 8}}},
			[]map[string][]string{nil, nil},
			map[string]string{"request": "bytes"},
			"",
			0,
		},
		{
			"Check units not over-written for keys with default units",
			[]map[string][]int64{{
				"alignment": {8},
				"request":   {8},
				"bytes":     {8},
			}},
			[]map[string][]string{{
				"alignment": {"seconds"},
				"request":   {"minutes"},
				"bytes":     {"hours"},
			}},
			map[string]string{
				"alignment": "seconds",
				"request":   "minutes",
				"bytes":     "hours",
			},
			"",
			0,
		},
	}
	for _, test := range tagFilterTests {
		t.Run(test.desc, func(*testing.T) {
			p := profile.Profile{Sample: make([]*profile.Sample, len(test.tagVals))}
			for i, numLabel := range test.tagVals {
				s := profile.Sample{
					NumLabel: numLabel,
					NumUnit:  test.tagUnits[i],
				}
				p.Sample[i] = &s
			}
			testUI := &proftest.TestUI{T: t, AllowRx: test.allowedRx}
			units := identifyNumLabelUnits(&p, testUI)
			if !reflect.DeepEqual(test.wantUnits, units) {
				t.Errorf("got %v units, want %v", units, test.wantUnits)
			}
			if got, want := testUI.NumAllowRxMatches, test.wantIgnoreErrCount; want != got {
				t.Errorf("got %d errors logged, want %d errors logged", got, want)
			}
		})
	}
}

func TestNumericTagFilter(t *testing.T) {
	var tagFilterTests = []struct {
		desc, value     string
		tags            map[string][]int64
		identifiedUnits map[string]string
		want            bool
	}{
		{
			"Match when unit conversion required",
			"128kb",
			map[string][]int64{"key1": {131072}, "key2": {128}},
			map[string]string{"key1": "bytes", "key2": "kilobytes"},
			true,
		},
		{
			"Match only when values equal after unit conversion",
			"512kb",
			map[string][]int64{"key1": {512}, "key2": {128}},
			map[string]string{"key1": "bytes", "key2": "kilobytes"},
			false,
		},
		{
			"Match when values and units initially equal",
			"10bytes",
			map[string][]int64{"key1": {10}, "key2": {128}},
			map[string]string{"key1": "bytes", "key2": "kilobytes"},
			true,
		},
		{
			"Match range without lower bound, no unit conversion required",
			":10bytes",
			map[string][]int64{"key1": {8}},
			map[string]string{"key1": "bytes"},
			true,
		},
		{
			"Match range without lower bound, unit conversion required",
			":10kb",
			map[string][]int64{"key1": {8}},
			map[string]string{"key1": "bytes"},
			true,
		},
		{
			"Match range without upper bound, unit conversion required",
			"10b:",
			map[string][]int64{"key1": {8}},
			map[string]string{"key1": "kilobytes"},
			true,
		},
		{
			"Match range without upper bound, no unit conversion required",
			"10b:",
			map[string][]int64{"key1": {12}},
			map[string]string{"key1": "bytes"},
			true,
		},
		{
			"Don't match range without upper bound, no unit conversion required",
			"10b:",
			map[string][]int64{"key1": {8}},
			map[string]string{"key1": "bytes"},
			false,
		},
		{
			"Multiple keys with different units, don't match range without upper bound",
			"10kb:",
			map[string][]int64{"key1": {8}},
			map[string]string{"key1": "bytes", "key2": "kilobytes"},
			false,
		},
		{
			"Match range without upper bound, unit conversion required",
			"10b:",
			map[string][]int64{"key1": {8}},
			map[string]string{"key1": "kilobytes"},
			true,
		},
		{
			"Don't match range without lower bound, no unit conversion required",
			":10b",
			map[string][]int64{"key1": {12}},
			map[string]string{"key1": "bytes"},
			false,
		},
		{
			"Match specific key, key present, one of two values match",
			"bytes=5b",
			map[string][]int64{"bytes": {10, 5}},
			map[string]string{"bytes": "bytes"},
			true,
		},
		{
			"Match specific key, key present and value matches",
			"bytes=1024b",
			map[string][]int64{"bytes": {1024}},
			map[string]string{"bytes": "kilobytes"},
			false,
		},
		{
			"Match specific key, matching key present and value matches, also non-matching key",
			"bytes=1024b",
			map[string][]int64{"bytes": {1024}, "key2": {5}},
			map[string]string{"bytes": "bytes", "key2": "bytes"},
			true,
		},
		{
			"Match specific key and range of values, value matches",
			"bytes=512b:1024b",
			map[string][]int64{"bytes": {780}},
			map[string]string{"bytes": "bytes"},
			true,
		},
		{
			"Match specific key and range of values, value too large",
			"key1=1kb:2kb",
			map[string][]int64{"key1": {4096}},
			map[string]string{"key1": "bytes"},
			false,
		},
		{
			"Match specific key and range of values, value too small",
			"key1=1kb:2kb",
			map[string][]int64{"key1": {256}},
			map[string]string{"key1": "bytes"},
			false,
		},
		{
			"Match specific key and value, unit conversion required",
			"bytes=1024b",
			map[string][]int64{"bytes": {1}},
			map[string]string{"bytes": "kilobytes"},
			true,
		},
		{
			"Match specific key and value, key does not appear",
			"key2=256bytes",
			map[string][]int64{"key1": {256}},
			map[string]string{"key1": "bytes"},
			false,
		},
	}
	for _, test := range tagFilterTests {
		t.Run(test.desc, func(*testing.T) {
			wantErrMsg := strings.Join([]string{"(", test.desc, ":Interpreted '", test.value[strings.Index(test.value, "=")+1:], "' as range, not regexp", ")"}, "")
			filter, err := compileTagFilter(test.desc, test.value, test.identifiedUnits, &proftest.TestUI{T: t,
				AllowRx: wantErrMsg}, nil)
			if err != nil {
				t.Fatalf("%v", err)
			}
			s := profile.Sample{
				NumLabel: test.tags,
			}
			if got := filter(&s); got != test.want {
				t.Fatalf("got %v, want %v", got, test.want)
			}
		})
	}
}

type testSymbolzMergeFetcher struct{}

func (testSymbolzMergeFetcher) Fetch(s string, d, t time.Duration) (*profile.Profile, string, error) {
	var p *profile.Profile
	switch s {
	case testSourceURL(8000) + "symbolz":
		p = symzProfile()
	case testSourceURL(8001) + "symbolz":
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

func TestSymbolzAfterMerge(t *testing.T) {
	baseVars := pprofVariables
	pprofVariables = baseVars.makeCopy()
	defer func() { pprofVariables = baseVars }()

	f := baseFlags()
	f.args = []string{
		testSourceURL(8000) + "symbolz",
		testSourceURL(8001) + "symbolz",
	}

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

	o.Fetch = testSymbolzMergeFetcher{}
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
			{Addr: 0x1000, Text: "instruction one", File: "file1000.src", Line: 1},
			{Addr: 0x1001, Text: "instruction two", File: "file1000.src", Line: 1},
			{Addr: 0x1002, Text: "instruction three", File: "file1000.src", Line: 2},
			{Addr: 0x1003, Text: "instruction four", File: "file1000.src", Line: 1},
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
	name, buildID string
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
	return m.buildID
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
			{
				Name: []string{"line1000"}, File: m.name,
				Start: 0x1000, End: 0x1003,
			},
			{
				Name: []string{"line3000"}, File: m.name,
				Start: 0x3000, End: 0x3004,
			},
		}, nil
	}
	return nil, fmt.Errorf("unimplemented")
}

// Close closes the file, releasing associated resources.
func (*mockFile) Close() error {
	return nil
}
