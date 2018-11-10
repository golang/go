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

package profile

import (
	"bytes"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestLegacyProfileType(t *testing.T) {
	type testcase struct {
		sampleTypes []string
		typeSet     [][]string
		want        bool
		setName     string
	}

	heap := heapzSampleTypes
	cont := contentionzSampleTypes
	testcases := []testcase{
		// True cases
		{[]string{"allocations", "size"}, heap, true, "heapzSampleTypes"},
		{[]string{"objects", "space"}, heap, true, "heapzSampleTypes"},
		{[]string{"inuse_objects", "inuse_space"}, heap, true, "heapzSampleTypes"},
		{[]string{"alloc_objects", "alloc_space"}, heap, true, "heapzSampleTypes"},
		{[]string{"contentions", "delay"}, cont, true, "contentionzSampleTypes"},
		// False cases
		{[]string{"objects"}, heap, false, "heapzSampleTypes"},
		{[]string{"objects", "unknown"}, heap, false, "heapzSampleTypes"},
		{[]string{"contentions", "delay"}, heap, false, "heapzSampleTypes"},
		{[]string{"samples", "cpu"}, heap, false, "heapzSampleTypes"},
		{[]string{"samples", "cpu"}, cont, false, "contentionzSampleTypes"},
	}

	for _, tc := range testcases {
		p := profileOfType(tc.sampleTypes)
		if got := isProfileType(p, tc.typeSet); got != tc.want {
			t.Error("isProfileType({"+strings.Join(tc.sampleTypes, ",")+"},", tc.setName, "), got", got, "want", tc.want)
		}
	}
}

func TestCpuParse(t *testing.T) {
	// profileString is a legacy encoded profile, represnted by words separated by ":"
	// Each sample has the form value : N : stack1..stackN
	// EOF is represented as "0:1:0"
	profileString := "1:3:100:999:100:"                                      // sample with bogus 999 and duplicate leaf
	profileString += "1:5:200:999:200:501:502:"                              // sample with bogus 999 and duplicate leaf
	profileString += "1:12:300:999:300:601:602:603:604:605:606:607:608:609:" // sample with bogus 999 and duplicate leaf
	profileString += "0:1:0000"                                              // EOF -- must use 4 bytes for the final zero

	p, err := cpuProfile([]byte(profileString), 1, parseString)
	if err != nil {
		t.Fatal(err)
	}

	if err := checkTestSample(p, []uint64{100}); err != nil {
		t.Error(err)
	}
	if err := checkTestSample(p, []uint64{200, 500, 501}); err != nil {
		t.Error(err)
	}
	if err := checkTestSample(p, []uint64{300, 600, 601, 602, 603, 604, 605, 606, 607, 608}); err != nil {
		t.Error(err)
	}
}

func parseString(b []byte) (uint64, []byte) {
	slices := bytes.SplitN(b, []byte(":"), 2)
	var value, remainder []byte
	if len(slices) > 0 {
		value = slices[0]
	}
	if len(slices) > 1 {
		remainder = slices[1]
	}
	v, _ := strconv.ParseUint(string(value), 10, 64)
	return v, remainder
}

func checkTestSample(p *Profile, want []uint64) error {
	for _, s := range p.Sample {
		got := []uint64{}
		for _, l := range s.Location {
			got = append(got, l.Address)
		}
		if reflect.DeepEqual(got, want) {
			return nil
		}
	}
	return fmt.Errorf("Could not find sample : %v", want)
}

// profileOfType creates an empty profile with only sample types set,
// for testing purposes only.
func profileOfType(sampleTypes []string) *Profile {
	p := new(Profile)
	p.SampleType = make([]*ValueType, len(sampleTypes))
	for i, t := range sampleTypes {
		p.SampleType[i] = new(ValueType)
		p.SampleType[i].Type = t
	}
	return p
}

func TestParseMappingEntry(t *testing.T) {
	for _, test := range []*struct {
		entry string
		want  *Mapping
	}{
		{
			entry: "00400000-02e00000 r-xp 00000000 00:00 0",
			want: &Mapping{
				Start: 0x400000,
				Limit: 0x2e00000,
			},
		},
		{
			entry: "02e00000-02e8a000 r-xp 02a00000 00:00 15953927    /foo/bin",
			want: &Mapping{
				Start:  0x2e00000,
				Limit:  0x2e8a000,
				Offset: 0x2a00000,
				File:   "/foo/bin",
			},
		},
		{
			entry: "02e00000-02e8a000 r-xp 000000 00:00 15953927    [vdso]",
			want: &Mapping{
				Start: 0x2e00000,
				Limit: 0x2e8a000,
				File:  "[vdso]",
			},
		},
		{
			entry: "  02e00000-02e8a000: /foo/bin (@2a00000)",
			want: &Mapping{
				Start:  0x2e00000,
				Limit:  0x2e8a000,
				Offset: 0x2a00000,
				File:   "/foo/bin",
			},
		},
		{
			entry: "  02e00000-02e8a000: /foo/bin (deleted)",
			want: &Mapping{
				Start: 0x2e00000,
				Limit: 0x2e8a000,
				File:  "/foo/bin",
			},
		},
		{
			entry: "  02e00000-02e8a000: /foo/bin",
			want: &Mapping{
				Start: 0x2e00000,
				Limit: 0x2e8a000,
				File:  "/foo/bin",
			},
		},
		{
			entry: "  02e00000-02e8a000: [vdso]",
			want: &Mapping{
				Start: 0x2e00000,
				Limit: 0x2e8a000,
				File:  "[vdso]",
			},
		},
		{entry: "0xff6810563000 0xff6810565000 r-xp abc_exe 87c4d547f895cfd6a370e08dc5c5ee7bd4199d5b",
			want: &Mapping{
				Start:   0xff6810563000,
				Limit:   0xff6810565000,
				File:    "abc_exe",
				BuildID: "87c4d547f895cfd6a370e08dc5c5ee7bd4199d5b",
			},
		},
		{entry: "7f5e5435e000-7f5e5455e000 --xp 00002000 00:00 1531        myprogram",
			want: &Mapping{
				Start:  0x7f5e5435e000,
				Limit:  0x7f5e5455e000,
				Offset: 0x2000,
				File:   "myprogram",
			},
		},
		{entry: "7f7472710000-7f7472722000 r-xp 00000000 fc:00 790190      /usr/lib/libfantastic-1.2.so",
			want: &Mapping{
				Start: 0x7f7472710000,
				Limit: 0x7f7472722000,
				File:  "/usr/lib/libfantastic-1.2.so",
			},
		},
		{entry: "7f47a542f000-7f47a5447000: /lib/libpthread-2.15.so",
			want: &Mapping{
				Start: 0x7f47a542f000,
				Limit: 0x7f47a5447000,
				File:  "/lib/libpthread-2.15.so",
			},
		},
		{entry: "0x40000-0x80000 /path/to/binary      (@FF00)            abc123456",
			want: &Mapping{
				Start:   0x40000,
				Limit:   0x80000,
				File:    "/path/to/binary",
				Offset:  0xFF00,
				BuildID: "abc123456",
			},
		},
		{entry: "W1220 15:07:15.201776    8272 logger.cc:12033] --- Memory map: ---\n" +
			"0x40000-0x80000 /path/to/binary      (@FF00)            abc123456",
			want: &Mapping{
				Start:   0x40000,
				Limit:   0x80000,
				File:    "/path/to/binary",
				Offset:  0xFF00,
				BuildID: "abc123456",
			},
		},
		{entry: "W1220 15:07:15.201776    8272 logger.cc:12033] --- Memory map: ---\n" +
			"W1220 15:07:15.202776    8272 logger.cc:12036]   0x40000-0x80000 /path/to/binary      (@FF00)            abc123456",
			want: &Mapping{
				Start:   0x40000,
				Limit:   0x80000,
				File:    "/path/to/binary",
				Offset:  0xFF00,
				BuildID: "abc123456",
			},
		},
		{entry: "7f5e5435e000-7f5e5455e000 ---p 00002000 00:00 1531        myprogram",
			want: nil,
		},
	} {
		got, err := ParseProcMaps(strings.NewReader(test.entry))
		if err != nil {
			t.Errorf("%s: %v", test.entry, err)
			continue
		}
		if test.want == nil {
			if got, want := len(got), 0; got != want {
				t.Errorf("%s: got %d mappings, want %d", test.entry, got, want)
			}
			continue
		}
		if got, want := len(got), 1; got != want {
			t.Errorf("%s: got %d mappings, want %d", test.entry, got, want)
			continue
		}
		if !reflect.DeepEqual(test.want, got[0]) {
			t.Errorf("%s want=%v got=%v", test.entry, test.want, got[0])
		}
	}
}

func TestParseThreadProfileWithInvalidAddress(t *testing.T) {
	profile := `
--- threadz 1 ---

--- Thread 7eff063d9940 (name: main/25376) stack: ---
  PC: 0x40b688 0x4d5f51 0x40be31 0x473add693e639c6f0
--- Memory map: ---
  00400000-00fcb000: /home/rsilvera/cppbench/cppbench_server_main.unstripped
	`
	wantErr := "failed to parse as hex 64-bit number: 0x473add693e639c6f0"
	if _, gotErr := parseThread([]byte(profile)); !strings.Contains(gotErr.Error(), wantErr) {
		t.Errorf("parseThread(): got error %q, want error containing %q", gotErr, wantErr)
	}
}

func TestParseGoCount(t *testing.T) {
	for _, test := range []struct {
		in  string
		typ string
	}{
		{
			in: `# ignored comment

threadcreate profile: total 123
`,
			typ: "threadcreate",
		},
		{
			in: `
# ignored comment
goroutine profile: total 123456
`,
			typ: "goroutine",
		},
		{
			in: `
sub/dir-ect_o.ry profile: total 999
`,
			typ: "sub/dir-ect_o.ry",
		},
	} {
		t.Run(test.typ, func(t *testing.T) {
			p, err := parseGoCount([]byte(test.in))
			if err != nil {
				t.Fatalf("parseGoCount(%q) = %v", test.in, err)
			}
			if typ := p.PeriodType.Type; typ != test.typ {
				t.Fatalf("parseGoCount(%q).PeriodType.Type = %q want %q", test.in, typ, test.typ)
			}
		})
	}
}
