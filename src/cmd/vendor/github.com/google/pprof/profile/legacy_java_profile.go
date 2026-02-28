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

// This file implements parsers to convert java legacy profiles into
// the profile.proto format.

package profile

import (
	"bytes"
	"fmt"
	"io"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

var (
	attributeRx            = regexp.MustCompile(`([\w ]+)=([\w ]+)`)
	javaSampleRx           = regexp.MustCompile(` *(\d+) +(\d+) +@ +([ x0-9a-f]*)`)
	javaLocationRx         = regexp.MustCompile(`^\s*0x([[:xdigit:]]+)\s+(.*)\s*$`)
	javaLocationFileLineRx = regexp.MustCompile(`^(.*)\s+\((.+):(-?[[:digit:]]+)\)$`)
	javaLocationPathRx     = regexp.MustCompile(`^(.*)\s+\((.*)\)$`)
)

// javaCPUProfile returns a new Profile from profilez data.
// b is the profile bytes after the header, period is the profiling
// period, and parse is a function to parse 8-byte chunks from the
// profile in its native endianness.
func javaCPUProfile(b []byte, period int64, parse func(b []byte) (uint64, []byte)) (*Profile, error) {
	p := &Profile{
		Period:     period * 1000,
		PeriodType: &ValueType{Type: "cpu", Unit: "nanoseconds"},
		SampleType: []*ValueType{{Type: "samples", Unit: "count"}, {Type: "cpu", Unit: "nanoseconds"}},
	}
	var err error
	var locs map[uint64]*Location
	if b, locs, err = parseCPUSamples(b, parse, false, p); err != nil {
		return nil, err
	}

	if err = parseJavaLocations(b, locs, p); err != nil {
		return nil, err
	}

	// Strip out addresses for better merge.
	if err = p.Aggregate(true, true, true, true, false, false); err != nil {
		return nil, err
	}

	return p, nil
}

// parseJavaProfile returns a new profile from heapz or contentionz
// data. b is the profile bytes after the header.
func parseJavaProfile(b []byte) (*Profile, error) {
	h := bytes.SplitAfterN(b, []byte("\n"), 2)
	if len(h) < 2 {
		return nil, errUnrecognized
	}

	p := &Profile{
		PeriodType: &ValueType{},
	}
	header := string(bytes.TrimSpace(h[0]))

	var err error
	var pType string
	switch header {
	case "--- heapz 1 ---":
		pType = "heap"
	case "--- contentionz 1 ---":
		pType = "contention"
	default:
		return nil, errUnrecognized
	}

	if b, err = parseJavaHeader(pType, h[1], p); err != nil {
		return nil, err
	}
	var locs map[uint64]*Location
	if b, locs, err = parseJavaSamples(pType, b, p); err != nil {
		return nil, err
	}
	if err = parseJavaLocations(b, locs, p); err != nil {
		return nil, err
	}

	// Strip out addresses for better merge.
	if err = p.Aggregate(true, true, true, true, false, false); err != nil {
		return nil, err
	}

	return p, nil
}

// parseJavaHeader parses the attribute section on a java profile and
// populates a profile. Returns the remainder of the buffer after all
// attributes.
func parseJavaHeader(pType string, b []byte, p *Profile) ([]byte, error) {
	nextNewLine := bytes.IndexByte(b, byte('\n'))
	for nextNewLine != -1 {
		line := string(bytes.TrimSpace(b[0:nextNewLine]))
		if line != "" {
			h := attributeRx.FindStringSubmatch(line)
			if h == nil {
				// Not a valid attribute, exit.
				return b, nil
			}

			attribute, value := strings.TrimSpace(h[1]), strings.TrimSpace(h[2])
			var err error
			switch pType + "/" + attribute {
			case "heap/format", "cpu/format", "contention/format":
				if value != "java" {
					return nil, errUnrecognized
				}
			case "heap/resolution":
				p.SampleType = []*ValueType{
					{Type: "inuse_objects", Unit: "count"},
					{Type: "inuse_space", Unit: value},
				}
			case "contention/resolution":
				p.SampleType = []*ValueType{
					{Type: "contentions", Unit: "count"},
					{Type: "delay", Unit: value},
				}
			case "contention/sampling period":
				p.PeriodType = &ValueType{
					Type: "contentions", Unit: "count",
				}
				if p.Period, err = strconv.ParseInt(value, 0, 64); err != nil {
					return nil, fmt.Errorf("failed to parse attribute %s: %v", line, err)
				}
			case "contention/ms since reset":
				millis, err := strconv.ParseInt(value, 0, 64)
				if err != nil {
					return nil, fmt.Errorf("failed to parse attribute %s: %v", line, err)
				}
				p.DurationNanos = millis * 1000 * 1000
			default:
				return nil, errUnrecognized
			}
		}
		// Grab next line.
		b = b[nextNewLine+1:]
		nextNewLine = bytes.IndexByte(b, byte('\n'))
	}
	return b, nil
}

// parseJavaSamples parses the samples from a java profile and
// populates the Samples in a profile. Returns the remainder of the
// buffer after the samples.
func parseJavaSamples(pType string, b []byte, p *Profile) ([]byte, map[uint64]*Location, error) {
	nextNewLine := bytes.IndexByte(b, byte('\n'))
	locs := make(map[uint64]*Location)
	for nextNewLine != -1 {
		line := string(bytes.TrimSpace(b[0:nextNewLine]))
		if line != "" {
			sample := javaSampleRx.FindStringSubmatch(line)
			if sample == nil {
				// Not a valid sample, exit.
				return b, locs, nil
			}

			// Java profiles have data/fields inverted compared to other
			// profile types.
			var err error
			value1, value2, value3 := sample[2], sample[1], sample[3]
			addrs, err := parseHexAddresses(value3)
			if err != nil {
				return nil, nil, fmt.Errorf("malformed sample: %s: %v", line, err)
			}

			var sloc []*Location
			for _, addr := range addrs {
				loc := locs[addr]
				if locs[addr] == nil {
					loc = &Location{
						Address: addr,
					}
					p.Location = append(p.Location, loc)
					locs[addr] = loc
				}
				sloc = append(sloc, loc)
			}
			s := &Sample{
				Value:    make([]int64, 2),
				Location: sloc,
			}

			if s.Value[0], err = strconv.ParseInt(value1, 0, 64); err != nil {
				return nil, nil, fmt.Errorf("parsing sample %s: %v", line, err)
			}
			if s.Value[1], err = strconv.ParseInt(value2, 0, 64); err != nil {
				return nil, nil, fmt.Errorf("parsing sample %s: %v", line, err)
			}

			switch pType {
			case "heap":
				const javaHeapzSamplingRate = 524288 // 512K
				if s.Value[0] == 0 {
					return nil, nil, fmt.Errorf("parsing sample %s: second value must be non-zero", line)
				}
				s.NumLabel = map[string][]int64{"bytes": {s.Value[1] / s.Value[0]}}
				s.Value[0], s.Value[1] = scaleHeapSample(s.Value[0], s.Value[1], javaHeapzSamplingRate)
			case "contention":
				if period := p.Period; period != 0 {
					s.Value[0] = s.Value[0] * p.Period
					s.Value[1] = s.Value[1] * p.Period
				}
			}
			p.Sample = append(p.Sample, s)
		}
		// Grab next line.
		b = b[nextNewLine+1:]
		nextNewLine = bytes.IndexByte(b, byte('\n'))
	}
	return b, locs, nil
}

// parseJavaLocations parses the location information in a java
// profile and populates the Locations in a profile. It uses the
// location addresses from the profile as both the ID of each
// location.
func parseJavaLocations(b []byte, locs map[uint64]*Location, p *Profile) error {
	r := bytes.NewBuffer(b)
	fns := make(map[string]*Function)
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return err
			}
			if line == "" {
				break
			}
		}

		if line = strings.TrimSpace(line); line == "" {
			continue
		}

		jloc := javaLocationRx.FindStringSubmatch(line)
		if len(jloc) != 3 {
			continue
		}
		addr, err := strconv.ParseUint(jloc[1], 16, 64)
		if err != nil {
			return fmt.Errorf("parsing sample %s: %v", line, err)
		}
		loc := locs[addr]
		if loc == nil {
			// Unused/unseen
			continue
		}
		var lineFunc, lineFile string
		var lineNo int64

		if fileLine := javaLocationFileLineRx.FindStringSubmatch(jloc[2]); len(fileLine) == 4 {
			// Found a line of the form: "function (file:line)"
			lineFunc, lineFile = fileLine[1], fileLine[2]
			if n, err := strconv.ParseInt(fileLine[3], 10, 64); err == nil && n > 0 {
				lineNo = n
			}
		} else if filePath := javaLocationPathRx.FindStringSubmatch(jloc[2]); len(filePath) == 3 {
			// If there's not a file:line, it's a shared library path.
			// The path isn't interesting, so just give the .so.
			lineFunc, lineFile = filePath[1], filepath.Base(filePath[2])
		} else if strings.Contains(jloc[2], "generated stub/JIT") {
			lineFunc = "STUB"
		} else {
			// Treat whole line as the function name. This is used by the
			// java agent for internal states such as "GC" or "VM".
			lineFunc = jloc[2]
		}
		fn := fns[lineFunc]

		if fn == nil {
			fn = &Function{
				Name:       lineFunc,
				SystemName: lineFunc,
				Filename:   lineFile,
			}
			fns[lineFunc] = fn
			p.Function = append(p.Function, fn)
		}
		loc.Line = []Line{
			{
				Function: fn,
				Line:     lineNo,
			},
		}
		loc.Address = 0
	}

	p.remapLocationIDs()
	p.remapFunctionIDs()
	p.remapMappingIDs()

	return nil
}
