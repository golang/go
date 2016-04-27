// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements parsers to convert legacy profiles into the
// profile.proto format.

package profile

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
)

var (
	countStartRE = regexp.MustCompile(`\A(\w+) profile: total \d+\n\z`)
	countRE      = regexp.MustCompile(`\A(\d+) @(( 0x[0-9a-f]+)+)\n\z`)

	heapHeaderRE = regexp.MustCompile(`heap profile: *(\d+): *(\d+) *\[ *(\d+): *(\d+) *\] *@ *(heap[_a-z0-9]*)/?(\d*)`)
	heapSampleRE = regexp.MustCompile(`(-?\d+): *(-?\d+) *\[ *(\d+): *(\d+) *] @([ x0-9a-f]*)`)

	contentionSampleRE = regexp.MustCompile(`(\d+) *(\d+) @([ x0-9a-f]*)`)

	hexNumberRE = regexp.MustCompile(`0x[0-9a-f]+`)

	growthHeaderRE = regexp.MustCompile(`heap profile: *(\d+): *(\d+) *\[ *(\d+): *(\d+) *\] @ growthz`)

	fragmentationHeaderRE = regexp.MustCompile(`heap profile: *(\d+): *(\d+) *\[ *(\d+): *(\d+) *\] @ fragmentationz`)

	threadzStartRE = regexp.MustCompile(`--- threadz \d+ ---`)
	threadStartRE  = regexp.MustCompile(`--- Thread ([[:xdigit:]]+) \(name: (.*)/(\d+)\) stack: ---`)

	procMapsRE = regexp.MustCompile(`([[:xdigit:]]+)-([[:xdigit:]]+)\s+([-rwxp]+)\s+([[:xdigit:]]+)\s+([[:xdigit:]]+):([[:xdigit:]]+)\s+([[:digit:]]+)\s*(\S+)?`)

	briefMapsRE = regexp.MustCompile(`\s*([[:xdigit:]]+)-([[:xdigit:]]+):\s*(\S+)(\s.*@)?([[:xdigit:]]+)?`)

	// LegacyHeapAllocated instructs the heapz parsers to use the
	// allocated memory stats instead of the default in-use memory. Note
	// that tcmalloc doesn't provide all allocated memory, only in-use
	// stats.
	LegacyHeapAllocated bool
)

func isSpaceOrComment(line string) bool {
	trimmed := strings.TrimSpace(line)
	return len(trimmed) == 0 || trimmed[0] == '#'
}

// parseGoCount parses a Go count profile (e.g., threadcreate or
// goroutine) and returns a new Profile.
func parseGoCount(b []byte) (*Profile, error) {
	r := bytes.NewBuffer(b)

	var line string
	var err error
	for {
		// Skip past comments and empty lines seeking a real header.
		line, err = r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		if !isSpaceOrComment(line) {
			break
		}
	}

	m := countStartRE.FindStringSubmatch(line)
	if m == nil {
		return nil, errUnrecognized
	}
	profileType := m[1]
	p := &Profile{
		PeriodType: &ValueType{Type: profileType, Unit: "count"},
		Period:     1,
		SampleType: []*ValueType{{Type: profileType, Unit: "count"}},
	}
	locations := make(map[uint64]*Location)
	for {
		line, err = r.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		if isSpaceOrComment(line) {
			continue
		}
		if strings.HasPrefix(line, "---") {
			break
		}
		m := countRE.FindStringSubmatch(line)
		if m == nil {
			return nil, errMalformed
		}
		n, err := strconv.ParseInt(m[1], 0, 64)
		if err != nil {
			return nil, errMalformed
		}
		fields := strings.Fields(m[2])
		locs := make([]*Location, 0, len(fields))
		for _, stk := range fields {
			addr, err := strconv.ParseUint(stk, 0, 64)
			if err != nil {
				return nil, errMalformed
			}
			// Adjust all frames by -1 to land on the call instruction.
			addr--
			loc := locations[addr]
			if loc == nil {
				loc = &Location{
					Address: addr,
				}
				locations[addr] = loc
				p.Location = append(p.Location, loc)
			}
			locs = append(locs, loc)
		}
		p.Sample = append(p.Sample, &Sample{
			Location: locs,
			Value:    []int64{n},
		})
	}

	if err = parseAdditionalSections(strings.TrimSpace(line), r, p); err != nil {
		return nil, err
	}
	return p, nil
}

// remapLocationIDs ensures there is a location for each address
// referenced by a sample, and remaps the samples to point to the new
// location ids.
func (p *Profile) remapLocationIDs() {
	seen := make(map[*Location]bool, len(p.Location))
	var locs []*Location

	for _, s := range p.Sample {
		for _, l := range s.Location {
			if seen[l] {
				continue
			}
			l.ID = uint64(len(locs) + 1)
			locs = append(locs, l)
			seen[l] = true
		}
	}
	p.Location = locs
}

func (p *Profile) remapFunctionIDs() {
	seen := make(map[*Function]bool, len(p.Function))
	var fns []*Function

	for _, l := range p.Location {
		for _, ln := range l.Line {
			fn := ln.Function
			if fn == nil || seen[fn] {
				continue
			}
			fn.ID = uint64(len(fns) + 1)
			fns = append(fns, fn)
			seen[fn] = true
		}
	}
	p.Function = fns
}

// remapMappingIDs matches location addresses with existing mappings
// and updates them appropriately. This is O(N*M), if this ever shows
// up as a bottleneck, evaluate sorting the mappings and doing a
// binary search, which would make it O(N*log(M)).
func (p *Profile) remapMappingIDs() {
	if len(p.Mapping) == 0 {
		return
	}

	// Some profile handlers will incorrectly set regions for the main
	// executable if its section is remapped. Fix them through heuristics.

	// Remove the initial mapping if named '/anon_hugepage' and has a
	// consecutive adjacent mapping.
	if m := p.Mapping[0]; strings.HasPrefix(m.File, "/anon_hugepage") {
		if len(p.Mapping) > 1 && m.Limit == p.Mapping[1].Start {
			p.Mapping = p.Mapping[1:]
		}
	}

	// Subtract the offset from the start of the main mapping if it
	// ends up at a recognizable start address.
	const expectedStart = 0x400000
	if m := p.Mapping[0]; m.Start-m.Offset == expectedStart {
		m.Start = expectedStart
		m.Offset = 0
	}

	for _, l := range p.Location {
		if a := l.Address; a != 0 {
			for _, m := range p.Mapping {
				if m.Start <= a && a < m.Limit {
					l.Mapping = m
					break
				}
			}
		}
	}

	// Reset all mapping IDs.
	for i, m := range p.Mapping {
		m.ID = uint64(i + 1)
	}
}

var cpuInts = []func([]byte) (uint64, []byte){
	get32l,
	get32b,
	get64l,
	get64b,
}

func get32l(b []byte) (uint64, []byte) {
	if len(b) < 4 {
		return 0, nil
	}
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24, b[4:]
}

func get32b(b []byte) (uint64, []byte) {
	if len(b) < 4 {
		return 0, nil
	}
	return uint64(b[3]) | uint64(b[2])<<8 | uint64(b[1])<<16 | uint64(b[0])<<24, b[4:]
}

func get64l(b []byte) (uint64, []byte) {
	if len(b) < 8 {
		return 0, nil
	}
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56, b[8:]
}

func get64b(b []byte) (uint64, []byte) {
	if len(b) < 8 {
		return 0, nil
	}
	return uint64(b[7]) | uint64(b[6])<<8 | uint64(b[5])<<16 | uint64(b[4])<<24 | uint64(b[3])<<32 | uint64(b[2])<<40 | uint64(b[1])<<48 | uint64(b[0])<<56, b[8:]
}

// ParseTracebacks parses a set of tracebacks and returns a newly
// populated profile. It will accept any text file and generate a
// Profile out of it with any hex addresses it can identify, including
// a process map if it can recognize one. Each sample will include a
// tag "source" with the addresses recognized in string format.
func ParseTracebacks(b []byte) (*Profile, error) {
	r := bytes.NewBuffer(b)

	p := &Profile{
		PeriodType: &ValueType{Type: "trace", Unit: "count"},
		Period:     1,
		SampleType: []*ValueType{
			{Type: "trace", Unit: "count"},
		},
	}

	var sources []string
	var sloc []*Location

	locs := make(map[uint64]*Location)
	for {
		l, err := r.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return nil, err
			}
			if l == "" {
				break
			}
		}
		if sectionTrigger(l) == memoryMapSection {
			break
		}
		if s, addrs := extractHexAddresses(l); len(s) > 0 {
			for _, addr := range addrs {
				// Addresses from stack traces point to the next instruction after
				// each call. Adjust by -1 to land somewhere on the actual call.
				addr--
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

			sources = append(sources, s...)
		} else {
			if len(sources) > 0 || len(sloc) > 0 {
				addTracebackSample(sloc, sources, p)
				sloc, sources = nil, nil
			}
		}
	}

	// Add final sample to save any leftover data.
	if len(sources) > 0 || len(sloc) > 0 {
		addTracebackSample(sloc, sources, p)
	}

	if err := p.ParseMemoryMap(r); err != nil {
		return nil, err
	}
	return p, nil
}

func addTracebackSample(l []*Location, s []string, p *Profile) {
	p.Sample = append(p.Sample,
		&Sample{
			Value:    []int64{1},
			Location: l,
			Label:    map[string][]string{"source": s},
		})
}

// parseCPU parses a profilez legacy profile and returns a newly
// populated Profile.
//
// The general format for profilez samples is a sequence of words in
// binary format. The first words are a header with the following data:
//   1st word -- 0
//   2nd word -- 3
//   3rd word -- 0 if a c++ application, 1 if a java application.
//   4th word -- Sampling period (in microseconds).
//   5th word -- Padding.
func parseCPU(b []byte) (*Profile, error) {
	var parse func([]byte) (uint64, []byte)
	var n1, n2, n3, n4, n5 uint64
	for _, parse = range cpuInts {
		var tmp []byte
		n1, tmp = parse(b)
		n2, tmp = parse(tmp)
		n3, tmp = parse(tmp)
		n4, tmp = parse(tmp)
		n5, tmp = parse(tmp)

		if tmp != nil && n1 == 0 && n2 == 3 && n3 == 0 && n4 > 0 && n5 == 0 {
			b = tmp
			return cpuProfile(b, int64(n4), parse)
		}
	}
	return nil, errUnrecognized
}

// cpuProfile returns a new Profile from C++ profilez data.
// b is the profile bytes after the header, period is the profiling
// period, and parse is a function to parse 8-byte chunks from the
// profile in its native endianness.
func cpuProfile(b []byte, period int64, parse func(b []byte) (uint64, []byte)) (*Profile, error) {
	p := &Profile{
		Period:     period * 1000,
		PeriodType: &ValueType{Type: "cpu", Unit: "nanoseconds"},
		SampleType: []*ValueType{
			{Type: "samples", Unit: "count"},
			{Type: "cpu", Unit: "nanoseconds"},
		},
	}
	var err error
	if b, _, err = parseCPUSamples(b, parse, true, p); err != nil {
		return nil, err
	}

	// If all samples have the same second-to-the-bottom frame, it
	// strongly suggests that it is an uninteresting artifact of
	// measurement -- a stack frame pushed by the signal handler. The
	// bottom frame is always correct as it is picked up from the signal
	// structure, not the stack. Check if this is the case and if so,
	// remove.
	if len(p.Sample) > 1 && len(p.Sample[0].Location) > 1 {
		allSame := true
		id1 := p.Sample[0].Location[1].Address
		for _, s := range p.Sample {
			if len(s.Location) < 2 || id1 != s.Location[1].Address {
				allSame = false
				break
			}
		}
		if allSame {
			for _, s := range p.Sample {
				s.Location = append(s.Location[:1], s.Location[2:]...)
			}
		}
	}

	if err := p.ParseMemoryMap(bytes.NewBuffer(b)); err != nil {
		return nil, err
	}
	return p, nil
}

// parseCPUSamples parses a collection of profilez samples from a
// profile.
//
// profilez samples are a repeated sequence of stack frames of the
// form:
//    1st word -- The number of times this stack was encountered.
//    2nd word -- The size of the stack (StackSize).
//    3rd word -- The first address on the stack.
//    ...
//    StackSize + 2 -- The last address on the stack
// The last stack trace is of the form:
//   1st word -- 0
//   2nd word -- 1
//   3rd word -- 0
//
// Addresses from stack traces may point to the next instruction after
// each call. Optionally adjust by -1 to land somewhere on the actual
// call (except for the leaf, which is not a call).
func parseCPUSamples(b []byte, parse func(b []byte) (uint64, []byte), adjust bool, p *Profile) ([]byte, map[uint64]*Location, error) {
	locs := make(map[uint64]*Location)
	for len(b) > 0 {
		var count, nstk uint64
		count, b = parse(b)
		nstk, b = parse(b)
		if b == nil || nstk > uint64(len(b)/4) {
			return nil, nil, errUnrecognized
		}
		var sloc []*Location
		addrs := make([]uint64, nstk)
		for i := 0; i < int(nstk); i++ {
			addrs[i], b = parse(b)
		}

		if count == 0 && nstk == 1 && addrs[0] == 0 {
			// End of data marker
			break
		}
		for i, addr := range addrs {
			if adjust && i > 0 {
				addr--
			}
			loc := locs[addr]
			if loc == nil {
				loc = &Location{
					Address: addr,
				}
				locs[addr] = loc
				p.Location = append(p.Location, loc)
			}
			sloc = append(sloc, loc)
		}
		p.Sample = append(p.Sample,
			&Sample{
				Value:    []int64{int64(count), int64(count) * p.Period},
				Location: sloc,
			})
	}
	// Reached the end without finding the EOD marker.
	return b, locs, nil
}

// parseHeap parses a heapz legacy or a growthz profile and
// returns a newly populated Profile.
func parseHeap(b []byte) (p *Profile, err error) {
	r := bytes.NewBuffer(b)
	l, err := r.ReadString('\n')
	if err != nil {
		return nil, errUnrecognized
	}

	sampling := ""

	if header := heapHeaderRE.FindStringSubmatch(l); header != nil {
		p = &Profile{
			SampleType: []*ValueType{
				{Type: "objects", Unit: "count"},
				{Type: "space", Unit: "bytes"},
			},
			PeriodType: &ValueType{Type: "objects", Unit: "bytes"},
		}

		var period int64
		if len(header[6]) > 0 {
			if period, err = strconv.ParseInt(header[6], 10, 64); err != nil {
				return nil, errUnrecognized
			}
		}

		switch header[5] {
		case "heapz_v2", "heap_v2":
			sampling, p.Period = "v2", period
		case "heapprofile":
			sampling, p.Period = "", 1
		case "heap":
			sampling, p.Period = "v2", period/2
		default:
			return nil, errUnrecognized
		}
	} else if header = growthHeaderRE.FindStringSubmatch(l); header != nil {
		p = &Profile{
			SampleType: []*ValueType{
				{Type: "objects", Unit: "count"},
				{Type: "space", Unit: "bytes"},
			},
			PeriodType: &ValueType{Type: "heapgrowth", Unit: "count"},
			Period:     1,
		}
	} else if header = fragmentationHeaderRE.FindStringSubmatch(l); header != nil {
		p = &Profile{
			SampleType: []*ValueType{
				{Type: "objects", Unit: "count"},
				{Type: "space", Unit: "bytes"},
			},
			PeriodType: &ValueType{Type: "allocations", Unit: "count"},
			Period:     1,
		}
	} else {
		return nil, errUnrecognized
	}

	if LegacyHeapAllocated {
		for _, st := range p.SampleType {
			st.Type = "alloc_" + st.Type
		}
	} else {
		for _, st := range p.SampleType {
			st.Type = "inuse_" + st.Type
		}
	}

	locs := make(map[uint64]*Location)
	for {
		l, err = r.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return nil, err
			}

			if l == "" {
				break
			}
		}

		if isSpaceOrComment(l) {
			continue
		}
		l = strings.TrimSpace(l)

		if sectionTrigger(l) != unrecognizedSection {
			break
		}

		value, blocksize, addrs, err := parseHeapSample(l, p.Period, sampling)
		if err != nil {
			return nil, err
		}
		var sloc []*Location
		for _, addr := range addrs {
			// Addresses from stack traces point to the next instruction after
			// each call. Adjust by -1 to land somewhere on the actual call.
			addr--
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

		p.Sample = append(p.Sample, &Sample{
			Value:    value,
			Location: sloc,
			NumLabel: map[string][]int64{"bytes": {blocksize}},
		})
	}

	if err = parseAdditionalSections(l, r, p); err != nil {
		return nil, err
	}
	return p, nil
}

// parseHeapSample parses a single row from a heap profile into a new Sample.
func parseHeapSample(line string, rate int64, sampling string) (value []int64, blocksize int64, addrs []uint64, err error) {
	sampleData := heapSampleRE.FindStringSubmatch(line)
	if len(sampleData) != 6 {
		return value, blocksize, addrs, fmt.Errorf("unexpected number of sample values: got %d, want 6", len(sampleData))
	}

	// Use first two values by default; tcmalloc sampling generates the
	// same value for both, only the older heap-profile collect separate
	// stats for in-use and allocated objects.
	valueIndex := 1
	if LegacyHeapAllocated {
		valueIndex = 3
	}

	var v1, v2 int64
	if v1, err = strconv.ParseInt(sampleData[valueIndex], 10, 64); err != nil {
		return value, blocksize, addrs, fmt.Errorf("malformed sample: %s: %v", line, err)
	}
	if v2, err = strconv.ParseInt(sampleData[valueIndex+1], 10, 64); err != nil {
		return value, blocksize, addrs, fmt.Errorf("malformed sample: %s: %v", line, err)
	}

	if v1 == 0 {
		if v2 != 0 {
			return value, blocksize, addrs, fmt.Errorf("allocation count was 0 but allocation bytes was %d", v2)
		}
	} else {
		blocksize = v2 / v1
		if sampling == "v2" {
			v1, v2 = scaleHeapSample(v1, v2, rate)
		}
	}

	value = []int64{v1, v2}
	addrs = parseHexAddresses(sampleData[5])

	return value, blocksize, addrs, nil
}

// extractHexAddresses extracts hex numbers from a string and returns
// them, together with their numeric value, in a slice.
func extractHexAddresses(s string) ([]string, []uint64) {
	hexStrings := hexNumberRE.FindAllString(s, -1)
	var ids []uint64
	for _, s := range hexStrings {
		if id, err := strconv.ParseUint(s, 0, 64); err == nil {
			ids = append(ids, id)
		} else {
			// Do not expect any parsing failures due to the regexp matching.
			panic("failed to parse hex value:" + s)
		}
	}
	return hexStrings, ids
}

// parseHexAddresses parses hex numbers from a string and returns them
// in a slice.
func parseHexAddresses(s string) []uint64 {
	_, ids := extractHexAddresses(s)
	return ids
}

// scaleHeapSample adjusts the data from a heapz Sample to
// account for its probability of appearing in the collected
// data. heapz profiles are a sampling of the memory allocations
// requests in a program. We estimate the unsampled value by dividing
// each collected sample by its probability of appearing in the
// profile. heapz v2 profiles rely on a poisson process to determine
// which samples to collect, based on the desired average collection
// rate R. The probability of a sample of size S to appear in that
// profile is 1-exp(-S/R).
func scaleHeapSample(count, size, rate int64) (int64, int64) {
	if count == 0 || size == 0 {
		return 0, 0
	}

	if rate <= 1 {
		// if rate==1 all samples were collected so no adjustment is needed.
		// if rate<1 treat as unknown and skip scaling.
		return count, size
	}

	avgSize := float64(size) / float64(count)
	scale := 1 / (1 - math.Exp(-avgSize/float64(rate)))

	return int64(float64(count) * scale), int64(float64(size) * scale)
}

// parseContention parses a contentionz profile and returns a newly
// populated Profile.
func parseContention(b []byte) (p *Profile, err error) {
	r := bytes.NewBuffer(b)
	l, err := r.ReadString('\n')
	if err != nil {
		return nil, errUnrecognized
	}

	if !strings.HasPrefix(l, "--- contention") {
		return nil, errUnrecognized
	}

	p = &Profile{
		PeriodType: &ValueType{Type: "contentions", Unit: "count"},
		Period:     1,
		SampleType: []*ValueType{
			{Type: "contentions", Unit: "count"},
			{Type: "delay", Unit: "nanoseconds"},
		},
	}

	var cpuHz int64
	// Parse text of the form "attribute = value" before the samples.
	const delimiter = "="
	for {
		l, err = r.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return nil, err
			}

			if l == "" {
				break
			}
		}

		if l = strings.TrimSpace(l); l == "" {
			continue
		}

		if strings.HasPrefix(l, "---") {
			break
		}

		attr := strings.SplitN(l, delimiter, 2)
		if len(attr) != 2 {
			break
		}
		key, val := strings.TrimSpace(attr[0]), strings.TrimSpace(attr[1])
		var err error
		switch key {
		case "cycles/second":
			if cpuHz, err = strconv.ParseInt(val, 0, 64); err != nil {
				return nil, errUnrecognized
			}
		case "sampling period":
			if p.Period, err = strconv.ParseInt(val, 0, 64); err != nil {
				return nil, errUnrecognized
			}
		case "ms since reset":
			ms, err := strconv.ParseInt(val, 0, 64)
			if err != nil {
				return nil, errUnrecognized
			}
			p.DurationNanos = ms * 1000 * 1000
		case "format":
			// CPP contentionz profiles don't have format.
			return nil, errUnrecognized
		case "resolution":
			// CPP contentionz profiles don't have resolution.
			return nil, errUnrecognized
		case "discarded samples":
		default:
			return nil, errUnrecognized
		}
	}

	locs := make(map[uint64]*Location)
	for {
		if l = strings.TrimSpace(l); strings.HasPrefix(l, "---") {
			break
		}
		value, addrs, err := parseContentionSample(l, p.Period, cpuHz)
		if err != nil {
			return nil, err
		}
		var sloc []*Location
		for _, addr := range addrs {
			// Addresses from stack traces point to the next instruction after
			// each call. Adjust by -1 to land somewhere on the actual call.
			addr--
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
		p.Sample = append(p.Sample, &Sample{
			Value:    value,
			Location: sloc,
		})

		if l, err = r.ReadString('\n'); err != nil {
			if err != io.EOF {
				return nil, err
			}
			if l == "" {
				break
			}
		}
	}

	if err = parseAdditionalSections(l, r, p); err != nil {
		return nil, err
	}

	return p, nil
}

// parseContentionSample parses a single row from a contention profile
// into a new Sample.
func parseContentionSample(line string, period, cpuHz int64) (value []int64, addrs []uint64, err error) {
	sampleData := contentionSampleRE.FindStringSubmatch(line)
	if sampleData == nil {
		return value, addrs, errUnrecognized
	}

	v1, err := strconv.ParseInt(sampleData[1], 10, 64)
	if err != nil {
		return value, addrs, fmt.Errorf("malformed sample: %s: %v", line, err)
	}
	v2, err := strconv.ParseInt(sampleData[2], 10, 64)
	if err != nil {
		return value, addrs, fmt.Errorf("malformed sample: %s: %v", line, err)
	}

	// Unsample values if period and cpuHz are available.
	// - Delays are scaled to cycles and then to nanoseconds.
	// - Contentions are scaled to cycles.
	if period > 0 {
		if cpuHz > 0 {
			cpuGHz := float64(cpuHz) / 1e9
			v1 = int64(float64(v1) * float64(period) / cpuGHz)
		}
		v2 = v2 * period
	}

	value = []int64{v2, v1}
	addrs = parseHexAddresses(sampleData[3])

	return value, addrs, nil
}

// parseThread parses a Threadz profile and returns a new Profile.
func parseThread(b []byte) (*Profile, error) {
	r := bytes.NewBuffer(b)

	var line string
	var err error
	for {
		// Skip past comments and empty lines seeking a real header.
		line, err = r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		if !isSpaceOrComment(line) {
			break
		}
	}

	if m := threadzStartRE.FindStringSubmatch(line); m != nil {
		// Advance over initial comments until first stack trace.
		for {
			line, err = r.ReadString('\n')
			if err != nil {
				if err != io.EOF {
					return nil, err
				}

				if line == "" {
					break
				}
			}
			if sectionTrigger(line) != unrecognizedSection || line[0] == '-' {
				break
			}
		}
	} else if t := threadStartRE.FindStringSubmatch(line); len(t) != 4 {
		return nil, errUnrecognized
	}

	p := &Profile{
		SampleType: []*ValueType{{Type: "thread", Unit: "count"}},
		PeriodType: &ValueType{Type: "thread", Unit: "count"},
		Period:     1,
	}

	locs := make(map[uint64]*Location)
	// Recognize each thread and populate profile samples.
	for sectionTrigger(line) == unrecognizedSection {
		if strings.HasPrefix(line, "---- no stack trace for") {
			line = ""
			break
		}
		if t := threadStartRE.FindStringSubmatch(line); len(t) != 4 {
			return nil, errUnrecognized
		}

		var addrs []uint64
		line, addrs, err = parseThreadSample(r)
		if err != nil {
			return nil, errUnrecognized
		}
		if len(addrs) == 0 {
			// We got a --same as previous threads--. Bump counters.
			if len(p.Sample) > 0 {
				s := p.Sample[len(p.Sample)-1]
				s.Value[0]++
			}
			continue
		}

		var sloc []*Location
		for _, addr := range addrs {
			// Addresses from stack traces point to the next instruction after
			// each call. Adjust by -1 to land somewhere on the actual call.
			addr--
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

		p.Sample = append(p.Sample, &Sample{
			Value:    []int64{1},
			Location: sloc,
		})
	}

	if err = parseAdditionalSections(line, r, p); err != nil {
		return nil, err
	}

	return p, nil
}

// parseThreadSample parses a symbolized or unsymbolized stack trace.
// Returns the first line after the traceback, the sample (or nil if
// it hits a 'same-as-previous' marker) and an error.
func parseThreadSample(b *bytes.Buffer) (nextl string, addrs []uint64, err error) {
	var l string
	sameAsPrevious := false
	for {
		if l, err = b.ReadString('\n'); err != nil {
			if err != io.EOF {
				return "", nil, err
			}
			if l == "" {
				break
			}
		}
		if l = strings.TrimSpace(l); l == "" {
			continue
		}

		if strings.HasPrefix(l, "---") {
			break
		}
		if strings.Contains(l, "same as previous thread") {
			sameAsPrevious = true
			continue
		}

		addrs = append(addrs, parseHexAddresses(l)...)
	}

	if sameAsPrevious {
		return l, nil, nil
	}
	return l, addrs, nil
}

// parseAdditionalSections parses any additional sections in the
// profile, ignoring any unrecognized sections.
func parseAdditionalSections(l string, b *bytes.Buffer, p *Profile) (err error) {
	for {
		if sectionTrigger(l) == memoryMapSection {
			break
		}
		// Ignore any unrecognized sections.
		if l, err := b.ReadString('\n'); err != nil {
			if err != io.EOF {
				return err
			}
			if l == "" {
				break
			}
		}
	}
	return p.ParseMemoryMap(b)
}

// ParseMemoryMap parses a memory map in the format of
// /proc/self/maps, and overrides the mappings in the current profile.
// It renumbers the samples and locations in the profile correspondingly.
func (p *Profile) ParseMemoryMap(rd io.Reader) error {
	b := bufio.NewReader(rd)

	var attrs []string
	var r *strings.Replacer
	const delimiter = "="
	for {
		l, err := b.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return err
			}
			if l == "" {
				break
			}
		}
		if l = strings.TrimSpace(l); l == "" {
			continue
		}

		if r != nil {
			l = r.Replace(l)
		}
		m, err := parseMappingEntry(l)
		if err != nil {
			if err == errUnrecognized {
				// Recognize assignments of the form: attr=value, and replace
				// $attr with value on subsequent mappings.
				if attr := strings.SplitN(l, delimiter, 2); len(attr) == 2 {
					attrs = append(attrs, "$"+strings.TrimSpace(attr[0]), strings.TrimSpace(attr[1]))
					r = strings.NewReplacer(attrs...)
				}
				// Ignore any unrecognized entries
				continue
			}
			return err
		}
		if m == nil || (m.File == "" && len(p.Mapping) != 0) {
			// In some cases the first entry may include the address range
			// but not the name of the file. It should be followed by
			// another entry with the name.
			continue
		}
		if len(p.Mapping) == 1 && p.Mapping[0].File == "" {
			// Update the name if this is the entry following that empty one.
			p.Mapping[0].File = m.File
			continue
		}
		p.Mapping = append(p.Mapping, m)
	}
	p.remapLocationIDs()
	p.remapFunctionIDs()
	p.remapMappingIDs()
	return nil
}

func parseMappingEntry(l string) (*Mapping, error) {
	mapping := &Mapping{}
	var err error
	if me := procMapsRE.FindStringSubmatch(l); len(me) == 9 {
		if !strings.Contains(me[3], "x") {
			// Skip non-executable entries.
			return nil, nil
		}
		if mapping.Start, err = strconv.ParseUint(me[1], 16, 64); err != nil {
			return nil, errUnrecognized
		}
		if mapping.Limit, err = strconv.ParseUint(me[2], 16, 64); err != nil {
			return nil, errUnrecognized
		}
		if me[4] != "" {
			if mapping.Offset, err = strconv.ParseUint(me[4], 16, 64); err != nil {
				return nil, errUnrecognized
			}
		}
		mapping.File = me[8]
		return mapping, nil
	}

	if me := briefMapsRE.FindStringSubmatch(l); len(me) == 6 {
		if mapping.Start, err = strconv.ParseUint(me[1], 16, 64); err != nil {
			return nil, errUnrecognized
		}
		if mapping.Limit, err = strconv.ParseUint(me[2], 16, 64); err != nil {
			return nil, errUnrecognized
		}
		mapping.File = me[3]
		if me[5] != "" {
			if mapping.Offset, err = strconv.ParseUint(me[5], 16, 64); err != nil {
				return nil, errUnrecognized
			}
		}
		return mapping, nil
	}

	return nil, errUnrecognized
}

type sectionType int

const (
	unrecognizedSection sectionType = iota
	memoryMapSection
)

var memoryMapTriggers = []string{
	"--- Memory map: ---",
	"MAPPED_LIBRARIES:",
}

func sectionTrigger(line string) sectionType {
	for _, trigger := range memoryMapTriggers {
		if strings.Contains(line, trigger) {
			return memoryMapSection
		}
	}
	return unrecognizedSection
}

func (p *Profile) addLegacyFrameInfo() {
	switch {
	case isProfileType(p, heapzSampleTypes) ||
		isProfileType(p, heapzInUseSampleTypes) ||
		isProfileType(p, heapzAllocSampleTypes):
		p.DropFrames, p.KeepFrames = allocRxStr, allocSkipRxStr
	case isProfileType(p, contentionzSampleTypes):
		p.DropFrames, p.KeepFrames = lockRxStr, ""
	default:
		p.DropFrames, p.KeepFrames = cpuProfilerRxStr, ""
	}
}

var heapzSampleTypes = []string{"allocations", "size"} // early Go pprof profiles
var heapzInUseSampleTypes = []string{"inuse_objects", "inuse_space"}
var heapzAllocSampleTypes = []string{"alloc_objects", "alloc_space"}
var contentionzSampleTypes = []string{"contentions", "delay"}

func isProfileType(p *Profile, t []string) bool {
	st := p.SampleType
	if len(st) != len(t) {
		return false
	}

	for i := range st {
		if st[i].Type != t[i] {
			return false
		}
	}
	return true
}

var allocRxStr = strings.Join([]string{
	// POSIX entry points.
	`calloc`,
	`cfree`,
	`malloc`,
	`free`,
	`memalign`,
	`do_memalign`,
	`(__)?posix_memalign`,
	`pvalloc`,
	`valloc`,
	`realloc`,

	// TC malloc.
	`tcmalloc::.*`,
	`tc_calloc`,
	`tc_cfree`,
	`tc_malloc`,
	`tc_free`,
	`tc_memalign`,
	`tc_posix_memalign`,
	`tc_pvalloc`,
	`tc_valloc`,
	`tc_realloc`,
	`tc_new`,
	`tc_delete`,
	`tc_newarray`,
	`tc_deletearray`,
	`tc_new_nothrow`,
	`tc_newarray_nothrow`,

	// Memory-allocation routines on OS X.
	`malloc_zone_malloc`,
	`malloc_zone_calloc`,
	`malloc_zone_valloc`,
	`malloc_zone_realloc`,
	`malloc_zone_memalign`,
	`malloc_zone_free`,

	// Go runtime
	`runtime\..*`,

	// Other misc. memory allocation routines
	`BaseArena::.*`,
	`(::)?do_malloc_no_errno`,
	`(::)?do_malloc_pages`,
	`(::)?do_malloc`,
	`DoSampledAllocation`,
	`MallocedMemBlock::MallocedMemBlock`,
	`_M_allocate`,
	`__builtin_(vec_)?delete`,
	`__builtin_(vec_)?new`,
	`__gnu_cxx::new_allocator::allocate`,
	`__libc_malloc`,
	`__malloc_alloc_template::allocate`,
	`allocate`,
	`cpp_alloc`,
	`operator new(\[\])?`,
	`simple_alloc::allocate`,
}, `|`)

var allocSkipRxStr = strings.Join([]string{
	// Preserve Go runtime frames that appear in the middle/bottom of
	// the stack.
	`runtime\.panic`,
}, `|`)

var cpuProfilerRxStr = strings.Join([]string{
	`ProfileData::Add`,
	`ProfileData::prof_handler`,
	`CpuProfiler::prof_handler`,
	`__pthread_sighandler`,
	`__restore`,
}, `|`)

var lockRxStr = strings.Join([]string{
	`RecordLockProfileData`,
	`(base::)?RecordLockProfileData.*`,
	`(base::)?SubmitMutexProfileData.*`,
	`(base::)?SubmitSpinLockProfileData.*`,
	`(Mutex::)?AwaitCommon.*`,
	`(Mutex::)?Unlock.*`,
	`(Mutex::)?UnlockSlow.*`,
	`(Mutex::)?ReaderUnlock.*`,
	`(MutexLock::)?~MutexLock.*`,
	`(SpinLock::)?Unlock.*`,
	`(SpinLock::)?SlowUnlock.*`,
	`(SpinLockHolder::)?~SpinLockHolder.*`,
}, `|`)
