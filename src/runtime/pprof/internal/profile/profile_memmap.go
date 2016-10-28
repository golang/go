// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package profile

import (
	"bufio"
	"errors"
	"io"
	"strconv"
	"strings"
)

var errUnrecognized = errors.New("unrecognized profile format")

func hasLibFile(file string) string {
	ix := strings.Index(file, "so")
	if ix < 1 {
		return ""
	}
	start := ix - 1
	end := ix + 2
	s := file[start:end]
	if end < len(file) {
		endalt := end
		if file[endalt] != '.' && file[endalt] != '_' {
			return s
		}
		endalt++
		for file[endalt] >= '0' && file[endalt] <= '9' {
			endalt++
		}
		if endalt < end+2 {
			return s
		}
		return s[start:endalt]
	}
	return s
}

// massageMappings applies heuristic-based changes to the profile
// mappings to account for quirks of some environments.
func (p *Profile) massageMappings() {
	// Merge adjacent regions with matching names, checking that the offsets match
	if len(p.Mapping) > 1 {
		mappings := []*Mapping{p.Mapping[0]}
		for _, m := range p.Mapping[1:] {
			lm := mappings[len(mappings)-1]
			if offset := lm.Offset + (lm.Limit - lm.Start); lm.Limit == m.Start &&
				offset == m.Offset &&
				(lm.File == m.File || lm.File == "") {
				lm.File = m.File
				lm.Limit = m.Limit
				if lm.BuildID == "" {
					lm.BuildID = m.BuildID
				}
				p.updateLocationMapping(m, lm)
				continue
			}
			mappings = append(mappings, m)
		}
		p.Mapping = mappings
	}

	// Use heuristics to identify main binary and move it to the top of the list of mappings
	for i, m := range p.Mapping {
		file := strings.TrimSpace(strings.Replace(m.File, "(deleted)", "", -1))
		if len(file) == 0 {
			continue
		}
		if len(hasLibFile(file)) > 0 {
			continue
		}
		if strings.HasPrefix(file, "[") {
			continue
		}
		// Swap what we guess is main to position 0.
		p.Mapping[0], p.Mapping[i] = p.Mapping[i], p.Mapping[0]
		break
	}

	// Keep the mapping IDs neatly sorted
	for i, m := range p.Mapping {
		m.ID = uint64(i + 1)
	}
}

func (p *Profile) updateLocationMapping(from, to *Mapping) {
	for _, l := range p.Location {
		if l.Mapping == from {
			l.Mapping = to
		}
	}
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
	// Some profile handlers will incorrectly set regions for the main
	// executable if its section is remapped. Fix them through heuristics.

	if len(p.Mapping) > 0 {
		// Remove the initial mapping if named '/anon_hugepage' and has a
		// consecutive adjacent mapping.
		if m := p.Mapping[0]; strings.HasPrefix(m.File, "/anon_hugepage") {
			if len(p.Mapping) > 1 && m.Limit == p.Mapping[1].Start {
				p.Mapping = p.Mapping[1:]
			}
		}
	}

	// Subtract the offset from the start of the main mapping if it
	// ends up at a recognizable start address.
	if len(p.Mapping) > 0 {
		const expectedStart = 0x400000
		if m := p.Mapping[0]; m.Start-m.Offset == expectedStart {
			m.Start = expectedStart
			m.Offset = 0
		}
	}

	// Associate each location with an address to the corresponding
	// mapping. Create fake mapping if a suitable one isn't found.
	var fake *Mapping
nextLocation:
	for _, l := range p.Location {
		a := l.Address
		if l.Mapping != nil || a == 0 {
			continue
		}
		for _, m := range p.Mapping {
			if m.Start <= a && a < m.Limit {
				l.Mapping = m
				continue nextLocation
			}
		}
		// Work around legacy handlers failing to encode the first
		// part of mappings split into adjacent ranges.
		for _, m := range p.Mapping {
			if m.Offset != 0 && m.Start-m.Offset <= a && a < m.Start {
				m.Start -= m.Offset
				m.Offset = 0
				l.Mapping = m
				continue nextLocation
			}
		}
		// If there is still no mapping, create a fake one.
		// This is important for the Go legacy handler, which produced
		// no mappings.
		if fake == nil {
			fake = &Mapping{
				ID:    1,
				Limit: ^uint64(0),
			}
			p.Mapping = append(p.Mapping, fake)
		}
		l.Mapping = fake
	}

	// Reset all mapping IDs.
	for i, m := range p.Mapping {
		m.ID = uint64(i + 1)
	}
}

func (p *Profile) RemapAll() {
	p.remapLocationIDs()
	p.remapFunctionIDs()
	p.remapMappingIDs()
}

// ParseProcMaps parses a memory map in the format of /proc/self/maps.
// ParseMemoryMap should be called after setting on a profile to
// associate locations to the corresponding mapping based on their
// address.
func ParseProcMaps(rd io.Reader) ([]*Mapping, error) {
	var mapping []*Mapping

	b := bufio.NewReader(rd)

	var attrs []string
	var r *strings.Replacer
	const delimiter = "="
	for {
		l, err := b.ReadString('\n')
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
			return nil, err
		}
		if m == nil {
			continue
		}
		mapping = append(mapping, m)
	}
	return mapping, nil
}

// ParseMemoryMap parses a memory map in the format of
// /proc/self/maps, and overrides the mappings in the current profile.
// It renumbers the samples and locations in the profile correspondingly.
func (p *Profile) ParseMemoryMap(rd io.Reader) error {
	mapping, err := ParseProcMaps(rd)
	if err != nil {
		return err
	}
	p.Mapping = append(p.Mapping, mapping...)
	p.massageMappings()
	p.RemapAll()
	return nil
}

func parseMappingEntry(l string) (*Mapping, error) {
	mapping := &Mapping{}
	var err error
	fields := strings.Fields(l)
	// fmt.Println(len(me), me)
	if len(fields) == 6 {
		if !strings.Contains(fields[1], "x") {
			// Skip non-executable entries.
			return nil, nil
		}
		addrRange := strings.Split(fields[0], "-")
		if mapping.Start, err = strconv.ParseUint(addrRange[0], 16, 64); err != nil {
			return nil, errUnrecognized
		}
		if mapping.Limit, err = strconv.ParseUint(addrRange[1], 16, 64); err != nil {
			return nil, errUnrecognized
		}
		offset := fields[2]
		if offset != "" {
			if mapping.Offset, err = strconv.ParseUint(offset, 16, 64); err != nil {
				return nil, errUnrecognized
			}
		}
		mapping.File = fields[5]
		return mapping, nil
	}

	return nil, errUnrecognized
}
