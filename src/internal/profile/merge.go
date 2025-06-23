// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package profile

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Merge merges all the profiles in profs into a single Profile.
// Returns a new profile independent of the input profiles. The merged
// profile is compacted to eliminate unused samples, locations,
// functions and mappings. Profiles must have identical profile sample
// and period types or the merge will fail. profile.Period of the
// resulting profile will be the maximum of all profiles, and
// profile.TimeNanos will be the earliest nonzero one.
func Merge(srcs []*Profile) (*Profile, error) {
	if len(srcs) == 0 {
		return nil, fmt.Errorf("no profiles to merge")
	}
	p, err := combineHeaders(srcs)
	if err != nil {
		return nil, err
	}

	pm := &profileMerger{
		p:         p,
		samples:   make(map[sampleKey]*Sample, len(srcs[0].Sample)),
		locations: make(map[locationKey]*Location, len(srcs[0].Location)),
		functions: make(map[functionKey]*Function, len(srcs[0].Function)),
		mappings:  make(map[mappingKey]*Mapping, len(srcs[0].Mapping)),
	}

	for _, src := range srcs {
		// Clear the profile-specific hash tables
		pm.locationsByID = make(map[uint64]*Location, len(src.Location))
		pm.functionsByID = make(map[uint64]*Function, len(src.Function))
		pm.mappingsByID = make(map[uint64]mapInfo, len(src.Mapping))

		if len(pm.mappings) == 0 && len(src.Mapping) > 0 {
			// The Mapping list has the property that the first mapping
			// represents the main binary. Take the first Mapping we see,
			// otherwise the operations below will add mappings in an
			// arbitrary order.
			pm.mapMapping(src.Mapping[0])
		}

		for _, s := range src.Sample {
			if !isZeroSample(s) {
				pm.mapSample(s)
			}
		}
	}

	for _, s := range p.Sample {
		if isZeroSample(s) {
			// If there are any zero samples, re-merge the profile to GC
			// them.
			return Merge([]*Profile{p})
		}
	}

	return p, nil
}

// Normalize normalizes the source profile by multiplying each value in profile by the
// ratio of the sum of the base profile's values of that sample type to the sum of the
// source profile's value of that sample type.
func (p *Profile) Normalize(pb *Profile) error {

	if err := p.compatible(pb); err != nil {
		return err
	}

	baseVals := make([]int64, len(p.SampleType))
	for _, s := range pb.Sample {
		for i, v := range s.Value {
			baseVals[i] += v
		}
	}

	srcVals := make([]int64, len(p.SampleType))
	for _, s := range p.Sample {
		for i, v := range s.Value {
			srcVals[i] += v
		}
	}

	normScale := make([]float64, len(baseVals))
	for i := range baseVals {
		if srcVals[i] == 0 {
			normScale[i] = 0.0
		} else {
			normScale[i] = float64(baseVals[i]) / float64(srcVals[i])
		}
	}
	p.ScaleN(normScale)
	return nil
}

func isZeroSample(s *Sample) bool {
	for _, v := range s.Value {
		if v != 0 {
			return false
		}
	}
	return true
}

type profileMerger struct {
	p *Profile

	// Memoization tables within a profile.
	locationsByID map[uint64]*Location
	functionsByID map[uint64]*Function
	mappingsByID  map[uint64]mapInfo

	// Memoization tables for profile entities.
	samples   map[sampleKey]*Sample
	locations map[locationKey]*Location
	functions map[functionKey]*Function
	mappings  map[mappingKey]*Mapping
}

type mapInfo struct {
	m      *Mapping
	offset int64
}

func (pm *profileMerger) mapSample(src *Sample) *Sample {
	s := &Sample{
		Location: make([]*Location, len(src.Location)),
		Value:    make([]int64, len(src.Value)),
		Label:    make(map[string][]string, len(src.Label)),
		NumLabel: make(map[string][]int64, len(src.NumLabel)),
		NumUnit:  make(map[string][]string, len(src.NumLabel)),
	}
	for i, l := range src.Location {
		s.Location[i] = pm.mapLocation(l)
	}
	for k, v := range src.Label {
		vv := make([]string, len(v))
		copy(vv, v)
		s.Label[k] = vv
	}
	for k, v := range src.NumLabel {
		u := src.NumUnit[k]
		vv := make([]int64, len(v))
		uu := make([]string, len(u))
		copy(vv, v)
		copy(uu, u)
		s.NumLabel[k] = vv
		s.NumUnit[k] = uu
	}
	// Check memoization table. Must be done on the remapped location to
	// account for the remapped mapping. Add current values to the
	// existing sample.
	k := s.key()
	if ss, ok := pm.samples[k]; ok {
		for i, v := range src.Value {
			ss.Value[i] += v
		}
		return ss
	}
	copy(s.Value, src.Value)
	pm.samples[k] = s
	pm.p.Sample = append(pm.p.Sample, s)
	return s
}

// key generates sampleKey to be used as a key for maps.
func (sample *Sample) key() sampleKey {
	ids := make([]string, len(sample.Location))
	for i, l := range sample.Location {
		ids[i] = strconv.FormatUint(l.ID, 16)
	}

	labels := make([]string, 0, len(sample.Label))
	for k, v := range sample.Label {
		labels = append(labels, fmt.Sprintf("%q%q", k, v))
	}
	sort.Strings(labels)

	numlabels := make([]string, 0, len(sample.NumLabel))
	for k, v := range sample.NumLabel {
		numlabels = append(numlabels, fmt.Sprintf("%q%x%x", k, v, sample.NumUnit[k]))
	}
	sort.Strings(numlabels)

	return sampleKey{
		strings.Join(ids, "|"),
		strings.Join(labels, ""),
		strings.Join(numlabels, ""),
	}
}

type sampleKey struct {
	locations string
	labels    string
	numlabels string
}

func (pm *profileMerger) mapLocation(src *Location) *Location {
	if src == nil {
		return nil
	}

	if l, ok := pm.locationsByID[src.ID]; ok {
		pm.locationsByID[src.ID] = l
		return l
	}

	mi := pm.mapMapping(src.Mapping)
	l := &Location{
		ID:       uint64(len(pm.p.Location) + 1),
		Mapping:  mi.m,
		Address:  uint64(int64(src.Address) + mi.offset),
		Line:     make([]Line, len(src.Line)),
		IsFolded: src.IsFolded,
	}
	for i, ln := range src.Line {
		l.Line[i] = pm.mapLine(ln)
	}
	// Check memoization table. Must be done on the remapped location to
	// account for the remapped mapping ID.
	k := l.key()
	if ll, ok := pm.locations[k]; ok {
		pm.locationsByID[src.ID] = ll
		return ll
	}
	pm.locationsByID[src.ID] = l
	pm.locations[k] = l
	pm.p.Location = append(pm.p.Location, l)
	return l
}

// key generates locationKey to be used as a key for maps.
func (l *Location) key() locationKey {
	key := locationKey{
		addr:     l.Address,
		isFolded: l.IsFolded,
	}
	if l.Mapping != nil {
		// Normalizes address to handle address space randomization.
		key.addr -= l.Mapping.Start
		key.mappingID = l.Mapping.ID
	}
	lines := make([]string, len(l.Line)*2)
	for i, line := range l.Line {
		if line.Function != nil {
			lines[i*2] = strconv.FormatUint(line.Function.ID, 16)
		}
		lines[i*2+1] = strconv.FormatInt(line.Line, 16)
	}
	key.lines = strings.Join(lines, "|")
	return key
}

type locationKey struct {
	addr, mappingID uint64
	lines           string
	isFolded        bool
}

func (pm *profileMerger) mapMapping(src *Mapping) mapInfo {
	if src == nil {
		return mapInfo{}
	}

	if mi, ok := pm.mappingsByID[src.ID]; ok {
		return mi
	}

	// Check memoization tables.
	mk := src.key()
	if m, ok := pm.mappings[mk]; ok {
		mi := mapInfo{m, int64(m.Start) - int64(src.Start)}
		pm.mappingsByID[src.ID] = mi
		return mi
	}
	m := &Mapping{
		ID:              uint64(len(pm.p.Mapping) + 1),
		Start:           src.Start,
		Limit:           src.Limit,
		Offset:          src.Offset,
		File:            src.File,
		BuildID:         src.BuildID,
		HasFunctions:    src.HasFunctions,
		HasFilenames:    src.HasFilenames,
		HasLineNumbers:  src.HasLineNumbers,
		HasInlineFrames: src.HasInlineFrames,
	}
	pm.p.Mapping = append(pm.p.Mapping, m)

	// Update memoization tables.
	pm.mappings[mk] = m
	mi := mapInfo{m, 0}
	pm.mappingsByID[src.ID] = mi
	return mi
}

// key generates encoded strings of Mapping to be used as a key for
// maps.
func (m *Mapping) key() mappingKey {
	// Normalize addresses to handle address space randomization.
	// Round up to next 4K boundary to avoid minor discrepancies.
	const mapsizeRounding = 0x1000

	size := m.Limit - m.Start
	size = size + mapsizeRounding - 1
	size = size - (size % mapsizeRounding)
	key := mappingKey{
		size:   size,
		offset: m.Offset,
	}

	switch {
	case m.BuildID != "":
		key.buildIDOrFile = m.BuildID
	case m.File != "":
		key.buildIDOrFile = m.File
	default:
		// A mapping containing neither build ID nor file name is a fake mapping. A
		// key with empty buildIDOrFile is used for fake mappings so that they are
		// treated as the same mapping during merging.
	}
	return key
}

type mappingKey struct {
	size, offset  uint64
	buildIDOrFile string
}

func (pm *profileMerger) mapLine(src Line) Line {
	ln := Line{
		Function: pm.mapFunction(src.Function),
		Line:     src.Line,
	}
	return ln
}

func (pm *profileMerger) mapFunction(src *Function) *Function {
	if src == nil {
		return nil
	}
	if f, ok := pm.functionsByID[src.ID]; ok {
		return f
	}
	k := src.key()
	if f, ok := pm.functions[k]; ok {
		pm.functionsByID[src.ID] = f
		return f
	}
	f := &Function{
		ID:         uint64(len(pm.p.Function) + 1),
		Name:       src.Name,
		SystemName: src.SystemName,
		Filename:   src.Filename,
		StartLine:  src.StartLine,
	}
	pm.functions[k] = f
	pm.functionsByID[src.ID] = f
	pm.p.Function = append(pm.p.Function, f)
	return f
}

// key generates a struct to be used as a key for maps.
func (f *Function) key() functionKey {
	return functionKey{
		f.StartLine,
		f.Name,
		f.SystemName,
		f.Filename,
	}
}

type functionKey struct {
	startLine                  int64
	name, systemName, fileName string
}

// combineHeaders checks that all profiles can be merged and returns
// their combined profile.
func combineHeaders(srcs []*Profile) (*Profile, error) {
	for _, s := range srcs[1:] {
		if err := srcs[0].compatible(s); err != nil {
			return nil, err
		}
	}

	var timeNanos, durationNanos, period int64
	var comments []string
	seenComments := map[string]bool{}
	var defaultSampleType string
	for _, s := range srcs {
		if timeNanos == 0 || s.TimeNanos < timeNanos {
			timeNanos = s.TimeNanos
		}
		durationNanos += s.DurationNanos
		if period == 0 || period < s.Period {
			period = s.Period
		}
		for _, c := range s.Comments {
			if seen := seenComments[c]; !seen {
				comments = append(comments, c)
				seenComments[c] = true
			}
		}
		if defaultSampleType == "" {
			defaultSampleType = s.DefaultSampleType
		}
	}

	p := &Profile{
		SampleType: make([]*ValueType, len(srcs[0].SampleType)),

		DropFrames: srcs[0].DropFrames,
		KeepFrames: srcs[0].KeepFrames,

		TimeNanos:     timeNanos,
		DurationNanos: durationNanos,
		PeriodType:    srcs[0].PeriodType,
		Period:        period,

		Comments:          comments,
		DefaultSampleType: defaultSampleType,
	}
	copy(p.SampleType, srcs[0].SampleType)
	return p, nil
}

// compatible determines if two profiles can be compared/merged.
// returns nil if the profiles are compatible; otherwise an error with
// details on the incompatibility.
func (p *Profile) compatible(pb *Profile) error {
	if !equalValueType(p.PeriodType, pb.PeriodType) {
		return fmt.Errorf("incompatible period types %v and %v", p.PeriodType, pb.PeriodType)
	}

	if len(p.SampleType) != len(pb.SampleType) {
		return fmt.Errorf("incompatible sample types %v and %v", p.SampleType, pb.SampleType)
	}

	for i := range p.SampleType {
		if !equalValueType(p.SampleType[i], pb.SampleType[i]) {
			return fmt.Errorf("incompatible sample types %v and %v", p.SampleType, pb.SampleType)
		}
	}
	return nil
}

// equalValueType returns true if the two value types are semantically
// equal. It ignores the internal fields used during encode/decode.
func equalValueType(st1, st2 *ValueType) bool {
	return st1.Type == st2.Type && st1.Unit == st2.Unit
}
