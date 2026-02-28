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
	"encoding/binary"
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// Compact performs garbage collection on a profile to remove any
// unreferenced fields. This is useful to reduce the size of a profile
// after samples or locations have been removed.
func (p *Profile) Compact() *Profile {
	p, _ = Merge([]*Profile{p})
	return p
}

// Merge merges all the profiles in profs into a single Profile.
// Returns a new profile independent of the input profiles. The merged
// profile is compacted to eliminate unused samples, locations,
// functions and mappings. Profiles must have identical profile sample
// and period types or the merge will fail. profile.Period of the
// resulting profile will be the maximum of all profiles, and
// profile.TimeNanos will be the earliest nonzero one. Merges are
// associative with the caveat of the first profile having some
// specialization in how headers are combined. There may be other
// subtleties now or in the future regarding associativity.
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
		pm.locationsByID = makeLocationIDMap(len(src.Location))
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

	if slices.ContainsFunc(p.Sample, isZeroSample) {
		// If there are any zero samples, re-merge the profile to GC
		// them.
		return Merge([]*Profile{p})
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
	locationsByID locationIDMap
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
	// Check memoization table
	k := pm.sampleKey(src)
	if ss, ok := pm.samples[k]; ok {
		for i, v := range src.Value {
			ss.Value[i] += v
		}
		return ss
	}

	// Make new sample.
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
	copy(s.Value, src.Value)
	pm.samples[k] = s
	pm.p.Sample = append(pm.p.Sample, s)
	return s
}

func (pm *profileMerger) sampleKey(sample *Sample) sampleKey {
	// Accumulate contents into a string.
	var buf strings.Builder
	buf.Grow(64) // Heuristic to avoid extra allocs

	// encode a number
	putNumber := func(v uint64) {
		var num [binary.MaxVarintLen64]byte
		n := binary.PutUvarint(num[:], v)
		buf.Write(num[:n])
	}

	// encode a string prefixed with its length.
	putDelimitedString := func(s string) {
		putNumber(uint64(len(s)))
		buf.WriteString(s)
	}

	for _, l := range sample.Location {
		// Get the location in the merged profile, which may have a different ID.
		if loc := pm.mapLocation(l); loc != nil {
			putNumber(loc.ID)
		}
	}
	putNumber(0) // Delimiter

	for _, l := range sortedKeys1(sample.Label) {
		putDelimitedString(l)
		values := sample.Label[l]
		putNumber(uint64(len(values)))
		for _, v := range values {
			putDelimitedString(v)
		}
	}

	for _, l := range sortedKeys2(sample.NumLabel) {
		putDelimitedString(l)
		values := sample.NumLabel[l]
		putNumber(uint64(len(values)))
		for _, v := range values {
			putNumber(uint64(v))
		}
		units := sample.NumUnit[l]
		putNumber(uint64(len(units)))
		for _, v := range units {
			putDelimitedString(v)
		}
	}

	return sampleKey(buf.String())
}

type sampleKey string

// sortedKeys1 returns the sorted keys found in a string->[]string map.
//
// Note: this is currently non-generic since github pprof runs golint,
// which does not support generics. When that issue is fixed, it can
// be merged with sortedKeys2 and made into a generic function.
func sortedKeys1(m map[string][]string) []string {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// sortedKeys2 returns the sorted keys found in a string->[]int64 map.
//
// Note: this is currently non-generic since github pprof runs golint,
// which does not support generics. When that issue is fixed, it can
// be merged with sortedKeys1 and made into a generic function.
func sortedKeys2(m map[string][]int64) []string {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func (pm *profileMerger) mapLocation(src *Location) *Location {
	if src == nil {
		return nil
	}

	if l := pm.locationsByID.get(src.ID); l != nil {
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
		pm.locationsByID.set(src.ID, ll)
		return ll
	}
	pm.locationsByID.set(src.ID, l)
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
	lines := make([]string, len(l.Line)*3)
	for i, line := range l.Line {
		if line.Function != nil {
			lines[i*2] = strconv.FormatUint(line.Function.ID, 16)
		}
		lines[i*2+1] = strconv.FormatInt(line.Line, 16)
		lines[i*2+2] = strconv.FormatInt(line.Column, 16)
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
		ID:                     uint64(len(pm.p.Mapping) + 1),
		Start:                  src.Start,
		Limit:                  src.Limit,
		Offset:                 src.Offset,
		File:                   src.File,
		KernelRelocationSymbol: src.KernelRelocationSymbol,
		BuildID:                src.BuildID,
		HasFunctions:           src.HasFunctions,
		HasFilenames:           src.HasFilenames,
		HasLineNumbers:         src.HasLineNumbers,
		HasInlineFrames:        src.HasInlineFrames,
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
		Column:   src.Column,
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
	var docURL string
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
		if docURL == "" {
			docURL = s.DocURL
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
		DocURL:            docURL,
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

// locationIDMap is like a map[uint64]*Location, but provides efficiency for
// ids that are densely numbered, which is often the case.
type locationIDMap struct {
	dense  []*Location          // indexed by id for id < len(dense)
	sparse map[uint64]*Location // indexed by id for id >= len(dense)
}

func makeLocationIDMap(n int) locationIDMap {
	return locationIDMap{
		dense:  make([]*Location, n),
		sparse: map[uint64]*Location{},
	}
}

func (lm locationIDMap) get(id uint64) *Location {
	if id < uint64(len(lm.dense)) {
		return lm.dense[int(id)]
	}
	return lm.sparse[id]
}

func (lm locationIDMap) set(id uint64, loc *Location) {
	if id < uint64(len(lm.dense)) {
		lm.dense[id] = loc
		return
	}
	lm.sparse[id] = loc
}

// CompatibilizeSampleTypes makes profiles compatible to be compared/merged. It
// keeps sample types that appear in all profiles only and drops/reorders the
// sample types as necessary.
//
// In the case of sample types order is not the same for given profiles the
// order is derived from the first profile.
//
// Profiles are modified in-place.
//
// It returns an error if the sample type's intersection is empty.
func CompatibilizeSampleTypes(ps []*Profile) error {
	sTypes := commonSampleTypes(ps)
	if len(sTypes) == 0 {
		return fmt.Errorf("profiles have empty common sample type list")
	}
	for _, p := range ps {
		if err := compatibilizeSampleTypes(p, sTypes); err != nil {
			return err
		}
	}
	return nil
}

// commonSampleTypes returns sample types that appear in all profiles in the
// order how they ordered in the first profile.
func commonSampleTypes(ps []*Profile) []string {
	if len(ps) == 0 {
		return nil
	}
	sTypes := map[string]int{}
	for _, p := range ps {
		for _, st := range p.SampleType {
			sTypes[st.Type]++
		}
	}
	var res []string
	for _, st := range ps[0].SampleType {
		if sTypes[st.Type] == len(ps) {
			res = append(res, st.Type)
		}
	}
	return res
}

// compatibilizeSampleTypes drops sample types that are not present in sTypes
// list and reorder them if needed.
//
// It sets DefaultSampleType to sType[0] if it is not in sType list.
//
// It assumes that all sample types from the sTypes list are present in the
// given profile otherwise it returns an error.
func compatibilizeSampleTypes(p *Profile, sTypes []string) error {
	if len(sTypes) == 0 {
		return fmt.Errorf("sample type list is empty")
	}
	defaultSampleType := sTypes[0]
	reMap, needToModify := make([]int, len(sTypes)), false
	for i, st := range sTypes {
		if st == p.DefaultSampleType {
			defaultSampleType = p.DefaultSampleType
		}
		idx := searchValueType(p.SampleType, st)
		if idx < 0 {
			return fmt.Errorf("%q sample type is not found in profile", st)
		}
		reMap[i] = idx
		if idx != i {
			needToModify = true
		}
	}
	if !needToModify && len(sTypes) == len(p.SampleType) {
		return nil
	}
	p.DefaultSampleType = defaultSampleType
	oldSampleTypes := p.SampleType
	p.SampleType = make([]*ValueType, len(sTypes))
	for i, idx := range reMap {
		p.SampleType[i] = oldSampleTypes[idx]
	}
	values := make([]int64, len(sTypes))
	for _, s := range p.Sample {
		for i, idx := range reMap {
			values[i] = s.Value[idx]
		}
		s.Value = s.Value[:len(values)]
		copy(s.Value, values)
	}
	return nil
}

func searchValueType(vts []*ValueType, s string) int {
	for i, vt := range vts {
		if vt.Type == s {
			return i
		}
	}
	return -1
}
