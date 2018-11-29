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

// Package profile provides a representation of profile.proto and
// methods to encode/decode profiles in this format.
package profile

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// Profile is an in-memory representation of profile.proto.
type Profile struct {
	SampleType        []*ValueType
	DefaultSampleType string
	Sample            []*Sample
	Mapping           []*Mapping
	Location          []*Location
	Function          []*Function
	Comments          []string

	DropFrames string
	KeepFrames string

	TimeNanos     int64
	DurationNanos int64
	PeriodType    *ValueType
	Period        int64

	// The following fields are modified during encoding and copying,
	// so are protected by a Mutex.
	encodeMu sync.Mutex

	commentX           []int64
	dropFramesX        int64
	keepFramesX        int64
	stringTable        []string
	defaultSampleTypeX int64
}

// ValueType corresponds to Profile.ValueType
type ValueType struct {
	Type string // cpu, wall, inuse_space, etc
	Unit string // seconds, nanoseconds, bytes, etc

	typeX int64
	unitX int64
}

// Sample corresponds to Profile.Sample
type Sample struct {
	Location []*Location
	Value    []int64
	Label    map[string][]string
	NumLabel map[string][]int64
	NumUnit  map[string][]string

	locationIDX []uint64
	labelX      []label
}

// label corresponds to Profile.Label
type label struct {
	keyX int64
	// Exactly one of the two following values must be set
	strX int64
	numX int64 // Integer value for this label
	// can be set if numX has value
	unitX int64
}

// Mapping corresponds to Profile.Mapping
type Mapping struct {
	ID              uint64
	Start           uint64
	Limit           uint64
	Offset          uint64
	File            string
	BuildID         string
	HasFunctions    bool
	HasFilenames    bool
	HasLineNumbers  bool
	HasInlineFrames bool

	fileX    int64
	buildIDX int64
}

// Location corresponds to Profile.Location
type Location struct {
	ID       uint64
	Mapping  *Mapping
	Address  uint64
	Line     []Line
	IsFolded bool

	mappingIDX uint64
}

// Line corresponds to Profile.Line
type Line struct {
	Function *Function
	Line     int64

	functionIDX uint64
}

// Function corresponds to Profile.Function
type Function struct {
	ID         uint64
	Name       string
	SystemName string
	Filename   string
	StartLine  int64

	nameX       int64
	systemNameX int64
	filenameX   int64
}

// Parse parses a profile and checks for its validity. The input
// may be a gzip-compressed encoded protobuf or one of many legacy
// profile formats which may be unsupported in the future.
func Parse(r io.Reader) (*Profile, error) {
	data, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return ParseData(data)
}

// ParseData parses a profile from a buffer and checks for its
// validity.
func ParseData(data []byte) (*Profile, error) {
	var p *Profile
	var err error
	if len(data) >= 2 && data[0] == 0x1f && data[1] == 0x8b {
		gz, err := gzip.NewReader(bytes.NewBuffer(data))
		if err == nil {
			data, err = ioutil.ReadAll(gz)
		}
		if err != nil {
			return nil, fmt.Errorf("decompressing profile: %v", err)
		}
	}
	if p, err = ParseUncompressed(data); err != nil && err != errNoData && err != errConcatProfile {
		p, err = parseLegacy(data)
	}

	if err != nil {
		return nil, fmt.Errorf("parsing profile: %v", err)
	}

	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("malformed profile: %v", err)
	}
	return p, nil
}

var errUnrecognized = fmt.Errorf("unrecognized profile format")
var errMalformed = fmt.Errorf("malformed profile format")
var errNoData = fmt.Errorf("empty input file")
var errConcatProfile = fmt.Errorf("concatenated profiles detected")

func parseLegacy(data []byte) (*Profile, error) {
	parsers := []func([]byte) (*Profile, error){
		parseCPU,
		parseHeap,
		parseGoCount, // goroutine, threadcreate
		parseThread,
		parseContention,
		parseJavaProfile,
	}

	for _, parser := range parsers {
		p, err := parser(data)
		if err == nil {
			p.addLegacyFrameInfo()
			return p, nil
		}
		if err != errUnrecognized {
			return nil, err
		}
	}
	return nil, errUnrecognized
}

// ParseUncompressed parses an uncompressed protobuf into a profile.
func ParseUncompressed(data []byte) (*Profile, error) {
	if len(data) == 0 {
		return nil, errNoData
	}
	p := &Profile{}
	if err := unmarshal(data, p); err != nil {
		return nil, err
	}

	if err := p.postDecode(); err != nil {
		return nil, err
	}

	return p, nil
}

var libRx = regexp.MustCompile(`([.]so$|[.]so[._][0-9]+)`)

// massageMappings applies heuristic-based changes to the profile
// mappings to account for quirks of some environments.
func (p *Profile) massageMappings() {
	// Merge adjacent regions with matching names, checking that the offsets match
	if len(p.Mapping) > 1 {
		mappings := []*Mapping{p.Mapping[0]}
		for _, m := range p.Mapping[1:] {
			lm := mappings[len(mappings)-1]
			if adjacent(lm, m) {
				lm.Limit = m.Limit
				if m.File != "" {
					lm.File = m.File
				}
				if m.BuildID != "" {
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
		if len(libRx.FindStringSubmatch(file)) > 0 {
			continue
		}
		if file[0] == '[' {
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

// adjacent returns whether two mapping entries represent the same
// mapping that has been split into two. Check that their addresses are adjacent,
// and if the offsets match, if they are available.
func adjacent(m1, m2 *Mapping) bool {
	if m1.File != "" && m2.File != "" {
		if m1.File != m2.File {
			return false
		}
	}
	if m1.BuildID != "" && m2.BuildID != "" {
		if m1.BuildID != m2.BuildID {
			return false
		}
	}
	if m1.Limit != m2.Start {
		return false
	}
	if m1.Offset != 0 && m2.Offset != 0 {
		offset := m1.Offset + (m1.Limit - m1.Start)
		if offset != m2.Offset {
			return false
		}
	}
	return true
}

func (p *Profile) updateLocationMapping(from, to *Mapping) {
	for _, l := range p.Location {
		if l.Mapping == from {
			l.Mapping = to
		}
	}
}

func serialize(p *Profile) []byte {
	p.encodeMu.Lock()
	p.preEncode()
	b := marshal(p)
	p.encodeMu.Unlock()
	return b
}

// Write writes the profile as a gzip-compressed marshaled protobuf.
func (p *Profile) Write(w io.Writer) error {
	zw := gzip.NewWriter(w)
	defer zw.Close()
	_, err := zw.Write(serialize(p))
	return err
}

// WriteUncompressed writes the profile as a marshaled protobuf.
func (p *Profile) WriteUncompressed(w io.Writer) error {
	_, err := w.Write(serialize(p))
	return err
}

// CheckValid tests whether the profile is valid. Checks include, but are
// not limited to:
//   - len(Profile.Sample[n].value) == len(Profile.value_unit)
//   - Sample.id has a corresponding Profile.Location
func (p *Profile) CheckValid() error {
	// Check that sample values are consistent
	sampleLen := len(p.SampleType)
	if sampleLen == 0 && len(p.Sample) != 0 {
		return fmt.Errorf("missing sample type information")
	}
	for _, s := range p.Sample {
		if s == nil {
			return fmt.Errorf("profile has nil sample")
		}
		if len(s.Value) != sampleLen {
			return fmt.Errorf("mismatch: sample has %d values vs. %d types", len(s.Value), len(p.SampleType))
		}
		for _, l := range s.Location {
			if l == nil {
				return fmt.Errorf("sample has nil location")
			}
		}
	}

	// Check that all mappings/locations/functions are in the tables
	// Check that there are no duplicate ids
	mappings := make(map[uint64]*Mapping, len(p.Mapping))
	for _, m := range p.Mapping {
		if m == nil {
			return fmt.Errorf("profile has nil mapping")
		}
		if m.ID == 0 {
			return fmt.Errorf("found mapping with reserved ID=0")
		}
		if mappings[m.ID] != nil {
			return fmt.Errorf("multiple mappings with same id: %d", m.ID)
		}
		mappings[m.ID] = m
	}
	functions := make(map[uint64]*Function, len(p.Function))
	for _, f := range p.Function {
		if f == nil {
			return fmt.Errorf("profile has nil function")
		}
		if f.ID == 0 {
			return fmt.Errorf("found function with reserved ID=0")
		}
		if functions[f.ID] != nil {
			return fmt.Errorf("multiple functions with same id: %d", f.ID)
		}
		functions[f.ID] = f
	}
	locations := make(map[uint64]*Location, len(p.Location))
	for _, l := range p.Location {
		if l == nil {
			return fmt.Errorf("profile has nil location")
		}
		if l.ID == 0 {
			return fmt.Errorf("found location with reserved id=0")
		}
		if locations[l.ID] != nil {
			return fmt.Errorf("multiple locations with same id: %d", l.ID)
		}
		locations[l.ID] = l
		if m := l.Mapping; m != nil {
			if m.ID == 0 || mappings[m.ID] != m {
				return fmt.Errorf("inconsistent mapping %p: %d", m, m.ID)
			}
		}
		for _, ln := range l.Line {
			if f := ln.Function; f != nil {
				if f.ID == 0 || functions[f.ID] != f {
					return fmt.Errorf("inconsistent function %p: %d", f, f.ID)
				}
			}
		}
	}
	return nil
}

// Aggregate merges the locations in the profile into equivalence
// classes preserving the request attributes. It also updates the
// samples to point to the merged locations.
func (p *Profile) Aggregate(inlineFrame, function, filename, linenumber, address bool) error {
	for _, m := range p.Mapping {
		m.HasInlineFrames = m.HasInlineFrames && inlineFrame
		m.HasFunctions = m.HasFunctions && function
		m.HasFilenames = m.HasFilenames && filename
		m.HasLineNumbers = m.HasLineNumbers && linenumber
	}

	// Aggregate functions
	if !function || !filename {
		for _, f := range p.Function {
			if !function {
				f.Name = ""
				f.SystemName = ""
			}
			if !filename {
				f.Filename = ""
			}
		}
	}

	// Aggregate locations
	if !inlineFrame || !address || !linenumber {
		for _, l := range p.Location {
			if !inlineFrame && len(l.Line) > 1 {
				l.Line = l.Line[len(l.Line)-1:]
			}
			if !linenumber {
				for i := range l.Line {
					l.Line[i].Line = 0
				}
			}
			if !address {
				l.Address = 0
			}
		}
	}

	return p.CheckValid()
}

// NumLabelUnits returns a map of numeric label keys to the units
// associated with those keys and a map of those keys to any units
// that were encountered but not used.
// Unit for a given key is the first encountered unit for that key. If multiple
// units are encountered for values paired with a particular key, then the first
// unit encountered is used and all other units are returned in sorted order
// in map of ignored units.
// If no units are encountered for a particular key, the unit is then inferred
// based on the key.
func (p *Profile) NumLabelUnits() (map[string]string, map[string][]string) {
	numLabelUnits := map[string]string{}
	ignoredUnits := map[string]map[string]bool{}
	encounteredKeys := map[string]bool{}

	// Determine units based on numeric tags for each sample.
	for _, s := range p.Sample {
		for k := range s.NumLabel {
			encounteredKeys[k] = true
			for _, unit := range s.NumUnit[k] {
				if unit == "" {
					continue
				}
				if wantUnit, ok := numLabelUnits[k]; !ok {
					numLabelUnits[k] = unit
				} else if wantUnit != unit {
					if v, ok := ignoredUnits[k]; ok {
						v[unit] = true
					} else {
						ignoredUnits[k] = map[string]bool{unit: true}
					}
				}
			}
		}
	}
	// Infer units for keys without any units associated with
	// numeric tag values.
	for key := range encounteredKeys {
		unit := numLabelUnits[key]
		if unit == "" {
			switch key {
			case "alignment", "request":
				numLabelUnits[key] = "bytes"
			default:
				numLabelUnits[key] = key
			}
		}
	}

	// Copy ignored units into more readable format
	unitsIgnored := make(map[string][]string, len(ignoredUnits))
	for key, values := range ignoredUnits {
		units := make([]string, len(values))
		i := 0
		for unit := range values {
			units[i] = unit
			i++
		}
		sort.Strings(units)
		unitsIgnored[key] = units
	}

	return numLabelUnits, unitsIgnored
}

// String dumps a text representation of a profile. Intended mainly
// for debugging purposes.
func (p *Profile) String() string {
	ss := make([]string, 0, len(p.Comments)+len(p.Sample)+len(p.Mapping)+len(p.Location))
	for _, c := range p.Comments {
		ss = append(ss, "Comment: "+c)
	}
	if pt := p.PeriodType; pt != nil {
		ss = append(ss, fmt.Sprintf("PeriodType: %s %s", pt.Type, pt.Unit))
	}
	ss = append(ss, fmt.Sprintf("Period: %d", p.Period))
	if p.TimeNanos != 0 {
		ss = append(ss, fmt.Sprintf("Time: %v", time.Unix(0, p.TimeNanos)))
	}
	if p.DurationNanos != 0 {
		ss = append(ss, fmt.Sprintf("Duration: %.4v", time.Duration(p.DurationNanos)))
	}

	ss = append(ss, "Samples:")
	var sh1 string
	for _, s := range p.SampleType {
		dflt := ""
		if s.Type == p.DefaultSampleType {
			dflt = "[dflt]"
		}
		sh1 = sh1 + fmt.Sprintf("%s/%s%s ", s.Type, s.Unit, dflt)
	}
	ss = append(ss, strings.TrimSpace(sh1))
	for _, s := range p.Sample {
		ss = append(ss, s.string())
	}

	ss = append(ss, "Locations")
	for _, l := range p.Location {
		ss = append(ss, l.string())
	}

	ss = append(ss, "Mappings")
	for _, m := range p.Mapping {
		ss = append(ss, m.string())
	}

	return strings.Join(ss, "\n") + "\n"
}

// string dumps a text representation of a mapping. Intended mainly
// for debugging purposes.
func (m *Mapping) string() string {
	bits := ""
	if m.HasFunctions {
		bits = bits + "[FN]"
	}
	if m.HasFilenames {
		bits = bits + "[FL]"
	}
	if m.HasLineNumbers {
		bits = bits + "[LN]"
	}
	if m.HasInlineFrames {
		bits = bits + "[IN]"
	}
	return fmt.Sprintf("%d: %#x/%#x/%#x %s %s %s",
		m.ID,
		m.Start, m.Limit, m.Offset,
		m.File,
		m.BuildID,
		bits)
}

// string dumps a text representation of a location. Intended mainly
// for debugging purposes.
func (l *Location) string() string {
	ss := []string{}
	locStr := fmt.Sprintf("%6d: %#x ", l.ID, l.Address)
	if m := l.Mapping; m != nil {
		locStr = locStr + fmt.Sprintf("M=%d ", m.ID)
	}
	if l.IsFolded {
		locStr = locStr + "[F] "
	}
	if len(l.Line) == 0 {
		ss = append(ss, locStr)
	}
	for li := range l.Line {
		lnStr := "??"
		if fn := l.Line[li].Function; fn != nil {
			lnStr = fmt.Sprintf("%s %s:%d s=%d",
				fn.Name,
				fn.Filename,
				l.Line[li].Line,
				fn.StartLine)
			if fn.Name != fn.SystemName {
				lnStr = lnStr + "(" + fn.SystemName + ")"
			}
		}
		ss = append(ss, locStr+lnStr)
		// Do not print location details past the first line
		locStr = "             "
	}
	return strings.Join(ss, "\n")
}

// string dumps a text representation of a sample. Intended mainly
// for debugging purposes.
func (s *Sample) string() string {
	ss := []string{}
	var sv string
	for _, v := range s.Value {
		sv = fmt.Sprintf("%s %10d", sv, v)
	}
	sv = sv + ": "
	for _, l := range s.Location {
		sv = sv + fmt.Sprintf("%d ", l.ID)
	}
	ss = append(ss, sv)
	const labelHeader = "                "
	if len(s.Label) > 0 {
		ss = append(ss, labelHeader+labelsToString(s.Label))
	}
	if len(s.NumLabel) > 0 {
		ss = append(ss, labelHeader+numLabelsToString(s.NumLabel, s.NumUnit))
	}
	return strings.Join(ss, "\n")
}

// labelsToString returns a string representation of a
// map representing labels.
func labelsToString(labels map[string][]string) string {
	ls := []string{}
	for k, v := range labels {
		ls = append(ls, fmt.Sprintf("%s:%v", k, v))
	}
	sort.Strings(ls)
	return strings.Join(ls, " ")
}

// numLablesToString returns a string representation of a map
// representing numeric labels.
func numLabelsToString(numLabels map[string][]int64, numUnits map[string][]string) string {
	ls := []string{}
	for k, v := range numLabels {
		units := numUnits[k]
		var labelString string
		if len(units) == len(v) {
			values := make([]string, len(v))
			for i, vv := range v {
				values[i] = fmt.Sprintf("%d %s", vv, units[i])
			}
			labelString = fmt.Sprintf("%s:%v", k, values)
		} else {
			labelString = fmt.Sprintf("%s:%v", k, v)
		}
		ls = append(ls, labelString)
	}
	sort.Strings(ls)
	return strings.Join(ls, " ")
}

// SetLabel sets the specified key to the specified value for all samples in the
// profile.
func (p *Profile) SetLabel(key string, value []string) {
	for _, sample := range p.Sample {
		if sample.Label == nil {
			sample.Label = map[string][]string{key: value}
		} else {
			sample.Label[key] = value
		}
	}
}

// RemoveLabel removes all labels associated with the specified key for all
// samples in the profile.
func (p *Profile) RemoveLabel(key string) {
	for _, sample := range p.Sample {
		delete(sample.Label, key)
	}
}

// HasLabel returns true if a sample has a label with indicated key and value.
func (s *Sample) HasLabel(key, value string) bool {
	for _, v := range s.Label[key] {
		if v == value {
			return true
		}
	}
	return false
}

// DiffBaseSample returns true if a sample belongs to the diff base and false
// otherwise.
func (s *Sample) DiffBaseSample() bool {
	return s.HasLabel("pprof::base", "true")
}

// Scale multiplies all sample values in a profile by a constant.
func (p *Profile) Scale(ratio float64) {
	if ratio == 1 {
		return
	}
	ratios := make([]float64, len(p.SampleType))
	for i := range p.SampleType {
		ratios[i] = ratio
	}
	p.ScaleN(ratios)
}

// ScaleN multiplies each sample values in a sample by a different amount.
func (p *Profile) ScaleN(ratios []float64) error {
	if len(p.SampleType) != len(ratios) {
		return fmt.Errorf("mismatched scale ratios, got %d, want %d", len(ratios), len(p.SampleType))
	}
	allOnes := true
	for _, r := range ratios {
		if r != 1 {
			allOnes = false
			break
		}
	}
	if allOnes {
		return nil
	}
	for _, s := range p.Sample {
		for i, v := range s.Value {
			if ratios[i] != 1 {
				s.Value[i] = int64(float64(v) * ratios[i])
			}
		}
	}
	return nil
}

// HasFunctions determines if all locations in this profile have
// symbolized function information.
func (p *Profile) HasFunctions() bool {
	for _, l := range p.Location {
		if l.Mapping != nil && !l.Mapping.HasFunctions {
			return false
		}
	}
	return true
}

// HasFileLines determines if all locations in this profile have
// symbolized file and line number information.
func (p *Profile) HasFileLines() bool {
	for _, l := range p.Location {
		if l.Mapping != nil && (!l.Mapping.HasFilenames || !l.Mapping.HasLineNumbers) {
			return false
		}
	}
	return true
}

// Unsymbolizable returns true if a mapping points to a binary for which
// locations can't be symbolized in principle, at least now. Examples are
// "[vdso]", [vsyscall]" and some others, see the code.
func (m *Mapping) Unsymbolizable() bool {
	name := filepath.Base(m.File)
	return strings.HasPrefix(name, "[") || strings.HasPrefix(name, "linux-vdso") || strings.HasPrefix(m.File, "/dev/dri/")
}

// Copy makes a fully independent copy of a profile.
func (p *Profile) Copy() *Profile {
	pp := &Profile{}
	if err := unmarshal(serialize(p), pp); err != nil {
		panic(err)
	}
	if err := pp.postDecode(); err != nil {
		panic(err)
	}

	return pp
}
