// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package profile provides a representation of
// github.com/google/pprof/proto/profile.proto and
// methods to encode/decode/merge profiles in this format.
package profile

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"regexp"
	"strings"
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
	labelX      []Label
}

// Label corresponds to Profile.Label
type Label struct {
	keyX int64
	// Exactly one of the two following values must be set
	strX int64
	numX int64 // Integer value for this label
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
	orig, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	var p *Profile
	if len(orig) >= 2 && orig[0] == 0x1f && orig[1] == 0x8b {
		gz, err := gzip.NewReader(bytes.NewBuffer(orig))
		if err != nil {
			return nil, fmt.Errorf("decompressing profile: %v", err)
		}
		data, err := ioutil.ReadAll(gz)
		if err != nil {
			return nil, fmt.Errorf("decompressing profile: %v", err)
		}
		orig = data
	}
	if p, err = parseUncompressed(orig); err != nil {
		if p, err = parseLegacy(orig); err != nil {
			return nil, fmt.Errorf("parsing profile: %v", err)
		}
	}

	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("malformed profile: %v", err)
	}
	return p, nil
}

var errUnrecognized = fmt.Errorf("unrecognized profile format")
var errMalformed = fmt.Errorf("malformed profile format")

func parseLegacy(data []byte) (*Profile, error) {
	parsers := []func([]byte) (*Profile, error){
		parseCPU,
		parseHeap,
		parseGoCount, // goroutine, threadcreate
		parseThread,
		parseContention,
	}

	for _, parser := range parsers {
		p, err := parser(data)
		if err == nil {
			p.setMain()
			p.addLegacyFrameInfo()
			return p, nil
		}
		if err != errUnrecognized {
			return nil, err
		}
	}
	return nil, errUnrecognized
}

func parseUncompressed(data []byte) (*Profile, error) {
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

// setMain scans Mapping entries and guesses which entry is main
// because legacy profiles don't obey the convention of putting main
// first.
func (p *Profile) setMain() {
	for i := 0; i < len(p.Mapping); i++ {
		file := strings.TrimSpace(strings.ReplaceAll(p.Mapping[i].File, "(deleted)", ""))
		if len(file) == 0 {
			continue
		}
		if len(libRx.FindStringSubmatch(file)) > 0 {
			continue
		}
		if strings.HasPrefix(file, "[") {
			continue
		}
		// Swap what we guess is main to position 0.
		p.Mapping[i], p.Mapping[0] = p.Mapping[0], p.Mapping[i]
		break
	}
}

// Write writes the profile as a gzip-compressed marshaled protobuf.
func (p *Profile) Write(w io.Writer) error {
	p.preEncode()
	b := marshal(p)
	zw := gzip.NewWriter(w)
	defer zw.Close()
	_, err := zw.Write(b)
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
		if len(s.Value) != sampleLen {
			return fmt.Errorf("mismatch: sample has: %d values vs. %d types", len(s.Value), len(p.SampleType))
		}
	}

	// Check that all mappings/locations/functions are in the tables
	// Check that there are no duplicate ids
	mappings := make(map[uint64]*Mapping, len(p.Mapping))
	for _, m := range p.Mapping {
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

// Print dumps a text representation of a profile. Intended mainly
// for debugging purposes.
func (p *Profile) String() string {

	ss := make([]string, 0, len(p.Sample)+len(p.Mapping)+len(p.Location))
	if pt := p.PeriodType; pt != nil {
		ss = append(ss, fmt.Sprintf("PeriodType: %s %s", pt.Type, pt.Unit))
	}
	ss = append(ss, fmt.Sprintf("Period: %d", p.Period))
	if p.TimeNanos != 0 {
		ss = append(ss, fmt.Sprintf("Time: %v", time.Unix(0, p.TimeNanos)))
	}
	if p.DurationNanos != 0 {
		ss = append(ss, fmt.Sprintf("Duration: %v", time.Duration(p.DurationNanos)))
	}

	ss = append(ss, "Samples:")
	var sh1 string
	for _, s := range p.SampleType {
		sh1 = sh1 + fmt.Sprintf("%s/%s ", s.Type, s.Unit)
	}
	ss = append(ss, strings.TrimSpace(sh1))
	for _, s := range p.Sample {
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
			ls := labelHeader
			for k, v := range s.Label {
				ls = ls + fmt.Sprintf("%s:%v ", k, v)
			}
			ss = append(ss, ls)
		}
		if len(s.NumLabel) > 0 {
			ls := labelHeader
			for k, v := range s.NumLabel {
				ls = ls + fmt.Sprintf("%s:%v ", k, v)
			}
			ss = append(ss, ls)
		}
	}

	ss = append(ss, "Locations")
	for _, l := range p.Location {
		locStr := fmt.Sprintf("%6d: %#x ", l.ID, l.Address)
		if m := l.Mapping; m != nil {
			locStr = locStr + fmt.Sprintf("M=%d ", m.ID)
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
	}

	ss = append(ss, "Mappings")
	for _, m := range p.Mapping {
		bits := ""
		if m.HasFunctions {
			bits += "[FN]"
		}
		if m.HasFilenames {
			bits += "[FL]"
		}
		if m.HasLineNumbers {
			bits += "[LN]"
		}
		if m.HasInlineFrames {
			bits += "[IN]"
		}
		ss = append(ss, fmt.Sprintf("%d: %#x/%#x/%#x %s %s %s",
			m.ID,
			m.Start, m.Limit, m.Offset,
			m.File,
			m.BuildID,
			bits))
	}

	return strings.Join(ss, "\n") + "\n"
}

// Merge adds profile p adjusted by ratio r into profile p. Profiles
// must be compatible (same Type and SampleType).
// TODO(rsilvera): consider normalizing the profiles based on the
// total samples collected.
func (p *Profile) Merge(pb *Profile, r float64) error {
	if err := p.Compatible(pb); err != nil {
		return err
	}

	pb = pb.Copy()

	// Keep the largest of the two periods.
	if pb.Period > p.Period {
		p.Period = pb.Period
	}

	p.DurationNanos += pb.DurationNanos

	p.Mapping = append(p.Mapping, pb.Mapping...)
	for i, m := range p.Mapping {
		m.ID = uint64(i + 1)
	}
	p.Location = append(p.Location, pb.Location...)
	for i, l := range p.Location {
		l.ID = uint64(i + 1)
	}
	p.Function = append(p.Function, pb.Function...)
	for i, f := range p.Function {
		f.ID = uint64(i + 1)
	}

	if r != 1.0 {
		for _, s := range pb.Sample {
			for i, v := range s.Value {
				s.Value[i] = int64((float64(v) * r))
			}
		}
	}
	p.Sample = append(p.Sample, pb.Sample...)
	return p.CheckValid()
}

// Compatible determines if two profiles can be compared/merged.
// returns nil if the profiles are compatible; otherwise an error with
// details on the incompatibility.
func (p *Profile) Compatible(pb *Profile) error {
	if !compatibleValueTypes(p.PeriodType, pb.PeriodType) {
		return fmt.Errorf("incompatible period types %v and %v", p.PeriodType, pb.PeriodType)
	}

	if len(p.SampleType) != len(pb.SampleType) {
		return fmt.Errorf("incompatible sample types %v and %v", p.SampleType, pb.SampleType)
	}

	for i := range p.SampleType {
		if !compatibleValueTypes(p.SampleType[i], pb.SampleType[i]) {
			return fmt.Errorf("incompatible sample types %v and %v", p.SampleType, pb.SampleType)
		}
	}

	return nil
}

// HasFunctions determines if all locations in this profile have
// symbolized function information.
func (p *Profile) HasFunctions() bool {
	for _, l := range p.Location {
		if l.Mapping == nil || !l.Mapping.HasFunctions {
			return false
		}
	}
	return true
}

// HasFileLines determines if all locations in this profile have
// symbolized file and line number information.
func (p *Profile) HasFileLines() bool {
	for _, l := range p.Location {
		if l.Mapping == nil || (!l.Mapping.HasFilenames || !l.Mapping.HasLineNumbers) {
			return false
		}
	}
	return true
}

func compatibleValueTypes(v1, v2 *ValueType) bool {
	if v1 == nil || v2 == nil {
		return true // No grounds to disqualify.
	}
	return v1.Type == v2.Type && v1.Unit == v2.Unit
}

// Copy makes a fully independent copy of a profile.
func (p *Profile) Copy() *Profile {
	p.preEncode()
	b := marshal(p)

	pp := &Profile{}
	if err := unmarshal(b, pp); err != nil {
		panic(err)
	}
	if err := pp.postDecode(); err != nil {
		panic(err)
	}

	return pp
}

// Demangler maps symbol names to a human-readable form. This may
// include C++ demangling and additional simplification. Names that
// are not demangled may be missing from the resulting map.
type Demangler func(name []string) (map[string]string, error)

// Demangle attempts to demangle and optionally simplify any function
// names referenced in the profile. It works on a best-effort basis:
// it will silently preserve the original names in case of any errors.
func (p *Profile) Demangle(d Demangler) error {
	// Collect names to demangle.
	var names []string
	for _, fn := range p.Function {
		names = append(names, fn.SystemName)
	}

	// Update profile with demangled names.
	demangled, err := d(names)
	if err != nil {
		return err
	}
	for _, fn := range p.Function {
		if dd, ok := demangled[fn.SystemName]; ok {
			fn.Name = dd
		}
	}
	return nil
}

// Empty reports whether the profile contains no samples.
func (p *Profile) Empty() bool {
	return len(p.Sample) == 0
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
