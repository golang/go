// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains a decoder to test proto profiles

package pprof_test

import (
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"time"
)

type buffer struct {
	field int
	typ   int
	u64   uint64
	data  []byte
	tmp   [16]byte
}

type decoder func(*buffer, message) error

type message interface {
	decoder() []decoder
}

func unmarshal(data []byte, m message) (err error) {
	b := buffer{data: data, typ: 2}
	return decodeMessage(&b, m)
}

func le64(p []byte) uint64 {
	return uint64(p[0]) | uint64(p[1])<<8 | uint64(p[2])<<16 | uint64(p[3])<<24 | uint64(p[4])<<32 | uint64(p[5])<<40 | uint64(p[6])<<48 | uint64(p[7])<<56
}

func le32(p []byte) uint32 {
	return uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16 | uint32(p[3])<<24
}

func decodeVarint(data []byte) (uint64, []byte, error) {
	var i int
	var u uint64
	for i = 0; ; i++ {
		if i >= 10 || i >= len(data) {
			return 0, nil, errors.New("bad varint")
		}
		u |= uint64(data[i]&0x7F) << uint(7*i)
		if data[i]&0x80 == 0 {
			return u, data[i+1:], nil
		}
	}
}

func decodeField(b *buffer, data []byte) ([]byte, error) {
	x, data, err := decodeVarint(data)
	if err != nil {
		return nil, err
	}
	b.field = int(x >> 3)
	b.typ = int(x & 7)
	b.data = nil
	b.u64 = 0
	switch b.typ {
	case 0:
		b.u64, data, err = decodeVarint(data)
		if err != nil {
			return nil, err
		}
	case 1:
		if len(data) < 8 {
			return nil, errors.New("not enough data")
		}
		b.u64 = le64(data[:8])
		data = data[8:]
	case 2:
		var n uint64
		n, data, err = decodeVarint(data)
		if err != nil {
			return nil, err
		}
		if n > uint64(len(data)) {
			return nil, errors.New("too much data")
		}
		b.data = data[:n]
		data = data[n:]
	case 5:
		if len(data) < 4 {
			return nil, errors.New("not enough data")
		}
		b.u64 = uint64(le32(data[:4]))
		data = data[4:]
	default:
		return nil, errors.New("unknown type: " + string(b.typ))
	}

	return data, nil
}

func checkType(b *buffer, typ int) error {
	if b.typ != typ {
		return errors.New("type mismatch")
	}
	return nil
}

func decodeMessage(b *buffer, m message) error {
	if err := checkType(b, 2); err != nil {
		return err
	}
	dec := m.decoder()
	data := b.data
	for len(data) > 0 {
		// pull varint field# + type
		var err error
		data, err = decodeField(b, data)
		if err != nil {
			return err
		}
		if b.field >= len(dec) || dec[b.field] == nil {
			continue
		}
		if err := dec[b.field](b, m); err != nil {
			return err
		}
	}
	return nil
}

func decodeInt64(b *buffer, x *int64) error {
	if err := checkType(b, 0); err != nil {
		return err
	}
	*x = int64(b.u64)
	return nil
}

func decodeInt64s(b *buffer, x *[]int64) error {
	if b.typ == 2 {
		// Packed encoding
		data := b.data
		for len(data) > 0 {
			var u uint64
			var err error

			if u, data, err = decodeVarint(data); err != nil {
				return err
			}
			*x = append(*x, int64(u))
		}
		return nil
	}
	var i int64
	if err := decodeInt64(b, &i); err != nil {
		return err
	}
	*x = append(*x, i)
	return nil
}

func decodeUint64(b *buffer, x *uint64) error {
	if err := checkType(b, 0); err != nil {
		return err
	}
	*x = b.u64
	return nil
}

func decodeUint64s(b *buffer, x *[]uint64) error {
	if b.typ == 2 {
		data := b.data
		// Packed encoding
		for len(data) > 0 {
			var u uint64
			var err error

			if u, data, err = decodeVarint(data); err != nil {
				return err
			}
			*x = append(*x, u)
		}
		return nil
	}
	var u uint64
	if err := decodeUint64(b, &u); err != nil {
		return err
	}
	*x = append(*x, u)
	return nil
}

func decodeString(b *buffer, x *string) error {
	if err := checkType(b, 2); err != nil {
		return err
	}
	*x = string(b.data)
	return nil
}

func decodeStrings(b *buffer, x *[]string) error {
	var s string
	if err := decodeString(b, &s); err != nil {
		return err
	}
	*x = append(*x, s)
	return nil
}

func decodeBool(b *buffer, x *bool) error {
	if err := checkType(b, 0); err != nil {
		return err
	}
	if int64(b.u64) == 0 {
		*x = false
	} else {
		*x = true
	}
	return nil
}

func (p *ProfileTest) decoder() []decoder {
	return profileDecoder
}

var profileDecoder = []decoder{
	nil, // 0
	// repeated ValueType sample_type = 1
	func(b *buffer, m message) error {
		x := new(ValueType)
		pp := m.(*ProfileTest)
		pp.SampleType = append(pp.SampleType, x)
		return decodeMessage(b, x)
	},
	// repeated Sample sample = 2
	func(b *buffer, m message) error {
		x := new(Sample)
		pp := m.(*ProfileTest)
		pp.Sample = append(pp.Sample, x)
		return decodeMessage(b, x)
	},
	// repeated Mapping mapping = 3
	func(b *buffer, m message) error {
		x := new(Mapping)
		pp := m.(*ProfileTest)
		pp.Mapping = append(pp.Mapping, x)
		return decodeMessage(b, x)
	},
	// repeated Location location = 4
	func(b *buffer, m message) error {
		x := new(Location)
		pp := m.(*ProfileTest)
		pp.Location = append(pp.Location, x)
		return decodeMessage(b, x)
	},
	// repeated Function function = 5
	func(b *buffer, m message) error {
		x := new(Function)
		pp := m.(*ProfileTest)
		pp.Function = append(pp.Function, x)
		return decodeMessage(b, x)
	},
	// repeated string string_table = 6
	func(b *buffer, m message) error {
		err := decodeStrings(b, &m.(*ProfileTest).stringTable)
		if err != nil {
			return err
		}
		if *&m.(*ProfileTest).stringTable[0] != "" {
			return errors.New("string_table[0] must be ''")
		}
		return nil
	},
	// repeated int64 drop_frames = 7
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*ProfileTest).dropFramesX) },
	// repeated int64 keep_frames = 8
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*ProfileTest).keepFramesX) },
	// repeated int64 time_nanos = 9
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*ProfileTest).TimeNanos) },
	// repeated int64 duration_nanos = 10
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*ProfileTest).DurationNanos) },
	// optional string period_type = 11
	func(b *buffer, m message) error {
		x := new(ValueType)
		pp := m.(*ProfileTest)
		pp.PeriodType = x
		return decodeMessage(b, x)
	},
	// repeated int64 period = 12
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*ProfileTest).Period) },
}

// postDecode takes the unexported fields populated by decode (with
// suffix X) and populates the corresponding exported fields.
// The unexported fields are cleared up to facilitate testing.
func (p *ProfileTest) postDecode() error {
	var err error

	mappings := make(map[uint64]*Mapping)
	for _, m := range p.Mapping {
		m.File, err = getString(p.stringTable, &m.fileX, err)
		m.BuildID, err = getString(p.stringTable, &m.buildIDX, err)
		mappings[m.ID] = m
	}

	functions := make(map[uint64]*Function)
	for _, f := range p.Function {
		f.Name, err = getString(p.stringTable, &f.nameX, err)
		f.SystemName, err = getString(p.stringTable, &f.systemNameX, err)
		f.Filename, err = getString(p.stringTable, &f.filenameX, err)
		functions[f.ID] = f
	}

	locations := make(map[uint64]*Location)
	for _, l := range p.Location {
		l.Mapping = mappings[l.mappingIDX]
		l.mappingIDX = 0
		for i, ln := range l.Line {
			if id := ln.functionIDX; id != 0 {
				l.Line[i].Function = functions[id]
				if l.Line[i].Function == nil {
					return fmt.Errorf("Function ID %d not found", id)
				}
				l.Line[i].functionIDX = 0
			}
		}
		locations[l.ID] = l
	}

	for _, st := range p.SampleType {
		st.Type, err = getString(p.stringTable, &st.typeX, err)
		st.Unit, err = getString(p.stringTable, &st.unitX, err)
	}

	for _, s := range p.Sample {
		labels := make(map[string][]string)
		numLabels := make(map[string][]int64)
		for _, l := range s.labelX {
			var key, value string
			key, err = getString(p.stringTable, &l.keyX, err)
			if l.strX != 0 {
				value, err = getString(p.stringTable, &l.strX, err)
				labels[key] = append(labels[key], value)
			} else {
				numLabels[key] = append(numLabels[key], l.numX)
			}
		}
		if len(labels) > 0 {
			s.Label = labels
		}
		if len(numLabels) > 0 {
			s.NumLabel = numLabels
		}
		s.Location = nil
		for _, lid := range s.locationIDX {
			s.Location = append(s.Location, locations[lid])
		}
		s.locationIDX = nil
	}

	p.DropFrames, err = getString(p.stringTable, &p.dropFramesX, err)
	p.KeepFrames, err = getString(p.stringTable, &p.keepFramesX, err)

	if pt := p.PeriodType; pt == nil {
		p.PeriodType = &ValueType{}
	}

	if pt := p.PeriodType; pt != nil {
		pt.Type, err = getString(p.stringTable, &pt.typeX, err)
		pt.Unit, err = getString(p.stringTable, &pt.unitX, err)
	}
	p.stringTable = nil
	return nil
}

func (p *ValueType) decoder() []decoder {
	return valueTypeDecoder
}

var valueTypeDecoder = []decoder{
	nil, // 0
	// optional int64 type = 1
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*ValueType).typeX) },
	// optional int64 unit = 2
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*ValueType).unitX) },
}

func (p *Sample) decoder() []decoder {
	return sampleDecoder
}

var sampleDecoder = []decoder{
	nil, // 0
	// repeated uint64 location = 1
	func(b *buffer, m message) error { return decodeUint64s(b, &m.(*Sample).locationIDX) },
	// repeated int64 value = 2
	func(b *buffer, m message) error { return decodeInt64s(b, &m.(*Sample).Value) },
	// repeated Label label = 3
	func(b *buffer, m message) error {
		s := m.(*Sample)
		n := len(s.labelX)
		s.labelX = append(s.labelX, Label{})
		return decodeMessage(b, &s.labelX[n])
	},
}

func (p Label) decoder() []decoder {
	return labelDecoder
}

var labelDecoder = []decoder{
	nil, // 0
	// optional int64 key = 1
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Label).keyX) },
	// optional int64 str = 2
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Label).strX) },
	// optional int64 num = 3
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Label).numX) },
}

func (p *Mapping) decoder() []decoder {
	return mappingDecoder
}

var mappingDecoder = []decoder{
	nil, // 0
	func(b *buffer, m message) error { return decodeUint64(b, &m.(*Mapping).ID) },            // optional uint64 id = 1
	func(b *buffer, m message) error { return decodeUint64(b, &m.(*Mapping).Start) },         // optional uint64 memory_offset = 2
	func(b *buffer, m message) error { return decodeUint64(b, &m.(*Mapping).Limit) },         // optional uint64 memory_limit = 3
	func(b *buffer, m message) error { return decodeUint64(b, &m.(*Mapping).Offset) },        // optional uint64 file_offset = 4
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Mapping).fileX) },          // optional int64 filename = 5
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Mapping).buildIDX) },       // optional int64 build_id = 6
	func(b *buffer, m message) error { return decodeBool(b, &m.(*Mapping).HasFunctions) },    // optional bool has_functions = 7
	func(b *buffer, m message) error { return decodeBool(b, &m.(*Mapping).HasFilenames) },    // optional bool has_filenames = 8
	func(b *buffer, m message) error { return decodeBool(b, &m.(*Mapping).HasLineNumbers) },  // optional bool has_line_numbers = 9
	func(b *buffer, m message) error { return decodeBool(b, &m.(*Mapping).HasInlineFrames) }, // optional bool has_inline_frames = 10
}

func (p *Location) decoder() []decoder {
	return locationDecoder
}

var locationDecoder = []decoder{
	nil, // 0
	func(b *buffer, m message) error { return decodeUint64(b, &m.(*Location).ID) },         // optional uint64 id = 1;
	func(b *buffer, m message) error { return decodeUint64(b, &m.(*Location).mappingIDX) }, // optional uint64 mapping_id = 2;
	func(b *buffer, m message) error { return decodeUint64(b, &m.(*Location).Address) },    // optional uint64 address = 3;
	func(b *buffer, m message) error { // repeated Line line = 4
		pp := m.(*Location)
		n := len(pp.Line)
		pp.Line = append(pp.Line, Line{})
		return decodeMessage(b, &pp.Line[n])
	},
}

func (p *Line) decoder() []decoder {
	return lineDecoder
}

var lineDecoder = []decoder{
	nil, // 0
	// optional uint64 function_id = 1
	func(b *buffer, m message) error { return decodeUint64(b, &m.(*Line).functionIDX) },
	// optional int64 line = 2
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Line).Line) },
}

func (p *Function) decoder() []decoder {
	return functionDecoder
}

var functionDecoder = []decoder{
	nil, // 0
	// optional uint64 id = 1
	func(b *buffer, m message) error { return decodeUint64(b, &m.(*Function).ID) },
	// optional int64 function_name = 2
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Function).nameX) },
	// optional int64 function_system_name = 3
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Function).systemNameX) },
	// repeated int64 filename = 4
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Function).filenameX) },
	// optional int64 start_line = 5
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Function).StartLine) },
}

func getString(strings []string, strng *int64, err error) (string, error) {
	if err != nil {
		return "", err
	}
	s := int(*strng)
	if s < 0 || s >= len(strings) {
		return "", fmt.Errorf("malformed profile format")
	}
	*strng = 0
	return strings[s], nil
}

// Profile is an in-memory representation of ProfileTest.proto.
type ProfileTest struct {
	SampleType []*ValueType
	Sample     []*Sample
	Mapping    []*Mapping
	Location   []*Location
	Function   []*Function

	DropFrames string
	KeepFrames string

	TimeNanos     int64
	DurationNanos int64
	PeriodType    *ValueType
	Period        int64

	dropFramesX int64
	keepFramesX int64
	stringTable []string
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
	ID      uint64
	Mapping *Mapping
	Address uint64
	Line    []Line

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
func Parse(r io.Reader) (*ProfileTest, error) {
	orig, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	var p *ProfileTest
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
		return nil, fmt.Errorf("parsing profile: %v", err)
	}

	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("malformed profile: %v", err)
	}
	return p, nil
}

func parseUncompressed(data []byte) (*ProfileTest, error) {
	p := &ProfileTest{}
	if err := unmarshal(data, p); err != nil {
		return nil, err
	}

	if err := p.postDecode(); err != nil {
		return nil, err
	}

	return p, nil
}

// CheckValid tests whether the profile is valid. Checks include, but are
// not limited to:
//   - len(Profile.Sample[n].value) == len(Profile.value_unit)
//   - Sample.id has a corresponding Profile.Location
func (p *ProfileTest) CheckValid() error {
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

// Print dumps a text representation of a profile. Intended mainly
// for debugging purposes.
func (p *ProfileTest) String() string {

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
		ss = append(ss, fmt.Sprintf("%d: %#x/%#x/%#x %s %s %s",
			m.ID,
			m.Start, m.Limit, m.Offset,
			m.File,
			m.BuildID,
			bits))
	}

	return strings.Join(ss, "\n") + "\n"
}
