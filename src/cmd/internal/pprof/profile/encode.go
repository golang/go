// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package profile

import (
	"errors"
	"fmt"
	"sort"
)

func (p *Profile) decoder() []decoder {
	return profileDecoder
}

// preEncode populates the unexported fields to be used by encode
// (with suffix X) from the corresponding exported fields. The
// exported fields are cleared up to facilitate testing.
func (p *Profile) preEncode() {
	strings := make(map[string]int)
	addString(strings, "")

	for _, st := range p.SampleType {
		st.typeX = addString(strings, st.Type)
		st.unitX = addString(strings, st.Unit)
	}

	for _, s := range p.Sample {
		s.labelX = nil
		var keys []string
		for k := range s.Label {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			vs := s.Label[k]
			for _, v := range vs {
				s.labelX = append(s.labelX,
					Label{
						keyX: addString(strings, k),
						strX: addString(strings, v),
					},
				)
			}
		}
		var numKeys []string
		for k := range s.NumLabel {
			numKeys = append(numKeys, k)
		}
		sort.Strings(numKeys)
		for _, k := range numKeys {
			vs := s.NumLabel[k]
			for _, v := range vs {
				s.labelX = append(s.labelX,
					Label{
						keyX: addString(strings, k),
						numX: v,
					},
				)
			}
		}
		s.locationIDX = nil
		for _, l := range s.Location {
			s.locationIDX = append(s.locationIDX, l.ID)
		}
	}

	for _, m := range p.Mapping {
		m.fileX = addString(strings, m.File)
		m.buildIDX = addString(strings, m.BuildID)
	}

	for _, l := range p.Location {
		for i, ln := range l.Line {
			if ln.Function != nil {
				l.Line[i].functionIDX = ln.Function.ID
			} else {
				l.Line[i].functionIDX = 0
			}
		}
		if l.Mapping != nil {
			l.mappingIDX = l.Mapping.ID
		} else {
			l.mappingIDX = 0
		}
	}
	for _, f := range p.Function {
		f.nameX = addString(strings, f.Name)
		f.systemNameX = addString(strings, f.SystemName)
		f.filenameX = addString(strings, f.Filename)
	}

	p.dropFramesX = addString(strings, p.DropFrames)
	p.keepFramesX = addString(strings, p.KeepFrames)

	if pt := p.PeriodType; pt != nil {
		pt.typeX = addString(strings, pt.Type)
		pt.unitX = addString(strings, pt.Unit)
	}

	p.stringTable = make([]string, len(strings))
	for s, i := range strings {
		p.stringTable[i] = s
	}
}

func (p *Profile) encode(b *buffer) {
	for _, x := range p.SampleType {
		encodeMessage(b, 1, x)
	}
	for _, x := range p.Sample {
		encodeMessage(b, 2, x)
	}
	for _, x := range p.Mapping {
		encodeMessage(b, 3, x)
	}
	for _, x := range p.Location {
		encodeMessage(b, 4, x)
	}
	for _, x := range p.Function {
		encodeMessage(b, 5, x)
	}
	encodeStrings(b, 6, p.stringTable)
	encodeInt64Opt(b, 7, p.dropFramesX)
	encodeInt64Opt(b, 8, p.keepFramesX)
	encodeInt64Opt(b, 9, p.TimeNanos)
	encodeInt64Opt(b, 10, p.DurationNanos)
	if pt := p.PeriodType; pt != nil && (pt.typeX != 0 || pt.unitX != 0) {
		encodeMessage(b, 11, p.PeriodType)
	}
	encodeInt64Opt(b, 12, p.Period)
}

var profileDecoder = []decoder{
	nil, // 0
	// repeated ValueType sample_type = 1
	func(b *buffer, m message) error {
		x := new(ValueType)
		pp := m.(*Profile)
		pp.SampleType = append(pp.SampleType, x)
		return decodeMessage(b, x)
	},
	// repeated Sample sample = 2
	func(b *buffer, m message) error {
		x := new(Sample)
		pp := m.(*Profile)
		pp.Sample = append(pp.Sample, x)
		return decodeMessage(b, x)
	},
	// repeated Mapping mapping = 3
	func(b *buffer, m message) error {
		x := new(Mapping)
		pp := m.(*Profile)
		pp.Mapping = append(pp.Mapping, x)
		return decodeMessage(b, x)
	},
	// repeated Location location = 4
	func(b *buffer, m message) error {
		x := new(Location)
		pp := m.(*Profile)
		pp.Location = append(pp.Location, x)
		return decodeMessage(b, x)
	},
	// repeated Function function = 5
	func(b *buffer, m message) error {
		x := new(Function)
		pp := m.(*Profile)
		pp.Function = append(pp.Function, x)
		return decodeMessage(b, x)
	},
	// repeated string string_table = 6
	func(b *buffer, m message) error {
		err := decodeStrings(b, &m.(*Profile).stringTable)
		if err != nil {
			return err
		}
		if *&m.(*Profile).stringTable[0] != "" {
			return errors.New("string_table[0] must be ''")
		}
		return nil
	},
	// repeated int64 drop_frames = 7
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).dropFramesX) },
	// repeated int64 keep_frames = 8
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).keepFramesX) },
	// repeated int64 time_nanos = 9
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).TimeNanos) },
	// repeated int64 duration_nanos = 10
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).DurationNanos) },
	// optional string period_type = 11
	func(b *buffer, m message) error {
		x := new(ValueType)
		pp := m.(*Profile)
		pp.PeriodType = x
		return decodeMessage(b, x)
	},
	// repeated int64 period = 12
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).Period) },
}

// postDecode takes the unexported fields populated by decode (with
// suffix X) and populates the corresponding exported fields.
// The unexported fields are cleared up to facilitate testing.
func (p *Profile) postDecode() error {
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

func (p *ValueType) encode(b *buffer) {
	encodeInt64Opt(b, 1, p.typeX)
	encodeInt64Opt(b, 2, p.unitX)
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

func (p *Sample) encode(b *buffer) {
	encodeUint64s(b, 1, p.locationIDX)
	for _, x := range p.Value {
		encodeInt64(b, 2, x)
	}
	for _, x := range p.labelX {
		encodeMessage(b, 3, x)
	}
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

func (p Label) encode(b *buffer) {
	encodeInt64Opt(b, 1, p.keyX)
	encodeInt64Opt(b, 2, p.strX)
	encodeInt64Opt(b, 3, p.numX)
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

func (p *Mapping) encode(b *buffer) {
	encodeUint64Opt(b, 1, p.ID)
	encodeUint64Opt(b, 2, p.Start)
	encodeUint64Opt(b, 3, p.Limit)
	encodeUint64Opt(b, 4, p.Offset)
	encodeInt64Opt(b, 5, p.fileX)
	encodeInt64Opt(b, 6, p.buildIDX)
	encodeBoolOpt(b, 7, p.HasFunctions)
	encodeBoolOpt(b, 8, p.HasFilenames)
	encodeBoolOpt(b, 9, p.HasLineNumbers)
	encodeBoolOpt(b, 10, p.HasInlineFrames)
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

func (p *Location) encode(b *buffer) {
	encodeUint64Opt(b, 1, p.ID)
	encodeUint64Opt(b, 2, p.mappingIDX)
	encodeUint64Opt(b, 3, p.Address)
	for i := range p.Line {
		encodeMessage(b, 4, &p.Line[i])
	}
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

func (p *Line) encode(b *buffer) {
	encodeUint64Opt(b, 1, p.functionIDX)
	encodeInt64Opt(b, 2, p.Line)
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

func (p *Function) encode(b *buffer) {
	encodeUint64Opt(b, 1, p.ID)
	encodeInt64Opt(b, 2, p.nameX)
	encodeInt64Opt(b, 3, p.systemNameX)
	encodeInt64Opt(b, 4, p.filenameX)
	encodeInt64Opt(b, 5, p.StartLine)
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

func addString(strings map[string]int, s string) int64 {
	i, ok := strings[s]
	if !ok {
		i = len(strings)
		strings[s] = i
	}
	return int64(i)
}

func getString(strings []string, strng *int64, err error) (string, error) {
	if err != nil {
		return "", err
	}
	s := int(*strng)
	if s < 0 || s >= len(strings) {
		return "", errMalformed
	}
	*strng = 0
	return strings[s], nil
}
