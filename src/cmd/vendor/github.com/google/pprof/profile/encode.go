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
	"errors"
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
					label{
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
			keyX := addString(strings, k)
			vs := s.NumLabel[k]
			units := s.NumUnit[k]
			for i, v := range vs {
				var unitX int64
				if len(units) != 0 {
					unitX = addString(strings, units[i])
				}
				s.labelX = append(s.labelX,
					label{
						keyX:  keyX,
						numX:  v,
						unitX: unitX,
					},
				)
			}
		}
		s.locationIDX = make([]uint64, len(s.Location))
		for i, loc := range s.Location {
			s.locationIDX[i] = loc.ID
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

	p.commentX = nil
	for _, c := range p.Comments {
		p.commentX = append(p.commentX, addString(strings, c))
	}

	p.defaultSampleTypeX = addString(strings, p.DefaultSampleType)

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
	encodeInt64s(b, 13, p.commentX)
	encodeInt64(b, 14, p.defaultSampleTypeX)
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
		x.Line = make([]Line, 0, 8) // Pre-allocate Line buffer
		pp := m.(*Profile)
		pp.Location = append(pp.Location, x)
		err := decodeMessage(b, x)
		var tmp []Line
		x.Line = append(tmp, x.Line...) // Shrink to allocated size
		return err
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
		if m.(*Profile).stringTable[0] != "" {
			return errors.New("string_table[0] must be ''")
		}
		return nil
	},
	// int64 drop_frames = 7
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).dropFramesX) },
	// int64 keep_frames = 8
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).keepFramesX) },
	// int64 time_nanos = 9
	func(b *buffer, m message) error {
		if m.(*Profile).TimeNanos != 0 {
			return errConcatProfile
		}
		return decodeInt64(b, &m.(*Profile).TimeNanos)
	},
	// int64 duration_nanos = 10
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).DurationNanos) },
	// ValueType period_type = 11
	func(b *buffer, m message) error {
		x := new(ValueType)
		pp := m.(*Profile)
		pp.PeriodType = x
		return decodeMessage(b, x)
	},
	// int64 period = 12
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).Period) },
	// repeated int64 comment = 13
	func(b *buffer, m message) error { return decodeInt64s(b, &m.(*Profile).commentX) },
	// int64 defaultSampleType = 14
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*Profile).defaultSampleTypeX) },
}

// postDecode takes the unexported fields populated by decode (with
// suffix X) and populates the corresponding exported fields.
// The unexported fields are cleared up to facilitate testing.
func (p *Profile) postDecode() error {
	var err error
	mappings := make(map[uint64]*Mapping, len(p.Mapping))
	mappingIds := make([]*Mapping, len(p.Mapping)+1)
	for _, m := range p.Mapping {
		m.File, err = getString(p.stringTable, &m.fileX, err)
		m.BuildID, err = getString(p.stringTable, &m.buildIDX, err)
		if m.ID < uint64(len(mappingIds)) {
			mappingIds[m.ID] = m
		} else {
			mappings[m.ID] = m
		}
	}

	functions := make(map[uint64]*Function, len(p.Function))
	functionIds := make([]*Function, len(p.Function)+1)
	for _, f := range p.Function {
		f.Name, err = getString(p.stringTable, &f.nameX, err)
		f.SystemName, err = getString(p.stringTable, &f.systemNameX, err)
		f.Filename, err = getString(p.stringTable, &f.filenameX, err)
		if f.ID < uint64(len(functionIds)) {
			functionIds[f.ID] = f
		} else {
			functions[f.ID] = f
		}
	}

	locations := make(map[uint64]*Location, len(p.Location))
	locationIds := make([]*Location, len(p.Location)+1)
	for _, l := range p.Location {
		if id := l.mappingIDX; id < uint64(len(mappingIds)) {
			l.Mapping = mappingIds[id]
		} else {
			l.Mapping = mappings[id]
		}
		l.mappingIDX = 0
		for i, ln := range l.Line {
			if id := ln.functionIDX; id != 0 {
				l.Line[i].functionIDX = 0
				if id < uint64(len(functionIds)) {
					l.Line[i].Function = functionIds[id]
				} else {
					l.Line[i].Function = functions[id]
				}
			}
		}
		if l.ID < uint64(len(locationIds)) {
			locationIds[l.ID] = l
		} else {
			locations[l.ID] = l
		}
	}

	for _, st := range p.SampleType {
		st.Type, err = getString(p.stringTable, &st.typeX, err)
		st.Unit, err = getString(p.stringTable, &st.unitX, err)
	}

	for _, s := range p.Sample {
		labels := make(map[string][]string, len(s.labelX))
		numLabels := make(map[string][]int64, len(s.labelX))
		numUnits := make(map[string][]string, len(s.labelX))
		for _, l := range s.labelX {
			var key, value string
			key, err = getString(p.stringTable, &l.keyX, err)
			if l.strX != 0 {
				value, err = getString(p.stringTable, &l.strX, err)
				labels[key] = append(labels[key], value)
			} else if l.numX != 0 || l.unitX != 0 {
				numValues := numLabels[key]
				units := numUnits[key]
				if l.unitX != 0 {
					var unit string
					unit, err = getString(p.stringTable, &l.unitX, err)
					units = padStringArray(units, len(numValues))
					numUnits[key] = append(units, unit)
				}
				numLabels[key] = append(numLabels[key], l.numX)
			}
		}
		if len(labels) > 0 {
			s.Label = labels
		}
		if len(numLabels) > 0 {
			s.NumLabel = numLabels
			for key, units := range numUnits {
				if len(units) > 0 {
					numUnits[key] = padStringArray(units, len(numLabels[key]))
				}
			}
			s.NumUnit = numUnits
		}
		s.Location = make([]*Location, len(s.locationIDX))
		for i, lid := range s.locationIDX {
			if lid < uint64(len(locationIds)) {
				s.Location[i] = locationIds[lid]
			} else {
				s.Location[i] = locations[lid]
			}
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

	for _, i := range p.commentX {
		var c string
		c, err = getString(p.stringTable, &i, err)
		p.Comments = append(p.Comments, c)
	}

	p.commentX = nil
	p.DefaultSampleType, err = getString(p.stringTable, &p.defaultSampleTypeX, err)
	p.stringTable = nil
	return err
}

// padStringArray pads arr with enough empty strings to make arr
// length l when arr's length is less than l.
func padStringArray(arr []string, l int) []string {
	if l <= len(arr) {
		return arr
	}
	return append(arr, make([]string, l-len(arr))...)
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
	encodeInt64s(b, 2, p.Value)
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
		s.labelX = append(s.labelX, label{})
		return decodeMessage(b, &s.labelX[n])
	},
}

func (p label) decoder() []decoder {
	return labelDecoder
}

func (p label) encode(b *buffer) {
	encodeInt64Opt(b, 1, p.keyX)
	encodeInt64Opt(b, 2, p.strX)
	encodeInt64Opt(b, 3, p.numX)
	encodeInt64Opt(b, 4, p.unitX)
}

var labelDecoder = []decoder{
	nil, // 0
	// optional int64 key = 1
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*label).keyX) },
	// optional int64 str = 2
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*label).strX) },
	// optional int64 num = 3
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*label).numX) },
	// optional int64 num = 4
	func(b *buffer, m message) error { return decodeInt64(b, &m.(*label).unitX) },
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
	encodeBoolOpt(b, 5, p.IsFolded)
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
	func(b *buffer, m message) error { return decodeBool(b, &m.(*Location).IsFolded) }, // optional bool is_folded = 5;
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
