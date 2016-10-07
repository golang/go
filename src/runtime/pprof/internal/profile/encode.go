// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package profile

import (
	"sort"
)

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
	encodeInt64Opt(b, 9, p.TimeNanos)
	encodeInt64Opt(b, 10, p.DurationNanos)
	if pt := p.PeriodType; pt != nil && (pt.typeX != 0 || pt.unitX != 0) {
		encodeMessage(b, 11, p.PeriodType)
	}
	encodeInt64Opt(b, 12, p.Period)
}

func (p *ValueType) encode(b *buffer) {
	encodeInt64Opt(b, 1, p.typeX)
	encodeInt64Opt(b, 2, p.unitX)
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

func (p Label) encode(b *buffer) {
	encodeInt64Opt(b, 1, p.keyX)
	encodeInt64Opt(b, 2, p.strX)
	encodeInt64Opt(b, 3, p.numX)
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

func (p *Location) encode(b *buffer) {
	encodeUint64Opt(b, 1, p.ID)
	encodeUint64Opt(b, 2, p.mappingIDX)
	encodeUint64Opt(b, 3, p.Address)
	for i := range p.Line {
		encodeMessage(b, 4, &p.Line[i])
	}
}

func (p *Line) encode(b *buffer) {
	encodeUint64Opt(b, 1, p.functionIDX)
	encodeInt64Opt(b, 2, p.Line)
}

func (p *Function) encode(b *buffer) {
	encodeUint64Opt(b, 1, p.ID)
	encodeInt64Opt(b, 2, p.nameX)
	encodeInt64Opt(b, 3, p.systemNameX)
	encodeInt64Opt(b, 4, p.filenameX)
	encodeInt64Opt(b, 5, p.StartLine)
}

func addString(strings map[string]int, s string) int64 {
	i, ok := strings[s]
	if !ok {
		i = len(strings)
		strings[s] = i
	}
	return int64(i)
}
