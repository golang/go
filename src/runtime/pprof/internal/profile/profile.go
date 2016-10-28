// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Package profile provides a representation of profile.proto and
// methods to encode/decode profiles in this format.
package profile

import (
	"io"
)

// Profile is an in-memory representation of profile.proto.
type Profile struct {
	SampleType []*ValueType
	Sample     []*Sample
	Mapping    []*Mapping
	Location   []*Location
	Function   []*Function

	TimeNanos     int64
	DurationNanos int64
	PeriodType    *ValueType
	Period        int64

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

// Write writes the profile as a gzip-compressed marshaled protobuf.
func (p *Profile) Write(w io.Writer) error {
	p.preEncode()
	var b buffer
	p.encode(&b)
	_, err := w.Write(b.data)
	return err
}
