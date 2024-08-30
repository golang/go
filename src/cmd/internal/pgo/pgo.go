// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pgo contains the compiler-agnostic portions of PGO profile handling.
// Notably, parsing pprof profiles and serializing/deserializing from a custom
// intermediate representation.
package pgo

// Profile contains the processed data from the PGO profile.
type Profile struct {
	// TotalWeight is the aggregated edge weights across the profile. This
	// helps us determine the percentage threshold for hot/cold
	// partitioning.
	TotalWeight int64

	// NamedEdgeMap contains all unique call edges in the profile and their
	// edge weight.
	NamedEdgeMap NamedEdgeMap
}

// NamedCallEdge identifies a call edge by linker symbol names and call site
// offset.
type NamedCallEdge struct {
	CallerName     string
	CalleeName     string
	CallSiteOffset int // Line offset from function start line.
}

// NamedEdgeMap contains all unique call edges in the profile and their
// edge weight.
type NamedEdgeMap struct {
	Weight map[NamedCallEdge]int64

	// ByWeight lists all keys in Weight, sorted by edge weight from
	// highest to lowest.
	ByWeight []NamedCallEdge
}

func emptyProfile() *Profile {
	// Initialize empty maps/slices for easier use without a requiring a
	// nil check.
	return &Profile{
		NamedEdgeMap: NamedEdgeMap{
			ByWeight: make([]NamedCallEdge, 0),
			Weight:   make(map[NamedCallEdge]int64),
		},
	}
}

// WeightInPercentage converts profile weights to a percentage.
func WeightInPercentage(value int64, total int64) float64 {
	return (float64(value) / float64(total)) * 100
}

