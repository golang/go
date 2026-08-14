// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pgo contains the compiler-agnostic portions of PGO profile handling.
// Notably, parsing pprof profiles and serializing/deserializing from a custom
// intermediate representation.
package pgo

import (
	"errors"
	"fmt"
	"internal/profile"
	"io"
	"slices"
	"sort"
)

// nonCPUSampleTypes are the sample value types from Go's non-CPU profiles
// (heap, mutex, block). A profile whose selected value type is one of these is
// not a CPU profile and is rejected.
var nonCPUSampleTypes = []string{
	"alloc_objects", "alloc_space",
	"inuse_objects", "inuse_space",
	"contentions", "delay",
}

// FromPProf parses Profile from a pprof profile.
func FromPProf(r io.Reader) (*Profile, error) {
	p, err := profile.Parse(r)
	if errors.Is(err, profile.ErrNoData) {
		// Treat a completely empty file the same as a profile with no
		// samples: nothing to do.
		return emptyProfile(), nil
	} else if err != nil {
		return nil, fmt.Errorf("error parsing profile: %w", err)
	}

	if len(p.Sample) == 0 {
		// We accept empty profiles, but there is nothing to do.
		return emptyProfile(), nil
	}

	valueIndex, err := sampleValueIndex(p)
	if err != nil {
		return nil, err
	}

	g := profile.NewGraph(p, &profile.Options{
		SampleValue: func(v []int64) int64 { return v[valueIndex] },
	})

	if len(g.Nodes) == 0 {
		// If all sample values are 0, the graph will have no nodes.
		// In this case, treat it as an empty profile.
		return emptyProfile(), nil
	}

	namedEdgeMap, totalWeight, err := createNamedEdgeMap(g)
	if err != nil {
		return nil, err
	}

	if totalWeight == 0 {
		return emptyProfile(), nil // accept but ignore profile with no samples.
	}

	return &Profile{
		TotalWeight:  totalWeight,
		NamedEdgeMap: namedEdgeMap,
	}, nil
}

// sampleValueIndex returns the index of the sample value to use as the PGO
// weight, or an error if the profile is not a CPU profile.
func sampleValueIndex(p *profile.Profile) (int, error) {
	for i, s := range p.SampleType {
		if (s.Type == "samples" && s.Unit == "count") ||
			(s.Type == "cpu" && s.Unit == "nanoseconds") {
			return i, nil
		}
	}

	// Not a Go CPU profile. Use the default sample type, or the last value
	// type if there is no default.
	index := len(p.SampleType) - 1
	if d := p.DefaultSampleType; d != "" {
		for i, s := range p.SampleType {
			if s.Type == d {
				index = i
				break
			}
		}
	}

	if index < 0 || slices.Contains(nonCPUSampleTypes, p.SampleType[index].Type) {
		return 0, fmt.Errorf(`profile does not contain a sample index with value/type "samples/count" or "cpu/nanoseconds", and does not default to a CPU sample type`)
	}
	return index, nil
}

// createNamedEdgeMap builds a map of callsite-callee edge weights from the
// profile-graph.
//
// Caller should ignore the profile if totalWeight == 0.
func createNamedEdgeMap(g *profile.Graph) (edgeMap NamedEdgeMap, totalWeight int64, err error) {
	seenStartLine := false

	// Process graph and build various node and edge maps which will
	// be consumed by AST walk.
	weight := make(map[NamedCallEdge]int64)
	for _, n := range g.Nodes {
		seenStartLine = seenStartLine || n.Info.StartLine != 0

		canonicalName := n.Info.Name
		// Create the key to the nodeMapKey.
		namedEdge := NamedCallEdge{
			CallerName:     canonicalName,
			CallSiteOffset: n.Info.Lineno - n.Info.StartLine,
		}

		for _, e := range n.Out {
			totalWeight += e.WeightValue()
			namedEdge.CalleeName = e.Dest.Info.Name
			// Create new entry or increment existing entry.
			weight[namedEdge] += e.WeightValue()
		}
	}

	if !seenStartLine {
		// TODO(prattmic): If Function.start_line is missing we could
		// fall back to using absolute line numbers, which is better
		// than nothing.
		return NamedEdgeMap{}, 0, fmt.Errorf("profile missing Function.start_line data (Go version of profiled application too old? Go 1.20+ automatically adds this to profiles)")
	}
	return postProcessNamedEdgeMap(weight, totalWeight)
}

func sortByWeight(edges []NamedCallEdge, weight map[NamedCallEdge]int64) {
	sort.Slice(edges, func(i, j int) bool {
		ei, ej := edges[i], edges[j]
		if wi, wj := weight[ei], weight[ej]; wi != wj {
			return wi > wj // want larger weight first
		}
		// same weight, order by name/line number
		if ei.CallerName != ej.CallerName {
			return ei.CallerName < ej.CallerName
		}
		if ei.CalleeName != ej.CalleeName {
			return ei.CalleeName < ej.CalleeName
		}
		return ei.CallSiteOffset < ej.CallSiteOffset
	})
}

func postProcessNamedEdgeMap(weight map[NamedCallEdge]int64, weightVal int64) (edgeMap NamedEdgeMap, totalWeight int64, err error) {
	if weightVal == 0 {
		return NamedEdgeMap{}, 0, nil // accept but ignore profile with no samples.
	}
	byWeight := make([]NamedCallEdge, 0, len(weight))
	for namedEdge := range weight {
		byWeight = append(byWeight, namedEdge)
	}
	sortByWeight(byWeight, weight)

	edgeMap = NamedEdgeMap{
		Weight:   weight,
		ByWeight: byWeight,
	}

	totalWeight = weightVal

	return edgeMap, totalWeight, nil
}
