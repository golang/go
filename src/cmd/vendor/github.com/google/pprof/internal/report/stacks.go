// Copyright 2022 Google Inc. All Rights Reserved.
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

package report

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"regexp"

	"github.com/google/pprof/internal/measurement"
	"github.com/google/pprof/profile"
)

// StackSet holds a set of stacks corresponding to a profile.
//
// Slices in StackSet and the types it contains are always non-nil,
// which makes Javascript code that uses the JSON encoding less error-prone.
type StackSet struct {
	Total   int64         // Total value of the profile.
	Scale   float64       // Multiplier to generate displayed value
	Type    string        // Profile type. E.g., "cpu".
	Unit    string        // One of "B", "s", "GCU", or "" (if unknown)
	Stacks  []Stack       // List of stored stacks
	Sources []StackSource // Mapping from source index to info
}

// Stack holds a single stack instance.
type Stack struct {
	Value   int64 // Total value for all samples of this stack.
	Sources []int // Indices in StackSet.Sources (callers before callees).
}

// StackSource holds function/location info for a stack entry.
type StackSource struct {
	FullName   string
	FileName   string
	UniqueName string // Disambiguates functions with same names
	Inlined    bool   // If true this source was inlined into its caller

	// Alternative names to display (with decreasing lengths) to make text fit.
	// Guaranteed to be non-empty.
	Display []string

	// Regular expression (anchored) that matches exactly FullName.
	RE string

	// Places holds the list of stack slots where this source occurs.
	// In particular, if [a,b] is an element in Places,
	// StackSet.Stacks[a].Sources[b] points to this source.
	//
	// No stack will be referenced twice in the Places slice for a given
	// StackSource. In case of recursion, Places will contain the outer-most
	// entry in the recursive stack. E.g., if stack S has source X at positions
	// 4,6,9,10, the Places entry for X will contain [S,4].
	Places []StackSlot

	// Combined count of stacks where this source is the leaf.
	Self int64

	// Color number to use for this source.
	// Colors with high numbers than supported may be treated as zero.
	Color int
}

// StackSlot identifies a particular StackSlot.
type StackSlot struct {
	Stack int // Index in StackSet.Stacks
	Pos   int // Index in Stack.Sources
}

// Stacks returns a StackSet for the profile in rpt.
func (rpt *Report) Stacks() StackSet {
	// Get scale for converting to default unit of the right type.
	scale, unit := measurement.Scale(1, rpt.options.SampleUnit, "default")
	if unit == "default" {
		unit = ""
	}
	if rpt.options.Ratio > 0 {
		scale *= rpt.options.Ratio
	}
	s := &StackSet{
		Total:   rpt.total,
		Scale:   scale,
		Type:    rpt.options.SampleType,
		Unit:    unit,
		Stacks:  []Stack{},       // Ensure non-nil
		Sources: []StackSource{}, // Ensure non-nil
	}
	s.makeInitialStacks(rpt)
	s.fillPlaces()
	s.assignColors()
	return *s
}

func (s *StackSet) makeInitialStacks(rpt *Report) {
	type key struct {
		line    profile.Line
		inlined bool
	}
	srcs := map[key]int{} // Sources identified so far.
	seenFunctions := map[string]bool{}
	unknownIndex := 1
	getSrc := func(line profile.Line, inlined bool) int {
		k := key{line, inlined}
		if i, ok := srcs[k]; ok {
			return i
		}
		x := StackSource{Places: []StackSlot{}} // Ensure Places is non-nil
		if fn := line.Function; fn != nil {
			x.FullName = fn.Name
			x.FileName = fn.Filename
			if !seenFunctions[fn.Name] {
				x.UniqueName = fn.Name
				seenFunctions[fn.Name] = true
			} else {
				// Assign a different name so pivoting picks this function.
				x.UniqueName = fmt.Sprint(fn.Name, "#", fn.ID)
			}
		} else {
			x.FullName = fmt.Sprintf("?%d?", unknownIndex)
			x.UniqueName = x.FullName
			unknownIndex++
		}
		x.Inlined = inlined
		x.RE = "^" + regexp.QuoteMeta(x.UniqueName) + "$"
		x.Display = shortNameList(x.FullName)
		s.Sources = append(s.Sources, x)
		srcs[k] = len(s.Sources) - 1
		return len(s.Sources) - 1
	}

	// Synthesized root location that will be placed at the beginning of each stack.
	s.Sources = []StackSource{{
		FullName: "root",
		Display:  []string{"root"},
		Places:   []StackSlot{},
	}}

	for _, sample := range rpt.prof.Sample {
		value := rpt.options.SampleValue(sample.Value)
		stack := Stack{Value: value, Sources: []int{0}} // Start with the root

		// Note: we need to reverse the order in the produced stack.
		for i := len(sample.Location) - 1; i >= 0; i-- {
			loc := sample.Location[i]
			for j := len(loc.Line) - 1; j >= 0; j-- {
				line := loc.Line[j]
				inlined := (j != len(loc.Line)-1)
				stack.Sources = append(stack.Sources, getSrc(line, inlined))
			}
		}

		leaf := stack.Sources[len(stack.Sources)-1]
		s.Sources[leaf].Self += value
		s.Stacks = append(s.Stacks, stack)
	}
}

func (s *StackSet) fillPlaces() {
	for i, stack := range s.Stacks {
		seenSrcs := map[int]bool{}
		for j, src := range stack.Sources {
			if seenSrcs[src] {
				continue
			}
			seenSrcs[src] = true
			s.Sources[src].Places = append(s.Sources[src].Places, StackSlot{i, j})
		}
	}
}

func (s *StackSet) assignColors() {
	// Assign different color indices to different packages.
	const numColors = 1048576
	for i, src := range s.Sources {
		pkg := packageName(src.FullName)
		h := sha256.Sum256([]byte(pkg))
		index := binary.LittleEndian.Uint32(h[:])
		s.Sources[i].Color = int(index % numColors)
	}
}
