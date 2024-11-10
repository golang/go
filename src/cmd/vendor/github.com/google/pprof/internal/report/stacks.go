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
	"path/filepath"

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
	report  *Report
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
		report:  rpt,
	}
	s.makeInitialStacks(rpt)
	s.fillPlaces()
	return *s
}

func (s *StackSet) makeInitialStacks(rpt *Report) {
	type key struct {
		funcName string
		fileName string
		line     int64
		column   int64
		inlined  bool
	}
	srcs := map[key]int{} // Sources identified so far.
	seenFunctions := map[string]bool{}
	unknownIndex := 1

	getSrc := func(line profile.Line, inlined bool) int {
		fn := line.Function
		if fn == nil {
			fn = &profile.Function{Name: fmt.Sprintf("?%d?", unknownIndex)}
			unknownIndex++
		}

		k := key{fn.Name, fn.Filename, line.Line, line.Column, inlined}
		if i, ok := srcs[k]; ok {
			return i
		}

		fileName := trimPath(fn.Filename, rpt.options.TrimPath, rpt.options.SourcePath)
		x := StackSource{
			FileName: fileName,
			Inlined:  inlined,
			Places:   []StackSlot{}, // Ensure Places is non-nil
		}
		if fn.Name != "" {
			x.FullName = addLineInfo(fn.Name, line)
			x.Display = shortNameList(x.FullName)
			x.Color = pickColor(packageName(fn.Name))
		} else { // Use file name, e.g., for file granularity display.
			x.FullName = addLineInfo(fileName, line)
			x.Display = fileNameSuffixes(x.FullName)
			x.Color = pickColor(filepath.Dir(fileName))
		}

		if !seenFunctions[x.FullName] {
			x.UniqueName = x.FullName
			seenFunctions[x.FullName] = true
		} else {
			// Assign a different name so pivoting picks this function.
			x.UniqueName = fmt.Sprint(x.FullName, "#", fn.ID)
		}

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

// pickColor picks a color for key.
func pickColor(key string) int {
	const numColors = 1048576
	h := sha256.Sum256([]byte(key))
	index := binary.LittleEndian.Uint32(h[:])
	return int(index % numColors)
}

// Legend returns the list of lines to display as the legend.
func (s *StackSet) Legend() []string {
	return reportLabels(s.report, s.report.total, len(s.Sources), len(s.Sources), 0, 0, false)
}

func addLineInfo(str string, line profile.Line) string {
	if line.Column != 0 {
		return fmt.Sprint(str, ":", line.Line, ":", line.Column)
	}
	if line.Line != 0 {
		return fmt.Sprint(str, ":", line.Line)
	}
	return str
}
