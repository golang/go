package driver

import (
	"strings"

	"github.com/google/pprof/internal/measurement"
	"github.com/google/pprof/profile"
)

// addLabelNodes adds pseudo stack frames "label:value" to each Sample with
// labels matching the supplied keys.
//
// rootKeys adds frames at the root of the callgraph (first key becomes new root).
// leafKeys adds frames at the leaf of the callgraph (last key becomes new leaf).
//
// Returns whether there were matches found for the label keys.
func addLabelNodes(p *profile.Profile, rootKeys, leafKeys []string, outputUnit string) (rootm, leafm bool) {
	// Find where to insert the new locations and functions at the end of
	// their ID spaces.
	var maxLocID uint64
	var maxFunctionID uint64
	for _, loc := range p.Location {
		if loc.ID > maxLocID {
			maxLocID = loc.ID
		}
	}
	for _, f := range p.Function {
		if f.ID > maxFunctionID {
			maxFunctionID = f.ID
		}
	}
	nextLocID := maxLocID + 1
	nextFuncID := maxFunctionID + 1

	// Intern the new locations and functions we are generating.
	type locKey struct {
		functionName, fileName string
	}
	locs := map[locKey]*profile.Location{}

	internLoc := func(locKey locKey) *profile.Location {
		loc, found := locs[locKey]
		if found {
			return loc
		}

		function := &profile.Function{
			ID:       nextFuncID,
			Name:     locKey.functionName,
			Filename: locKey.fileName,
		}
		nextFuncID++
		p.Function = append(p.Function, function)

		loc = &profile.Location{
			ID: nextLocID,
			Line: []profile.Line{
				{
					Function: function,
				},
			},
		}
		nextLocID++
		p.Location = append(p.Location, loc)
		locs[locKey] = loc
		return loc
	}

	makeLabelLocs := func(s *profile.Sample, keys []string) ([]*profile.Location, bool) {
		var locs []*profile.Location
		var match bool
		for i := range keys {
			// Loop backwards, ensuring the first tag is closest to the root,
			// and the last tag is closest to the leaves.
			k := keys[len(keys)-1-i]
			values := formatLabelValues(s, k, outputUnit)
			if len(values) > 0 {
				match = true
			}
			locKey := locKey{
				functionName: strings.Join(values, ","),
				fileName:     k,
			}
			loc := internLoc(locKey)
			locs = append(locs, loc)
		}
		return locs, match
	}

	for _, s := range p.Sample {
		rootsToAdd, sampleMatchedRoot := makeLabelLocs(s, rootKeys)
		if sampleMatchedRoot {
			rootm = true
		}
		leavesToAdd, sampleMatchedLeaf := makeLabelLocs(s, leafKeys)
		if sampleMatchedLeaf {
			leafm = true
		}

		if len(leavesToAdd)+len(rootsToAdd) == 0 {
			continue
		}

		var newLocs []*profile.Location
		newLocs = append(newLocs, leavesToAdd...)
		newLocs = append(newLocs, s.Location...)
		newLocs = append(newLocs, rootsToAdd...)
		s.Location = newLocs
	}
	return
}

// formatLabelValues returns all the string and numeric labels in Sample, with
// the numeric labels formatted according to outputUnit.
func formatLabelValues(s *profile.Sample, k string, outputUnit string) []string {
	var values []string
	values = append(values, s.Label[k]...)
	numLabels := s.NumLabel[k]
	numUnits := s.NumUnit[k]
	if len(numLabels) != len(numUnits) && len(numUnits) != 0 {
		return values
	}
	for i, numLabel := range numLabels {
		var value string
		if len(numUnits) != 0 {
			value = measurement.ScaledLabel(numLabel, numUnits[i], outputUnit)
		} else {
			value = measurement.ScaledLabel(numLabel, "", "")
		}
		values = append(values, value)
	}
	return values
}
