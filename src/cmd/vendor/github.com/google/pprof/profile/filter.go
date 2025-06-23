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

// Implements methods to filter samples from profiles.

import "regexp"

// FilterSamplesByName filters the samples in a profile and only keeps
// samples where at least one frame matches focus but none match ignore.
// Returns true is the corresponding regexp matched at least one sample.
func (p *Profile) FilterSamplesByName(focus, ignore, hide, show *regexp.Regexp) (fm, im, hm, hnm bool) {
	if focus == nil && ignore == nil && hide == nil && show == nil {
		fm = true // Missing focus implies a match
		return
	}
	focusOrIgnore := make(map[uint64]bool)
	hidden := make(map[uint64]bool)
	for _, l := range p.Location {
		if ignore != nil && l.matchesName(ignore) {
			im = true
			focusOrIgnore[l.ID] = false
		} else if focus == nil || l.matchesName(focus) {
			fm = true
			focusOrIgnore[l.ID] = true
		}

		if hide != nil && l.matchesName(hide) {
			hm = true
			l.Line = l.unmatchedLines(hide)
			if len(l.Line) == 0 {
				hidden[l.ID] = true
			}
		}
		if show != nil {
			l.Line = l.matchedLines(show)
			if len(l.Line) == 0 {
				hidden[l.ID] = true
			} else {
				hnm = true
			}
		}
	}

	s := make([]*Sample, 0, len(p.Sample))
	for _, sample := range p.Sample {
		if focusedAndNotIgnored(sample.Location, focusOrIgnore) {
			if len(hidden) > 0 {
				var locs []*Location
				for _, loc := range sample.Location {
					if !hidden[loc.ID] {
						locs = append(locs, loc)
					}
				}
				if len(locs) == 0 {
					// Remove sample with no locations (by not adding it to s).
					continue
				}
				sample.Location = locs
			}
			s = append(s, sample)
		}
	}
	p.Sample = s

	return
}

// ShowFrom drops all stack frames above the highest matching frame and returns
// whether a match was found. If showFrom is nil it returns false and does not
// modify the profile.
//
// Example: consider a sample with frames [A, B, C, B], where A is the root.
// ShowFrom(nil) returns false and has frames [A, B, C, B].
// ShowFrom(A) returns true and has frames [A, B, C, B].
// ShowFrom(B) returns true and has frames [B, C, B].
// ShowFrom(C) returns true and has frames [C, B].
// ShowFrom(D) returns false and drops the sample because no frames remain.
func (p *Profile) ShowFrom(showFrom *regexp.Regexp) (matched bool) {
	if showFrom == nil {
		return false
	}
	// showFromLocs stores location IDs that matched ShowFrom.
	showFromLocs := make(map[uint64]bool)
	// Apply to locations.
	for _, loc := range p.Location {
		if filterShowFromLocation(loc, showFrom) {
			showFromLocs[loc.ID] = true
			matched = true
		}
	}
	// For all samples, strip locations after the highest matching one.
	s := make([]*Sample, 0, len(p.Sample))
	for _, sample := range p.Sample {
		for i := len(sample.Location) - 1; i >= 0; i-- {
			if showFromLocs[sample.Location[i].ID] {
				sample.Location = sample.Location[:i+1]
				s = append(s, sample)
				break
			}
		}
	}
	p.Sample = s
	return matched
}

// filterShowFromLocation tests a showFrom regex against a location, removes
// lines after the last match and returns whether a match was found. If the
// mapping is matched, then all lines are kept.
func filterShowFromLocation(loc *Location, showFrom *regexp.Regexp) bool {
	if m := loc.Mapping; m != nil && showFrom.MatchString(m.File) {
		return true
	}
	if i := loc.lastMatchedLineIndex(showFrom); i >= 0 {
		loc.Line = loc.Line[:i+1]
		return true
	}
	return false
}

// lastMatchedLineIndex returns the index of the last line that matches a regex,
// or -1 if no match is found.
func (loc *Location) lastMatchedLineIndex(re *regexp.Regexp) int {
	for i := len(loc.Line) - 1; i >= 0; i-- {
		if fn := loc.Line[i].Function; fn != nil {
			if re.MatchString(fn.Name) || re.MatchString(fn.Filename) {
				return i
			}
		}
	}
	return -1
}

// FilterTagsByName filters the tags in a profile and only keeps
// tags that match show and not hide.
func (p *Profile) FilterTagsByName(show, hide *regexp.Regexp) (sm, hm bool) {
	matchRemove := func(name string) bool {
		matchShow := show == nil || show.MatchString(name)
		matchHide := hide != nil && hide.MatchString(name)

		if matchShow {
			sm = true
		}
		if matchHide {
			hm = true
		}
		return !matchShow || matchHide
	}
	for _, s := range p.Sample {
		for lab := range s.Label {
			if matchRemove(lab) {
				delete(s.Label, lab)
			}
		}
		for lab := range s.NumLabel {
			if matchRemove(lab) {
				delete(s.NumLabel, lab)
			}
		}
	}
	return
}

// matchesName returns whether the location matches the regular
// expression. It checks any available function names, file names, and
// mapping object filename.
func (loc *Location) matchesName(re *regexp.Regexp) bool {
	for _, ln := range loc.Line {
		if fn := ln.Function; fn != nil {
			if re.MatchString(fn.Name) || re.MatchString(fn.Filename) {
				return true
			}
		}
	}
	if m := loc.Mapping; m != nil && re.MatchString(m.File) {
		return true
	}
	return false
}

// unmatchedLines returns the lines in the location that do not match
// the regular expression.
func (loc *Location) unmatchedLines(re *regexp.Regexp) []Line {
	if m := loc.Mapping; m != nil && re.MatchString(m.File) {
		return nil
	}
	var lines []Line
	for _, ln := range loc.Line {
		if fn := ln.Function; fn != nil {
			if re.MatchString(fn.Name) || re.MatchString(fn.Filename) {
				continue
			}
		}
		lines = append(lines, ln)
	}
	return lines
}

// matchedLines returns the lines in the location that match
// the regular expression.
func (loc *Location) matchedLines(re *regexp.Regexp) []Line {
	if m := loc.Mapping; m != nil && re.MatchString(m.File) {
		return loc.Line
	}
	var lines []Line
	for _, ln := range loc.Line {
		if fn := ln.Function; fn != nil {
			if !re.MatchString(fn.Name) && !re.MatchString(fn.Filename) {
				continue
			}
		}
		lines = append(lines, ln)
	}
	return lines
}

// focusedAndNotIgnored looks up a slice of ids against a map of
// focused/ignored locations. The map only contains locations that are
// explicitly focused or ignored. Returns whether there is at least
// one focused location but no ignored locations.
func focusedAndNotIgnored(locs []*Location, m map[uint64]bool) bool {
	var f bool
	for _, loc := range locs {
		if focus, focusOrIgnore := m[loc.ID]; focusOrIgnore {
			if focus {
				// Found focused location. Must keep searching in case there
				// is an ignored one as well.
				f = true
			} else {
				// Found ignored location. Can return false right away.
				return false
			}
		}
	}
	return f
}

// TagMatch selects tags for filtering
type TagMatch func(s *Sample) bool

// FilterSamplesByTag removes all samples from the profile, except
// those that match focus and do not match the ignore regular
// expression.
func (p *Profile) FilterSamplesByTag(focus, ignore TagMatch) (fm, im bool) {
	samples := make([]*Sample, 0, len(p.Sample))
	for _, s := range p.Sample {
		focused, ignored := true, false
		if focus != nil {
			focused = focus(s)
		}
		if ignore != nil {
			ignored = ignore(s)
		}
		fm = fm || focused
		im = im || ignored
		if focused && !ignored {
			samples = append(samples, s)
		}
	}
	p.Sample = samples
	return
}
