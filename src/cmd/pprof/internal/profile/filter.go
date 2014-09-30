// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implements methods to filter samples from profiles.
package profile

import "regexp"

// FilterSamplesByName filters the samples in a profile and only keeps
// samples where at least one frame matches focus but none match ignore.
// Returns true is the corresponding regexp matched at least one sample.
func (p *Profile) FilterSamplesByName(focus, ignore, hide *regexp.Regexp) (fm, im, hm bool) {
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

// matchesName returns whether the function name or file in the
// location matches the regular expression.
func (loc *Location) matchesName(re *regexp.Regexp) bool {
	for _, ln := range loc.Line {
		if fn := ln.Function; fn != nil {
			if re.MatchString(fn.Name) {
				return true
			}
			if re.MatchString(fn.Filename) {
				return true
			}
		}
	}
	return false
}

// unmatchedLines returns the lines in the location that do not match
// the regular expression.
func (loc *Location) unmatchedLines(re *regexp.Regexp) []Line {
	var lines []Line
	for _, ln := range loc.Line {
		if fn := ln.Function; fn != nil {
			if re.MatchString(fn.Name) {
				continue
			}
			if re.MatchString(fn.Filename) {
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
type TagMatch func(key, val string, nval int64) bool

// FilterSamplesByTag removes all samples from the profile, except
// those that match focus and do not match the ignore regular
// expression.
func (p *Profile) FilterSamplesByTag(focus, ignore TagMatch) (fm, im bool) {
	samples := make([]*Sample, 0, len(p.Sample))
	for _, s := range p.Sample {
		focused, ignored := focusedSample(s, focus, ignore)
		fm = fm || focused
		im = im || ignored
		if focused && !ignored {
			samples = append(samples, s)
		}
	}
	p.Sample = samples
	return
}

// focusedTag checks a sample against focus and ignore regexps.
// Returns whether the focus/ignore regexps match any tags
func focusedSample(s *Sample, focus, ignore TagMatch) (fm, im bool) {
	fm = focus == nil
	for key, vals := range s.Label {
		for _, val := range vals {
			if ignore != nil && ignore(key, val, 0) {
				im = true
			}
			if !fm && focus(key, val, 0) {
				fm = true
			}
		}
	}
	for key, vals := range s.NumLabel {
		for _, val := range vals {
			if ignore != nil && ignore(key, "", val) {
				im = true
			}
			if !fm && focus(key, "", val) {
				fm = true
			}
		}
	}
	return fm, im
}
