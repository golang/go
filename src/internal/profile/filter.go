// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implements methods to filter samples from profiles.

package profile

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

// focusedSample checks a sample against focus and ignore regexps.
// Returns whether the focus/ignore regexps match any tags.
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
