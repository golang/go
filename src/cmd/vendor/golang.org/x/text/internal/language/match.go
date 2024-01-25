// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import "errors"

type scriptRegionFlags uint8

const (
	isList = 1 << iota
	scriptInFrom
	regionInFrom
)

func (t *Tag) setUndefinedLang(id Language) {
	if t.LangID == 0 {
		t.LangID = id
	}
}

func (t *Tag) setUndefinedScript(id Script) {
	if t.ScriptID == 0 {
		t.ScriptID = id
	}
}

func (t *Tag) setUndefinedRegion(id Region) {
	if t.RegionID == 0 || t.RegionID.Contains(id) {
		t.RegionID = id
	}
}

// ErrMissingLikelyTagsData indicates no information was available
// to compute likely values of missing tags.
var ErrMissingLikelyTagsData = errors.New("missing likely tags data")

// addLikelySubtags sets subtags to their most likely value, given the locale.
// In most cases this means setting fields for unknown values, but in some
// cases it may alter a value.  It returns an ErrMissingLikelyTagsData error
// if the given locale cannot be expanded.
func (t Tag) addLikelySubtags() (Tag, error) {
	id, err := addTags(t)
	if err != nil {
		return t, err
	} else if id.equalTags(t) {
		return t, nil
	}
	id.RemakeString()
	return id, nil
}

// specializeRegion attempts to specialize a group region.
func specializeRegion(t *Tag) bool {
	if i := regionInclusion[t.RegionID]; i < nRegionGroups {
		x := likelyRegionGroup[i]
		if Language(x.lang) == t.LangID && Script(x.script) == t.ScriptID {
			t.RegionID = Region(x.region)
		}
		return true
	}
	return false
}

// Maximize returns a new tag with missing tags filled in.
func (t Tag) Maximize() (Tag, error) {
	return addTags(t)
}

func addTags(t Tag) (Tag, error) {
	// We leave private use identifiers alone.
	if t.IsPrivateUse() {
		return t, nil
	}
	if t.ScriptID != 0 && t.RegionID != 0 {
		if t.LangID != 0 {
			// already fully specified
			specializeRegion(&t)
			return t, nil
		}
		// Search matches for und-script-region. Note that for these cases
		// region will never be a group so there is no need to check for this.
		list := likelyRegion[t.RegionID : t.RegionID+1]
		if x := list[0]; x.flags&isList != 0 {
			list = likelyRegionList[x.lang : x.lang+uint16(x.script)]
		}
		for _, x := range list {
			// Deviating from the spec. See match_test.go for details.
			if Script(x.script) == t.ScriptID {
				t.setUndefinedLang(Language(x.lang))
				return t, nil
			}
		}
	}
	if t.LangID != 0 {
		// Search matches for lang-script and lang-region, where lang != und.
		if t.LangID < langNoIndexOffset {
			x := likelyLang[t.LangID]
			if x.flags&isList != 0 {
				list := likelyLangList[x.region : x.region+uint16(x.script)]
				if t.ScriptID != 0 {
					for _, x := range list {
						if Script(x.script) == t.ScriptID && x.flags&scriptInFrom != 0 {
							t.setUndefinedRegion(Region(x.region))
							return t, nil
						}
					}
				} else if t.RegionID != 0 {
					count := 0
					goodScript := true
					tt := t
					for _, x := range list {
						// We visit all entries for which the script was not
						// defined, including the ones where the region was not
						// defined. This allows for proper disambiguation within
						// regions.
						if x.flags&scriptInFrom == 0 && t.RegionID.Contains(Region(x.region)) {
							tt.RegionID = Region(x.region)
							tt.setUndefinedScript(Script(x.script))
							goodScript = goodScript && tt.ScriptID == Script(x.script)
							count++
						}
					}
					if count == 1 {
						return tt, nil
					}
					// Even if we fail to find a unique Region, we might have
					// an unambiguous script.
					if goodScript {
						t.ScriptID = tt.ScriptID
					}
				}
			}
		}
	} else {
		// Search matches for und-script.
		if t.ScriptID != 0 {
			x := likelyScript[t.ScriptID]
			if x.region != 0 {
				t.setUndefinedRegion(Region(x.region))
				t.setUndefinedLang(Language(x.lang))
				return t, nil
			}
		}
		// Search matches for und-region. If und-script-region exists, it would
		// have been found earlier.
		if t.RegionID != 0 {
			if i := regionInclusion[t.RegionID]; i < nRegionGroups {
				x := likelyRegionGroup[i]
				if x.region != 0 {
					t.setUndefinedLang(Language(x.lang))
					t.setUndefinedScript(Script(x.script))
					t.RegionID = Region(x.region)
				}
			} else {
				x := likelyRegion[t.RegionID]
				if x.flags&isList != 0 {
					x = likelyRegionList[x.lang]
				}
				if x.script != 0 && x.flags != scriptInFrom {
					t.setUndefinedLang(Language(x.lang))
					t.setUndefinedScript(Script(x.script))
					return t, nil
				}
			}
		}
	}

	// Search matches for lang.
	if t.LangID < langNoIndexOffset {
		x := likelyLang[t.LangID]
		if x.flags&isList != 0 {
			x = likelyLangList[x.region]
		}
		if x.region != 0 {
			t.setUndefinedScript(Script(x.script))
			t.setUndefinedRegion(Region(x.region))
		}
		specializeRegion(&t)
		if t.LangID == 0 {
			t.LangID = _en // default language
		}
		return t, nil
	}
	return t, ErrMissingLikelyTagsData
}

func (t *Tag) setTagsFrom(id Tag) {
	t.LangID = id.LangID
	t.ScriptID = id.ScriptID
	t.RegionID = id.RegionID
}

// minimize removes the region or script subtags from t such that
// t.addLikelySubtags() == t.minimize().addLikelySubtags().
func (t Tag) minimize() (Tag, error) {
	t, err := minimizeTags(t)
	if err != nil {
		return t, err
	}
	t.RemakeString()
	return t, nil
}

// minimizeTags mimics the behavior of the ICU 51 C implementation.
func minimizeTags(t Tag) (Tag, error) {
	if t.equalTags(Und) {
		return t, nil
	}
	max, err := addTags(t)
	if err != nil {
		return t, err
	}
	for _, id := range [...]Tag{
		{LangID: t.LangID},
		{LangID: t.LangID, RegionID: t.RegionID},
		{LangID: t.LangID, ScriptID: t.ScriptID},
	} {
		if x, err := addTags(id); err == nil && max.equalTags(x) {
			t.setTagsFrom(id)
			break
		}
	}
	return t, nil
}
