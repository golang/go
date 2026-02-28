// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

// CompactCoreInfo is a compact integer with the three core tags encoded.
type CompactCoreInfo uint32

// GetCompactCore generates a uint32 value that is guaranteed to be unique for
// different language, region, and script values.
func GetCompactCore(t Tag) (cci CompactCoreInfo, ok bool) {
	if t.LangID > langNoIndexOffset {
		return 0, false
	}
	cci |= CompactCoreInfo(t.LangID) << (8 + 12)
	cci |= CompactCoreInfo(t.ScriptID) << 12
	cci |= CompactCoreInfo(t.RegionID)
	return cci, true
}

// Tag generates a tag from c.
func (c CompactCoreInfo) Tag() Tag {
	return Tag{
		LangID:   Language(c >> 20),
		RegionID: Region(c & 0x3ff),
		ScriptID: Script(c>>12) & 0xff,
	}
}
