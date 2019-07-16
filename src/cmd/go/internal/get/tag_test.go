// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package get

import "testing"

var selectTagTestTags = []string{
	"go.r58",
	"go.r58.1",
	"go.r59",
	"go.r59.1",
	"go.r61",
	"go.r61.1",
	"go.weekly.2010-01-02",
	"go.weekly.2011-10-12",
	"go.weekly.2011-10-12.1",
	"go.weekly.2011-10-14",
	"go.weekly.2011-11-01",
	"go1",
	"go1.0.1",
	"go1.999",
	"go1.9.2",
	"go5",

	// these should be ignored:
	"release.r59",
	"release.r59.1",
	"release",
	"weekly.2011-10-12",
	"weekly.2011-10-12.1",
	"weekly",
	"foo",
	"bar",
	"go.f00",
	"go!r60",
	"go.1999-01-01",
	"go.2x",
	"go.20000000000000",
	"go.2.",
	"go.2.0",
	"go2x",
	"go20000000000000",
	"go2.",
	"go2.0",
}

var selectTagTests = []struct {
	version  string
	selected string
}{
	/*
		{"release.r57", ""},
		{"release.r58.2", "go.r58.1"},
		{"release.r59", "go.r59"},
		{"release.r59.1", "go.r59.1"},
		{"release.r60", "go.r59.1"},
		{"release.r60.1", "go.r59.1"},
		{"release.r61", "go.r61"},
		{"release.r66", "go.r61.1"},
		{"weekly.2010-01-01", ""},
		{"weekly.2010-01-02", "go.weekly.2010-01-02"},
		{"weekly.2010-01-02.1", "go.weekly.2010-01-02"},
		{"weekly.2010-01-03", "go.weekly.2010-01-02"},
		{"weekly.2011-10-12", "go.weekly.2011-10-12"},
		{"weekly.2011-10-12.1", "go.weekly.2011-10-12.1"},
		{"weekly.2011-10-13", "go.weekly.2011-10-12.1"},
		{"weekly.2011-10-14", "go.weekly.2011-10-14"},
		{"weekly.2011-10-14.1", "go.weekly.2011-10-14"},
		{"weekly.2011-11-01", "go.weekly.2011-11-01"},
		{"weekly.2014-01-01", "go.weekly.2011-11-01"},
		{"weekly.3000-01-01", "go.weekly.2011-11-01"},
		{"go1", "go1"},
		{"go1.1", "go1.0.1"},
		{"go1.998", "go1.9.2"},
		{"go1.1000", "go1.999"},
		{"go6", "go5"},

		// faulty versions:
		{"release.f00", ""},
		{"weekly.1999-01-01", ""},
		{"junk", ""},
		{"", ""},
		{"go2x", ""},
		{"go200000000000", ""},
		{"go2.", ""},
		{"go2.0", ""},
	*/
	{"anything", "go1"},
}

func TestSelectTag(t *testing.T) {
	for _, c := range selectTagTests {
		selected := selectTag(c.version, selectTagTestTags)
		if selected != c.selected {
			t.Errorf("selectTag(%q) = %q, want %q", c.version, selected, c.selected)
		}
	}
}
