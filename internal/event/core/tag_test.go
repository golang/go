// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package core_test

import (
	"bytes"
	"fmt"
	"testing"

	"golang.org/x/tools/internal/event/core"
)

var (
	AKey = core.NewStringKey("A", "")
	BKey = core.NewStringKey("B", "")
	CKey = core.NewStringKey("C", "")
	A    = AKey.Of("a")
	B    = BKey.Of("b")
	C    = CKey.Of("c")
	all  = []core.Tag{A, B, C}
)

func TestTagList(t *testing.T) {
	for _, test := range []struct {
		name   string
		tags   []core.Tag
		expect string
	}{{
		name: "empty",
	}, {
		name:   "single",
		tags:   []core.Tag{A},
		expect: `A="a"`,
	}, {
		name:   "invalid",
		tags:   []core.Tag{{}},
		expect: ``,
	}, {
		name:   "two",
		tags:   []core.Tag{A, B},
		expect: `A="a", B="b"`,
	}, {
		name:   "three",
		tags:   []core.Tag{A, B, C},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "missing A",
		tags:   []core.Tag{{}, B, C},
		expect: `B="b", C="c"`,
	}, {
		name:   "missing B",
		tags:   []core.Tag{A, {}, C},
		expect: `A="a", C="c"`,
	}, {
		name:   "missing C",
		tags:   []core.Tag{A, B, {}},
		expect: `A="a", B="b"`,
	}, {
		name:   "missing AB",
		tags:   []core.Tag{{}, {}, C},
		expect: `C="c"`,
	}, {
		name:   "missing AC",
		tags:   []core.Tag{{}, B, {}},
		expect: `B="b"`,
	}, {
		name:   "missing BC",
		tags:   []core.Tag{A, {}, {}},
		expect: `A="a"`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			got := printList(core.NewTagList(test.tags...))
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestTagFilter(t *testing.T) {
	for _, test := range []struct {
		name    string
		tags    []core.Tag
		filters []core.Key
		expect  string
	}{{
		name:   "no filters",
		tags:   all,
		expect: `A="a", B="b", C="c"`,
	}, {
		name:    "no tags",
		filters: []core.Key{AKey},
		expect:  ``,
	}, {
		name:    "filter A",
		tags:    all,
		filters: []core.Key{AKey},
		expect:  `B="b", C="c"`,
	}, {
		name:    "filter B",
		tags:    all,
		filters: []core.Key{BKey},
		expect:  `A="a", C="c"`,
	}, {
		name:    "filter C",
		tags:    all,
		filters: []core.Key{CKey},
		expect:  `A="a", B="b"`,
	}, {
		name:    "filter AC",
		tags:    all,
		filters: []core.Key{AKey, CKey},
		expect:  `B="b"`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			tags := core.NewTagList(test.tags...)
			got := printList(core.Filter(tags, test.filters...))
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestTagMap(t *testing.T) {
	for _, test := range []struct {
		name   string
		tags   []core.Tag
		keys   []core.Key
		expect string
	}{{
		name:   "no tags",
		keys:   []core.Key{AKey},
		expect: `nil`,
	}, {
		name:   "match A",
		tags:   all,
		keys:   []core.Key{AKey},
		expect: `A="a"`,
	}, {
		name:   "match B",
		tags:   all,
		keys:   []core.Key{BKey},
		expect: `B="b"`,
	}, {
		name:   "match C",
		tags:   all,
		keys:   []core.Key{CKey},
		expect: `C="c"`,
	}, {
		name:   "match ABC",
		tags:   all,
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "missing A",
		tags:   []core.Tag{{}, B, C},
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `nil, B="b", C="c"`,
	}, {
		name:   "missing B",
		tags:   []core.Tag{A, {}, C},
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `A="a", nil, C="c"`,
	}, {
		name:   "missing C",
		tags:   []core.Tag{A, B, {}},
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			tagMap := core.NewTagMap(test.tags...)
			got := printTagMap(tagMap, test.keys)
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestTagMapMerge(t *testing.T) {
	for _, test := range []struct {
		name   string
		maps   []core.TagMap
		keys   []core.Key
		expect string
	}{{
		name:   "no maps",
		keys:   []core.Key{AKey},
		expect: `nil`,
	}, {
		name:   "one map",
		maps:   []core.TagMap{core.NewTagMap(all...)},
		keys:   []core.Key{AKey},
		expect: `A="a"`,
	}, {
		name:   "invalid map",
		maps:   []core.TagMap{core.NewTagMap()},
		keys:   []core.Key{AKey},
		expect: `nil`,
	}, {
		name:   "two maps",
		maps:   []core.TagMap{core.NewTagMap(B, C), core.NewTagMap(A)},
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "invalid start map",
		maps:   []core.TagMap{core.NewTagMap(), core.NewTagMap(B, C)},
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `nil, B="b", C="c"`,
	}, {
		name:   "invalid mid map",
		maps:   []core.TagMap{core.NewTagMap(A), core.NewTagMap(), core.NewTagMap(C)},
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `A="a", nil, C="c"`,
	}, {
		name:   "invalid end map",
		maps:   []core.TagMap{core.NewTagMap(A, B), core.NewTagMap()},
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}, {
		name:   "three maps one nil",
		maps:   []core.TagMap{core.NewTagMap(A), core.NewTagMap(B), nil},
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}, {
		name:   "two maps one nil",
		maps:   []core.TagMap{core.NewTagMap(A, B), nil},
		keys:   []core.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			tagMap := core.MergeTagMaps(test.maps...)
			got := printTagMap(tagMap, test.keys)
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func printList(l core.TagList) string {
	buf := &bytes.Buffer{}
	for index := 0; l.Valid(index); index++ {
		tag := l.Tag(index)
		if !tag.Valid() {
			continue
		}
		if buf.Len() > 0 {
			buf.WriteString(", ")
		}
		fmt.Fprint(buf, tag)
	}
	return buf.String()
}

func printTagMap(tagMap core.TagMap, keys []core.Key) string {
	buf := &bytes.Buffer{}
	for _, key := range keys {
		if buf.Len() > 0 {
			buf.WriteString(", ")
		}
		fmt.Fprint(buf, tagMap.Find(key))
	}
	return buf.String()
}
