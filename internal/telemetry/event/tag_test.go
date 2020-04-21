// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event_test

import (
	"bytes"
	"fmt"
	"testing"

	"golang.org/x/tools/internal/telemetry/event"
)

var (
	AKey = event.NewStringKey("A", "")
	BKey = event.NewStringKey("B", "")
	CKey = event.NewStringKey("C", "")
	A    = AKey.Of("a")
	B    = BKey.Of("b")
	C    = CKey.Of("c")
	all  = []event.Tag{A, B, C}
)

func TestTagList(t *testing.T) {
	for _, test := range []struct {
		name   string
		tags   []event.Tag
		expect string
	}{{
		name: "empty",
	}, {
		name:   "single",
		tags:   []event.Tag{A},
		expect: `A="a"`,
	}, {
		name:   "invalid",
		tags:   []event.Tag{{}},
		expect: ``,
	}, {
		name:   "two",
		tags:   []event.Tag{A, B},
		expect: `A="a", B="b"`,
	}, {
		name:   "three",
		tags:   []event.Tag{A, B, C},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "missing A",
		tags:   []event.Tag{{}, B, C},
		expect: `B="b", C="c"`,
	}, {
		name:   "missing B",
		tags:   []event.Tag{A, {}, C},
		expect: `A="a", C="c"`,
	}, {
		name:   "missing C",
		tags:   []event.Tag{A, B, {}},
		expect: `A="a", B="b"`,
	}, {
		name:   "missing AB",
		tags:   []event.Tag{{}, {}, C},
		expect: `C="c"`,
	}, {
		name:   "missing AC",
		tags:   []event.Tag{{}, B, {}},
		expect: `B="b"`,
	}, {
		name:   "missing BC",
		tags:   []event.Tag{A, {}, {}},
		expect: `A="a"`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			got := printList(event.NewTagList(test.tags...))
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestTagFilter(t *testing.T) {
	for _, test := range []struct {
		name    string
		tags    []event.Tag
		filters []event.Key
		expect  string
	}{{
		name:   "no filters",
		tags:   all,
		expect: `A="a", B="b", C="c"`,
	}, {
		name:    "no tags",
		filters: []event.Key{AKey},
		expect:  ``,
	}, {
		name:    "filter A",
		tags:    all,
		filters: []event.Key{AKey},
		expect:  `B="b", C="c"`,
	}, {
		name:    "filter B",
		tags:    all,
		filters: []event.Key{BKey},
		expect:  `A="a", C="c"`,
	}, {
		name:    "filter C",
		tags:    all,
		filters: []event.Key{CKey},
		expect:  `A="a", B="b"`,
	}, {
		name:    "filter AC",
		tags:    all,
		filters: []event.Key{AKey, CKey},
		expect:  `B="b"`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			tags := event.NewTagList(test.tags...)
			got := printList(event.Filter(tags, test.filters...))
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestTagMap(t *testing.T) {
	for _, test := range []struct {
		name   string
		tags   []event.Tag
		keys   []event.Key
		expect string
	}{{
		name:   "no tags",
		keys:   []event.Key{AKey},
		expect: `nil`,
	}, {
		name:   "match A",
		tags:   all,
		keys:   []event.Key{AKey},
		expect: `A="a"`,
	}, {
		name:   "match B",
		tags:   all,
		keys:   []event.Key{BKey},
		expect: `B="b"`,
	}, {
		name:   "match C",
		tags:   all,
		keys:   []event.Key{CKey},
		expect: `C="c"`,
	}, {
		name:   "match ABC",
		tags:   all,
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "missing A",
		tags:   []event.Tag{{}, B, C},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `nil, B="b", C="c"`,
	}, {
		name:   "missing B",
		tags:   []event.Tag{A, {}, C},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", nil, C="c"`,
	}, {
		name:   "missing C",
		tags:   []event.Tag{A, B, {}},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			tagMap := event.NewTagMap(test.tags...)
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
		maps   []event.TagMap
		keys   []event.Key
		expect string
	}{{
		name:   "no maps",
		keys:   []event.Key{AKey},
		expect: `nil`,
	}, {
		name:   "one map",
		maps:   []event.TagMap{event.NewTagMap(all...)},
		keys:   []event.Key{AKey},
		expect: `A="a"`,
	}, {
		name:   "invalid map",
		maps:   []event.TagMap{event.NewTagMap()},
		keys:   []event.Key{AKey},
		expect: `nil`,
	}, {
		name:   "two maps",
		maps:   []event.TagMap{event.NewTagMap(B, C), event.NewTagMap(A)},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "invalid start map",
		maps:   []event.TagMap{event.NewTagMap(), event.NewTagMap(B, C)},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `nil, B="b", C="c"`,
	}, {
		name:   "invalid mid map",
		maps:   []event.TagMap{event.NewTagMap(A), event.NewTagMap(), event.NewTagMap(C)},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", nil, C="c"`,
	}, {
		name:   "invalid end map",
		maps:   []event.TagMap{event.NewTagMap(A, B), event.NewTagMap()},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}, {
		name:   "three maps one nil",
		maps:   []event.TagMap{event.NewTagMap(A), event.NewTagMap(B), nil},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}, {
		name:   "two maps one nil",
		maps:   []event.TagMap{event.NewTagMap(A, B), nil},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			tagMap := event.MergeTagMaps(test.maps...)
			got := printTagMap(tagMap, test.keys)
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func printList(l event.TagList) string {
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

func printTagMap(tagMap event.TagMap, keys []event.Key) string {
	buf := &bytes.Buffer{}
	for _, key := range keys {
		if buf.Len() > 0 {
			buf.WriteString(", ")
		}
		fmt.Fprint(buf, tagMap.Find(key))
	}
	return buf.String()
}
