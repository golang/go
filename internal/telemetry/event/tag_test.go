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

func TestTagIterator(t *testing.T) {
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
			got := printIterator(event.NewTagIterator(test.tags...))
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
			tags := event.NewTagIterator(test.tags...)
			got := printIterator(event.Filter(tags, test.filters...))
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestTagChain(t *testing.T) {
	for _, test := range []struct {
		name   string
		tags   [][]event.Tag
		expect string
	}{{
		name:   "no iterators",
		expect: ``,
	}, {
		name:   "one iterator",
		tags:   [][]event.Tag{all},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "invalid iterator",
		tags:   [][]event.Tag{{}},
		expect: ``,
	}, {
		name:   "two iterators",
		tags:   [][]event.Tag{{B, C}, {A}},
		expect: `B="b", C="c", A="a"`,
	}, {
		name:   "invalid start iterator",
		tags:   [][]event.Tag{{}, {B, C}},
		expect: `B="b", C="c"`,
	}, {
		name:   "invalid mid iterator",
		tags:   [][]event.Tag{{A}, {}, {C}},
		expect: `A="a", C="c"`,
	}, {
		name:   "invalid end iterator",
		tags:   [][]event.Tag{{B, C}, {}},
		expect: `B="b", C="c"`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			iterators := make([]event.TagIterator, len(test.tags))
			for i, v := range test.tags {
				iterators[i] = event.NewTagIterator(v...)
			}
			got := printIterator(event.ChainTagIterators(iterators...))
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestTagMap(t *testing.T) {
	for _, test := range []struct {
		name    string
		tags    []event.Tag
		keys    []event.Key
		expect  string
		isEmpty bool
	}{{
		name:    "no tags",
		keys:    []event.Key{AKey},
		expect:  `nil`,
		isEmpty: true,
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
			if tagMap.IsEmpty() != test.isEmpty {
				t.Errorf("IsEmpty gave %v want %v", tagMap.IsEmpty(), test.isEmpty)
			}
			got := printTagMap(tagMap, test.keys)
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestTagMapMerge(t *testing.T) {
	for _, test := range []struct {
		name    string
		tags    [][]event.Tag
		keys    []event.Key
		expect  string
		isEmpty bool
	}{{
		name:    "no maps",
		keys:    []event.Key{AKey},
		expect:  `nil`,
		isEmpty: true,
	}, {
		name:   "one map",
		tags:   [][]event.Tag{all},
		keys:   []event.Key{AKey},
		expect: `A="a"`,
	}, {
		name:    "invalid map",
		tags:    [][]event.Tag{{}},
		keys:    []event.Key{AKey},
		expect:  `nil`,
		isEmpty: true,
	}, {
		name:   "two maps",
		tags:   [][]event.Tag{{B, C}, {A}},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "invalid start map",
		tags:   [][]event.Tag{{}, {B, C}},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `nil, B="b", C="c"`,
	}, {
		name:   "invalid mid map",
		tags:   [][]event.Tag{{A}, {}, {C}},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", nil, C="c"`,
	}, {
		name:   "invalid end map",
		tags:   [][]event.Tag{{A, B}, {}},
		keys:   []event.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			maps := make([]event.TagMap, len(test.tags))
			for i, v := range test.tags {
				maps[i] = event.NewTagMap(v...)
			}
			tagMap := event.MergeTagMaps(maps...)
			if tagMap.IsEmpty() != test.isEmpty {
				t.Errorf("IsEmpty gave %v want %v", tagMap.IsEmpty(), test.isEmpty)
			}
			got := printTagMap(tagMap, test.keys)
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func printIterator(it event.TagIterator) string {
	buf := &bytes.Buffer{}
	for ; it.Valid(); it.Advance() {
		if buf.Len() > 0 {
			buf.WriteString(", ")
		}
		fmt.Fprint(buf, it.Tag())
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
