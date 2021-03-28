// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package label_test

import (
	"bytes"
	"fmt"
	"runtime"
	"testing"
	"unsafe"

	"golang.org/x/tools/internal/event/keys"
	"golang.org/x/tools/internal/event/label"
)

var (
	AKey = keys.NewString("A", "")
	BKey = keys.NewString("B", "")
	CKey = keys.NewString("C", "")
	A    = AKey.Of("a")
	B    = BKey.Of("b")
	C    = CKey.Of("c")
	all  = []label.Label{A, B, C}
)

func TestList(t *testing.T) {
	for _, test := range []struct {
		name   string
		labels []label.Label
		expect string
	}{{
		name: "empty",
	}, {
		name:   "single",
		labels: []label.Label{A},
		expect: `A="a"`,
	}, {
		name:   "invalid",
		labels: []label.Label{{}},
		expect: ``,
	}, {
		name:   "two",
		labels: []label.Label{A, B},
		expect: `A="a", B="b"`,
	}, {
		name:   "three",
		labels: []label.Label{A, B, C},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "missing A",
		labels: []label.Label{{}, B, C},
		expect: `B="b", C="c"`,
	}, {
		name:   "missing B",
		labels: []label.Label{A, {}, C},
		expect: `A="a", C="c"`,
	}, {
		name:   "missing C",
		labels: []label.Label{A, B, {}},
		expect: `A="a", B="b"`,
	}, {
		name:   "missing AB",
		labels: []label.Label{{}, {}, C},
		expect: `C="c"`,
	}, {
		name:   "missing AC",
		labels: []label.Label{{}, B, {}},
		expect: `B="b"`,
	}, {
		name:   "missing BC",
		labels: []label.Label{A, {}, {}},
		expect: `A="a"`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			got := printList(label.NewList(test.labels...))
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestFilter(t *testing.T) {
	for _, test := range []struct {
		name    string
		labels  []label.Label
		filters []label.Key
		expect  string
	}{{
		name:   "no filters",
		labels: all,
		expect: `A="a", B="b", C="c"`,
	}, {
		name:    "no labels",
		filters: []label.Key{AKey},
		expect:  ``,
	}, {
		name:    "filter A",
		labels:  all,
		filters: []label.Key{AKey},
		expect:  `B="b", C="c"`,
	}, {
		name:    "filter B",
		labels:  all,
		filters: []label.Key{BKey},
		expect:  `A="a", C="c"`,
	}, {
		name:    "filter C",
		labels:  all,
		filters: []label.Key{CKey},
		expect:  `A="a", B="b"`,
	}, {
		name:    "filter AC",
		labels:  all,
		filters: []label.Key{AKey, CKey},
		expect:  `B="b"`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			labels := label.NewList(test.labels...)
			got := printList(label.Filter(labels, test.filters...))
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestMap(t *testing.T) {
	for _, test := range []struct {
		name   string
		labels []label.Label
		keys   []label.Key
		expect string
	}{{
		name:   "no labels",
		keys:   []label.Key{AKey},
		expect: `nil`,
	}, {
		name:   "match A",
		labels: all,
		keys:   []label.Key{AKey},
		expect: `A="a"`,
	}, {
		name:   "match B",
		labels: all,
		keys:   []label.Key{BKey},
		expect: `B="b"`,
	}, {
		name:   "match C",
		labels: all,
		keys:   []label.Key{CKey},
		expect: `C="c"`,
	}, {
		name:   "match ABC",
		labels: all,
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "missing A",
		labels: []label.Label{{}, B, C},
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `nil, B="b", C="c"`,
	}, {
		name:   "missing B",
		labels: []label.Label{A, {}, C},
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `A="a", nil, C="c"`,
	}, {
		name:   "missing C",
		labels: []label.Label{A, B, {}},
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			lm := label.NewMap(test.labels...)
			got := printMap(lm, test.keys)
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func TestMapMerge(t *testing.T) {
	for _, test := range []struct {
		name   string
		maps   []label.Map
		keys   []label.Key
		expect string
	}{{
		name:   "no maps",
		keys:   []label.Key{AKey},
		expect: `nil`,
	}, {
		name:   "one map",
		maps:   []label.Map{label.NewMap(all...)},
		keys:   []label.Key{AKey},
		expect: `A="a"`,
	}, {
		name:   "invalid map",
		maps:   []label.Map{label.NewMap()},
		keys:   []label.Key{AKey},
		expect: `nil`,
	}, {
		name:   "two maps",
		maps:   []label.Map{label.NewMap(B, C), label.NewMap(A)},
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", C="c"`,
	}, {
		name:   "invalid start map",
		maps:   []label.Map{label.NewMap(), label.NewMap(B, C)},
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `nil, B="b", C="c"`,
	}, {
		name:   "invalid mid map",
		maps:   []label.Map{label.NewMap(A), label.NewMap(), label.NewMap(C)},
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `A="a", nil, C="c"`,
	}, {
		name:   "invalid end map",
		maps:   []label.Map{label.NewMap(A, B), label.NewMap()},
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}, {
		name:   "three maps one nil",
		maps:   []label.Map{label.NewMap(A), label.NewMap(B), nil},
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}, {
		name:   "two maps one nil",
		maps:   []label.Map{label.NewMap(A, B), nil},
		keys:   []label.Key{AKey, BKey, CKey},
		expect: `A="a", B="b", nil`,
	}} {
		t.Run(test.name, func(t *testing.T) {
			tagMap := label.MergeMaps(test.maps...)
			got := printMap(tagMap, test.keys)
			if got != test.expect {
				t.Errorf("got %q want %q", got, test.expect)
			}
		})
	}
}

func printList(list label.List) string {
	buf := &bytes.Buffer{}
	for index := 0; list.Valid(index); index++ {
		l := list.Label(index)
		if !l.Valid() {
			continue
		}
		if buf.Len() > 0 {
			buf.WriteString(", ")
		}
		fmt.Fprint(buf, l)
	}
	return buf.String()
}

func printMap(lm label.Map, keys []label.Key) string {
	buf := &bytes.Buffer{}
	for _, key := range keys {
		if buf.Len() > 0 {
			buf.WriteString(", ")
		}
		fmt.Fprint(buf, lm.Find(key))
	}
	return buf.String()
}

func TestAttemptedStringCorruption(t *testing.T) {
	defer func() {
		r := recover()
		if _, ok := r.(*runtime.TypeAssertionError); !ok {
			t.Fatalf("wanted to recover TypeAssertionError, got %T", r)
		}
	}()

	var x uint64 = 12390
	p := unsafe.Pointer(&x)
	l := label.OfValue(AKey, p)
	_ = l.UnpackString()
}
