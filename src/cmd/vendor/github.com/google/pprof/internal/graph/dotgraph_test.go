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

package graph

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/google/pprof/internal/proftest"
)

var updateFlag = flag.Bool("update", false, "Update the golden files")

func TestComposeWithStandardGraph(t *testing.T) {
	g := baseGraph()
	a, c := baseAttrsAndConfig()

	var buf bytes.Buffer
	ComposeDot(&buf, g, a, c)

	compareGraphs(t, buf.Bytes(), "compose1.dot")
}

func TestComposeWithNodeAttributesAndZeroFlat(t *testing.T) {
	g := baseGraph()
	a, c := baseAttrsAndConfig()

	// Set NodeAttributes for Node 1.
	a.Nodes[g.Nodes[0]] = &DotNodeAttributes{
		Shape:       "folder",
		Bold:        true,
		Peripheries: 2,
		URL:         "www.google.com",
		Formatter: func(ni *NodeInfo) string {
			return strings.ToUpper(ni.Name)
		},
	}

	// Set Flat value to zero on Node 2.
	g.Nodes[1].Flat = 0

	var buf bytes.Buffer
	ComposeDot(&buf, g, a, c)

	compareGraphs(t, buf.Bytes(), "compose2.dot")
}

func TestComposeWithTagsAndResidualEdge(t *testing.T) {
	g := baseGraph()
	a, c := baseAttrsAndConfig()

	// Add tags to Node 1.
	g.Nodes[0].LabelTags["a"] = &Tag{
		Name: "tag1",
		Cum:  10,
		Flat: 10,
	}
	g.Nodes[0].NumericTags[""] = TagMap{
		"b": &Tag{
			Name: "tag2",
			Cum:  20,
			Flat: 20,
			Unit: "ms",
		},
	}

	// Set edge to be Residual.
	g.Nodes[0].Out[g.Nodes[1]].Residual = true

	var buf bytes.Buffer
	ComposeDot(&buf, g, a, c)

	compareGraphs(t, buf.Bytes(), "compose3.dot")
}

func TestComposeWithNestedTags(t *testing.T) {
	g := baseGraph()
	a, c := baseAttrsAndConfig()

	// Add tags to Node 1.
	g.Nodes[0].LabelTags["tag1"] = &Tag{
		Name: "tag1",
		Cum:  10,
		Flat: 10,
	}
	g.Nodes[0].NumericTags["tag1"] = TagMap{
		"tag2": &Tag{
			Name: "tag2",
			Cum:  20,
			Flat: 20,
			Unit: "ms",
		},
	}

	var buf bytes.Buffer
	ComposeDot(&buf, g, a, c)

	compareGraphs(t, buf.Bytes(), "compose5.dot")
}

func TestComposeWithEmptyGraph(t *testing.T) {
	g := &Graph{}
	a, c := baseAttrsAndConfig()

	var buf bytes.Buffer
	ComposeDot(&buf, g, a, c)

	compareGraphs(t, buf.Bytes(), "compose4.dot")
}

func TestComposeWithStandardGraphAndURL(t *testing.T) {
	g := baseGraph()
	a, c := baseAttrsAndConfig()
	c.LegendURL = "http://example.com"

	var buf bytes.Buffer
	ComposeDot(&buf, g, a, c)

	compareGraphs(t, buf.Bytes(), "compose6.dot")
}

func baseGraph() *Graph {
	src := &Node{
		Info:        NodeInfo{Name: "src"},
		Flat:        10,
		Cum:         25,
		In:          make(EdgeMap),
		Out:         make(EdgeMap),
		LabelTags:   make(TagMap),
		NumericTags: make(map[string]TagMap),
	}
	dest := &Node{
		Info:        NodeInfo{Name: "dest"},
		Flat:        15,
		Cum:         25,
		In:          make(EdgeMap),
		Out:         make(EdgeMap),
		LabelTags:   make(TagMap),
		NumericTags: make(map[string]TagMap),
	}
	edge := &Edge{
		Src:    src,
		Dest:   dest,
		Weight: 10,
	}
	src.Out[dest] = edge
	src.In[src] = edge
	return &Graph{
		Nodes: Nodes{
			src,
			dest,
		},
	}
}

func baseAttrsAndConfig() (*DotAttributes, *DotConfig) {
	a := &DotAttributes{
		Nodes: make(map[*Node]*DotNodeAttributes),
	}
	c := &DotConfig{
		Title:  "testtitle",
		Labels: []string{"label1", "label2"},
		Total:  100,
		FormatValue: func(v int64) string {
			return strconv.FormatInt(v, 10)
		},
	}
	return a, c
}

func compareGraphs(t *testing.T, got []byte, wantFile string) {
	wantFile = filepath.Join("testdata", wantFile)
	want, err := ioutil.ReadFile(wantFile)
	if err != nil {
		t.Fatalf("error reading test file %s: %v", wantFile, err)
	}

	if string(got) != string(want) {
		d, err := proftest.Diff(got, want)
		if err != nil {
			t.Fatalf("error finding diff: %v", err)
		}
		t.Errorf("Compose incorrectly wrote %s", string(d))
		if *updateFlag {
			err := ioutil.WriteFile(wantFile, got, 0644)
			if err != nil {
				t.Errorf("failed to update the golden file %q: %v", wantFile, err)
			}
		}
	}
}

func TestNodeletCountCapping(t *testing.T) {
	labelTags := make(TagMap)
	for i := 0; i < 10; i++ {
		name := fmt.Sprintf("tag-%d", i)
		labelTags[name] = &Tag{
			Name: name,
			Flat: 10,
			Cum:  10,
		}
	}
	numTags := make(TagMap)
	for i := 0; i < 10; i++ {
		name := fmt.Sprintf("num-tag-%d", i)
		numTags[name] = &Tag{
			Name:  name,
			Unit:  "mb",
			Value: 16,
			Flat:  10,
			Cum:   10,
		}
	}
	node1 := &Node{
		Info:        NodeInfo{Name: "node1-with-tags"},
		Flat:        10,
		Cum:         10,
		NumericTags: map[string]TagMap{"": numTags},
		LabelTags:   labelTags,
	}
	node2 := &Node{
		Info: NodeInfo{Name: "node2"},
		Flat: 15,
		Cum:  15,
	}
	node3 := &Node{
		Info: NodeInfo{Name: "node3"},
		Flat: 15,
		Cum:  15,
	}
	g := &Graph{
		Nodes: Nodes{
			node1,
			node2,
			node3,
		},
	}
	for n := 1; n <= 3; n++ {
		input := maxNodelets + n
		if got, want := len(g.SelectTopNodes(input, true)), n; got != want {
			t.Errorf("SelectTopNodes(%d): got %d nodes, want %d", input, got, want)
		}
	}
}

func TestMultilinePrintableName(t *testing.T) {
	ni := &NodeInfo{
		Name:    "test1.test2::test3",
		File:    "src/file.cc",
		Address: 123,
		Lineno:  999,
	}

	want := fmt.Sprintf(`%016x\ntest1\ntest2\ntest3\nfile.cc:999\n`, 123)
	if got := multilinePrintableName(ni); got != want {
		t.Errorf("multilinePrintableName(%#v) == %q, want %q", ni, got, want)
	}
}

func TestTagCollapse(t *testing.T) {

	makeTag := func(name, unit string, value, flat, cum int64) *Tag {
		return &Tag{name, unit, value, flat, 0, cum, 0}
	}

	tagSource := []*Tag{
		makeTag("12mb", "mb", 12, 100, 100),
		makeTag("1kb", "kb", 1, 1, 1),
		makeTag("1mb", "mb", 1, 1000, 1000),
		makeTag("2048mb", "mb", 2048, 1000, 1000),
		makeTag("1b", "b", 1, 100, 100),
		makeTag("2b", "b", 2, 100, 100),
		makeTag("7b", "b", 7, 100, 100),
	}

	tagWant := [][]*Tag{
		{
			makeTag("1B..2GB", "", 0, 2401, 2401),
		},
		{
			makeTag("2GB", "", 0, 1000, 1000),
			makeTag("1B..12MB", "", 0, 1401, 1401),
		},
		{
			makeTag("2GB", "", 0, 1000, 1000),
			makeTag("12MB", "", 0, 100, 100),
			makeTag("1B..1MB", "", 0, 1301, 1301),
		},
		{
			makeTag("2GB", "", 0, 1000, 1000),
			makeTag("1MB", "", 0, 1000, 1000),
			makeTag("2B..1kB", "", 0, 201, 201),
			makeTag("1B", "", 0, 100, 100),
			makeTag("12MB", "", 0, 100, 100),
		},
	}

	for _, tc := range tagWant {
		var got, want []*Tag
		b := builder{nil, &DotAttributes{}, &DotConfig{}}
		got = b.collapsedTags(tagSource, len(tc), true)
		want = SortTags(tc, true)

		if !reflect.DeepEqual(got, want) {
			t.Errorf("collapse to %d, got:\n%v\nwant:\n%v", len(tc), tagString(got), tagString(want))
		}
	}
}

func tagString(t []*Tag) string {
	var ret []string
	for _, s := range t {
		ret = append(ret, fmt.Sprintln(s))
	}
	return strings.Join(ret, ":")
}
