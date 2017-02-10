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
	"fmt"
	"io"
	"math"
	"path/filepath"
	"strings"

	"github.com/google/pprof/internal/measurement"
)

// DotAttributes contains details about the graph itself, giving
// insight into how its elements should be rendered.
type DotAttributes struct {
	Nodes map[*Node]*DotNodeAttributes // A map allowing each Node to have its own visualization option
}

// DotNodeAttributes contains Node specific visualization options.
type DotNodeAttributes struct {
	Shape       string                 // The optional shape of the node when rendered visually
	Bold        bool                   // If the node should be bold or not
	Peripheries int                    // An optional number of borders to place around a node
	URL         string                 // An optional url link to add to a node
	Formatter   func(*NodeInfo) string // An optional formatter for the node's label
}

// DotConfig contains attributes about how a graph should be
// constructed and how it should look.
type DotConfig struct {
	Title  string   // The title of the DOT graph
	Labels []string // The labels for the DOT's legend

	FormatValue func(int64) string         // A formatting function for values
	FormatTag   func(int64, string) string // A formatting function for numeric tags
	Total       int64                      // The total weight of the graph, used to compute percentages
}

// Compose creates and writes a in the DOT format to the writer, using
// the configurations given.
func ComposeDot(w io.Writer, g *Graph, a *DotAttributes, c *DotConfig) {
	builder := &builder{w, a, c}

	// Begin constructing DOT by adding a title and legend.
	builder.start()
	defer builder.finish()
	builder.addLegend()

	if len(g.Nodes) == 0 {
		return
	}

	// Preprocess graph to get id map and find max flat.
	nodeIDMap := make(map[*Node]int)
	hasNodelets := make(map[*Node]bool)

	maxFlat := float64(abs64(g.Nodes[0].FlatValue()))
	for i, n := range g.Nodes {
		nodeIDMap[n] = i + 1
		if float64(abs64(n.FlatValue())) > maxFlat {
			maxFlat = float64(abs64(n.FlatValue()))
		}
	}

	edges := EdgeMap{}

	// Add nodes and nodelets to DOT builder.
	for _, n := range g.Nodes {
		builder.addNode(n, nodeIDMap[n], maxFlat)
		hasNodelets[n] = builder.addNodelets(n, nodeIDMap[n])

		// Collect all edges. Use a fake node to support multiple incoming edges.
		for _, e := range n.Out {
			edges[&Node{}] = e
		}
	}

	// Add edges to DOT builder. Sort edges by frequency as a hint to the graph layout engine.
	for _, e := range edges.Sort() {
		builder.addEdge(e, nodeIDMap[e.Src], nodeIDMap[e.Dest], hasNodelets[e.Src])
	}
}

// builder wraps an io.Writer and understands how to compose DOT formatted elements.
type builder struct {
	io.Writer
	attributes *DotAttributes
	config     *DotConfig
}

// start generates a title and initial node in DOT format.
func (b *builder) start() {
	graphname := "unnamed"
	if b.config.Title != "" {
		graphname = b.config.Title
	}
	fmt.Fprintln(b, `digraph "`+graphname+`" {`)
	fmt.Fprintln(b, `node [style=filled fillcolor="#f8f8f8"]`)
}

// finish closes the opening curly bracket in the constructed DOT buffer.
func (b *builder) finish() {
	fmt.Fprintln(b, "}")
}

// addLegend generates a legend in DOT format.
func (b *builder) addLegend() {
	labels := b.config.Labels
	var title string
	if len(labels) > 0 {
		title = labels[0]
	}
	fmt.Fprintf(b, `subgraph cluster_L { "%s" [shape=box fontsize=16 label="%s\l"] }`+"\n", title, strings.Join(labels, `\l`))
}

// addNode generates a graph node in DOT format.
func (b *builder) addNode(node *Node, nodeID int, maxFlat float64) {
	flat, cum := node.FlatValue(), node.CumValue()
	attrs := b.attributes.Nodes[node]

	// Populate label for node.
	var label string
	if attrs != nil && attrs.Formatter != nil {
		label = attrs.Formatter(&node.Info)
	} else {
		label = multilinePrintableName(&node.Info)
	}

	flatValue := b.config.FormatValue(flat)
	if flat != 0 {
		label = label + fmt.Sprintf(`%s (%s)`,
			flatValue,
			strings.TrimSpace(percentage(flat, b.config.Total)))
	} else {
		label = label + "0"
	}
	cumValue := flatValue
	if cum != flat {
		if flat != 0 {
			label = label + `\n`
		} else {
			label = label + " "
		}
		cumValue = b.config.FormatValue(cum)
		label = label + fmt.Sprintf(`of %s (%s)`,
			cumValue,
			strings.TrimSpace(percentage(cum, b.config.Total)))
	}

	// Scale font sizes from 8 to 24 based on percentage of flat frequency.
	// Use non linear growth to emphasize the size difference.
	baseFontSize, maxFontGrowth := 8, 16.0
	fontSize := baseFontSize
	if maxFlat != 0 && flat != 0 && float64(abs64(flat)) <= maxFlat {
		fontSize += int(math.Ceil(maxFontGrowth * math.Sqrt(float64(abs64(flat))/maxFlat)))
	}

	// Determine node shape.
	shape := "box"
	if attrs != nil && attrs.Shape != "" {
		shape = attrs.Shape
	}

	// Create DOT attribute for node.
	attr := fmt.Sprintf(`label="%s" fontsize=%d shape=%s tooltip="%s (%s)" color="%s" fillcolor="%s"`,
		label, fontSize, shape, node.Info.PrintableName(), cumValue,
		dotColor(float64(node.CumValue())/float64(abs64(b.config.Total)), false),
		dotColor(float64(node.CumValue())/float64(abs64(b.config.Total)), true))

	// Add on extra attributes if provided.
	if attrs != nil {
		// Make bold if specified.
		if attrs.Bold {
			attr += ` style="bold,filled"`
		}

		// Add peripheries if specified.
		if attrs.Peripheries != 0 {
			attr += fmt.Sprintf(` peripheries=%d`, attrs.Peripheries)
		}

		// Add URL if specified. target="_blank" forces the link to open in a new tab.
		if attrs.URL != "" {
			attr += fmt.Sprintf(` URL="%s" target="_blank"`, attrs.URL)
		}
	}

	fmt.Fprintf(b, "N%d [%s]\n", nodeID, attr)
}

// addNodelets generates the DOT boxes for the node tags if they exist.
func (b *builder) addNodelets(node *Node, nodeID int) bool {
	const maxNodelets = 4    // Number of nodelets for alphanumeric labels
	const maxNumNodelets = 4 // Number of nodelets for numeric labels
	var nodelets string

	// Populate two Tag slices, one for LabelTags and one for NumericTags.
	var ts []*Tag
	lnts := make(map[string][]*Tag, 0)
	for _, t := range node.LabelTags {
		ts = append(ts, t)
	}
	for l, tm := range node.NumericTags {
		for _, t := range tm {
			lnts[l] = append(lnts[l], t)
		}
	}

	// For leaf nodes, print cumulative tags (includes weight from
	// children that have been deleted).
	// For internal nodes, print only flat tags.
	flatTags := len(node.Out) > 0

	// Select the top maxNodelets alphanumeric labels by weight.
	SortTags(ts, flatTags)
	if len(ts) > maxNodelets {
		ts = ts[:maxNodelets]
	}
	for i, t := range ts {
		w := t.CumValue()
		if flatTags {
			w = t.FlatValue()
		}
		if w == 0 {
			continue
		}
		weight := b.config.FormatValue(w)
		nodelets += fmt.Sprintf(`N%d_%d [label = "%s" fontsize=8 shape=box3d tooltip="%s"]`+"\n", nodeID, i, t.Name, weight)
		nodelets += fmt.Sprintf(`N%d -> N%d_%d [label=" %s" weight=100 tooltip="%s" labeltooltip="%s"]`+"\n", nodeID, nodeID, i, weight, weight, weight)
		if nts := lnts[t.Name]; nts != nil {
			nodelets += b.numericNodelets(nts, maxNumNodelets, flatTags, fmt.Sprintf(`N%d_%d`, nodeID, i))
		}
	}

	if nts := lnts[""]; nts != nil {
		nodelets += b.numericNodelets(nts, maxNumNodelets, flatTags, fmt.Sprintf(`N%d`, nodeID))
	}

	fmt.Fprint(b, nodelets)
	return nodelets != ""
}

func (b *builder) numericNodelets(nts []*Tag, maxNumNodelets int, flatTags bool, source string) string {
	nodelets := ""

	// Collapse numeric labels into maxNumNodelets buckets, of the form:
	// 1MB..2MB, 3MB..5MB, ...
	for j, t := range b.collapsedTags(nts, maxNumNodelets, flatTags) {
		w, attr := t.CumValue(), ` style="dotted"`
		if flatTags || t.FlatValue() == t.CumValue() {
			w, attr = t.FlatValue(), ""
		}
		if w != 0 {
			weight := b.config.FormatValue(w)
			nodelets += fmt.Sprintf(`N%s_%d [label = "%s" fontsize=8 shape=box3d tooltip="%s"]`+"\n", source, j, t.Name, weight)
			nodelets += fmt.Sprintf(`%s -> N%s_%d [label=" %s" weight=100 tooltip="%s" labeltooltip="%s"%s]`+"\n", source, source, j, weight, weight, weight, attr)
		}
	}
	return nodelets
}

// addEdge generates a graph edge in DOT format.
func (b *builder) addEdge(edge *Edge, from, to int, hasNodelets bool) {
	var inline string
	if edge.Inline {
		inline = `\n (inline)`
	}
	w := b.config.FormatValue(edge.WeightValue())
	attr := fmt.Sprintf(`label=" %s%s"`, w, inline)
	if b.config.Total != 0 {
		// Note: edge.weight > b.config.Total is possible for profile diffs.
		if weight := 1 + int(min64(abs64(edge.WeightValue()*100/b.config.Total), 100)); weight > 1 {
			attr = fmt.Sprintf(`%s weight=%d`, attr, weight)
		}
		if width := 1 + int(min64(abs64(edge.WeightValue()*5/b.config.Total), 5)); width > 1 {
			attr = fmt.Sprintf(`%s penwidth=%d`, attr, width)
		}
		attr = fmt.Sprintf(`%s color="%s"`, attr,
			dotColor(float64(edge.WeightValue())/float64(abs64(b.config.Total)), false))
	}
	arrow := "->"
	if edge.Residual {
		arrow = "..."
	}
	tooltip := fmt.Sprintf(`"%s %s %s (%s)"`,
		edge.Src.Info.PrintableName(), arrow, edge.Dest.Info.PrintableName(), w)
	attr = fmt.Sprintf(`%s tooltip=%s labeltooltip=%s`, attr, tooltip, tooltip)

	if edge.Residual {
		attr = attr + ` style="dotted"`
	}

	if hasNodelets {
		// Separate children further if source has tags.
		attr = attr + " minlen=2"
	}

	fmt.Fprintf(b, "N%d -> N%d [%s]\n", from, to, attr)
}

// dotColor returns a color for the given score (between -1.0 and
// 1.0), with -1.0 colored red, 0.0 colored grey, and 1.0 colored
// green. If isBackground is true, then a light (low-saturation)
// color is returned (suitable for use as a background color);
// otherwise, a darker color is returned (suitable for use as a
// foreground color).
func dotColor(score float64, isBackground bool) string {
	// A float between 0.0 and 1.0, indicating the extent to which
	// colors should be shifted away from grey (to make positive and
	// negative values easier to distinguish, and to make more use of
	// the color range.)
	const shift = 0.7

	// Saturation and value (in hsv colorspace) for background colors.
	const bgSaturation = 0.1
	const bgValue = 0.93

	// Saturation and value (in hsv colorspace) for foreground colors.
	const fgSaturation = 1.0
	const fgValue = 0.7

	// Choose saturation and value based on isBackground.
	var saturation float64
	var value float64
	if isBackground {
		saturation = bgSaturation
		value = bgValue
	} else {
		saturation = fgSaturation
		value = fgValue
	}

	// Limit the score values to the range [-1.0, 1.0].
	score = math.Max(-1.0, math.Min(1.0, score))

	// Reduce saturation near score=0 (so it is colored grey, rather than yellow).
	if math.Abs(score) < 0.2 {
		saturation *= math.Abs(score) / 0.2
	}

	// Apply 'shift' to move scores away from 0.0 (grey).
	if score > 0.0 {
		score = math.Pow(score, (1.0 - shift))
	}
	if score < 0.0 {
		score = -math.Pow(-score, (1.0 - shift))
	}

	var r, g, b float64 // red, green, blue
	if score < 0.0 {
		g = value
		r = value * (1 + saturation*score)
	} else {
		r = value
		g = value * (1 - saturation*score)
	}
	b = value * (1 - saturation)
	return fmt.Sprintf("#%02x%02x%02x", uint8(r*255.0), uint8(g*255.0), uint8(b*255.0))
}

// percentage computes the percentage of total of a value, and encodes
// it as a string. At least two digits of precision are printed.
func percentage(value, total int64) string {
	var ratio float64
	if total != 0 {
		ratio = math.Abs(float64(value)/float64(total)) * 100
	}
	switch {
	case math.Abs(ratio) >= 99.95 && math.Abs(ratio) <= 100.05:
		return "  100%"
	case math.Abs(ratio) >= 1.0:
		return fmt.Sprintf("%5.2f%%", ratio)
	default:
		return fmt.Sprintf("%5.2g%%", ratio)
	}
}

func multilinePrintableName(info *NodeInfo) string {
	infoCopy := *info
	infoCopy.Name = strings.Replace(infoCopy.Name, "::", `\n`, -1)
	infoCopy.Name = strings.Replace(infoCopy.Name, ".", `\n`, -1)
	if infoCopy.File != "" {
		infoCopy.File = filepath.Base(infoCopy.File)
	}
	return strings.Join(infoCopy.NameComponents(), `\n`) + `\n`
}

// collapsedTags trims and sorts a slice of tags.
func (b *builder) collapsedTags(ts []*Tag, count int, flatTags bool) []*Tag {
	ts = SortTags(ts, flatTags)
	if len(ts) <= count {
		return ts
	}

	tagGroups := make([][]*Tag, count)
	for i, t := range (ts)[:count] {
		tagGroups[i] = []*Tag{t}
	}
	for _, t := range (ts)[count:] {
		g, d := 0, tagDistance(t, tagGroups[0][0])
		for i := 1; i < count; i++ {
			if nd := tagDistance(t, tagGroups[i][0]); nd < d {
				g, d = i, nd
			}
		}
		tagGroups[g] = append(tagGroups[g], t)
	}

	var nts []*Tag
	for _, g := range tagGroups {
		l, w, c := b.tagGroupLabel(g)
		nts = append(nts, &Tag{
			Name: l,
			Flat: w,
			Cum:  c,
		})
	}
	return SortTags(nts, flatTags)
}

func tagDistance(t, u *Tag) float64 {
	v, _ := measurement.Scale(u.Value, u.Unit, t.Unit)
	if v < float64(t.Value) {
		return float64(t.Value) - v
	}
	return v - float64(t.Value)
}

func (b *builder) tagGroupLabel(g []*Tag) (label string, flat, cum int64) {
	formatTag := b.config.FormatTag
	if formatTag == nil {
		formatTag = measurement.Label
	}

	if len(g) == 1 {
		t := g[0]
		return formatTag(t.Value, t.Unit), t.FlatValue(), t.CumValue()
	}
	min := g[0]
	max := g[0]
	df, f := min.FlatDiv, min.Flat
	dc, c := min.CumDiv, min.Cum
	for _, t := range g[1:] {
		if v, _ := measurement.Scale(t.Value, t.Unit, min.Unit); int64(v) < min.Value {
			min = t
		}
		if v, _ := measurement.Scale(t.Value, t.Unit, max.Unit); int64(v) > max.Value {
			max = t
		}
		f += t.Flat
		df += t.FlatDiv
		c += t.Cum
		dc += t.CumDiv
	}
	if df != 0 {
		f = f / df
	}
	if dc != 0 {
		c = c / dc
	}
	return formatTag(min.Value, min.Unit) + ".." + formatTag(max.Value, max.Unit), f, c
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
