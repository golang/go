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

	"github.com/google/pprof/internal/dotwriter"
	"github.com/google/pprof/internal/measurement"
)

// DotAttributes contains details about the graph itself, giving
// insight into how its elements should be rendered.
type DotAttributes struct {
	Nodes map[*Node]*DotNodeAttributes // A map allowing each Node to have its own visualization option
}

// DotNodeAttributes contains Node specific visualization options.
type DotNodeAttributes struct {
	// Shape is the optional shape of the node when rendered visually.
	Shape string
	// Bold indicates if the node should be bold or not.
	Bold bool
	// Peripheries is an optional number of borders to place around a node.
	Peripheries int
	// URL is an optional URL link to add to a node.
	URL string
	// Formatter formats the node's label prefix. Its output is treated as
	// untrusted plain text: ComposeDot DOT-escapes each line while preserving
	// newline ("\n") characters as centered DOT line breaks (\n).
	Formatter func(*NodeInfo) string
}

// DotConfig contains attributes about how a graph should be
// constructed and how it should look.
type DotConfig struct {
	Title     string   // The title of the DOT graph
	LegendURL string   // The URL to link to from the legend.
	Labels    []string // The labels for the DOT's legend

	FormatValue func(int64) string // A formatting function for values
	Total       int64              // The total weight of the graph, used to compute percentages
}

const maxNodelets = 4 // Number of nodelets for labels (both numeric and non)

// ComposeDot creates and writes a graph in the DOT format to the writer, using
// the configurations given.
func ComposeDot(w io.Writer, g *Graph, a *DotAttributes, c *DotConfig) {
	graphname := "unnamed"
	if c.Title != "" {
		graphname = c.Title
	}
	dw := dotwriter.Begin(w, graphname)
	defer dw.End()
	dw.Node("node",
		dotwriter.Attr{Key: "style", Val: dotwriter.ID("filled")},
		dotwriter.Attr{Key: "fillcolor", Val: "#f8f8f8"},
	)

	builder := &builder{dw: dw, attributes: a, config: c}
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

// builder composes DOT elements into a dotwriter.Writer.
type builder struct {
	dw         *dotwriter.Writer
	attributes *DotAttributes
	config     *DotConfig
}

// addLegend generates a legend in DOT format.
func (b *builder) addLegend() {
	labels := b.config.Labels
	if len(labels) == 0 {
		return
	}
	title := labels[0]
	attrs := []dotwriter.Attr{
		{Key: "shape", Val: dotwriter.ID("box")},
		{Key: "fontsize", Val: 16},
		{Key: "label", Val: dotwriter.LinesLeft(labels)},
	}
	if b.config.LegendURL != "" {
		attrs = append(attrs, dotwriter.Attr{Key: "URL", Val: b.config.LegendURL}, dotwriter.Attr{Key: "target", Val: "_blank"})
	}
	if b.config.Title != "" {
		attrs = append(attrs, dotwriter.Attr{Key: "tooltip", Val: b.config.Title})
	}
	b.dw.Legend(title, attrs...)
}

// addNode generates a graph node in DOT format.
func (b *builder) addNode(node *Node, nodeID int, maxFlat float64) {
	flat, cum := node.FlatValue(), node.CumValue()
	nodeAttrs := b.attributes.Nodes[node]

	// Populate label for node.
	formatter := multilinePrintableName
	if nodeAttrs != nil && nodeAttrs.Formatter != nil {
		formatter = nodeAttrs.Formatter
	}
	label := dotwriter.LinesCenter(strings.Split(formatter(&node.Info), "\n"))

	flatValue := b.config.FormatValue(flat)
	if flat != 0 {
		flatLine := dotwriter.Escaped(fmt.Sprintf(`%s (%s)`,
			flatValue,
			strings.TrimSpace(measurement.Percentage(flat, b.config.Total))))
		label = dotwriter.Join(label, flatLine)
	} else {
		label = dotwriter.Join(label, dotwriter.Escaped("0"))
	}
	cumValue := flatValue
	if cum != flat {
		cumLine := dotwriter.Escaped(fmt.Sprintf(`of %s (%s)`,
			b.config.FormatValue(cum),
			strings.TrimSpace(measurement.Percentage(cum, b.config.Total))))
		cumValue = b.config.FormatValue(cum)
		if flat != 0 {
			label = dotwriter.JoinLinesCenter(label, cumLine)
		} else {
			label = dotwriter.Join(label, dotwriter.Escaped(" "), cumLine)
		}
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
	if nodeAttrs != nil && nodeAttrs.Shape != "" {
		shape = nodeAttrs.Shape
	}

	cumScore := float64(node.CumValue()) / float64(abs64(b.config.Total))
	attrs := []dotwriter.Attr{
		{Key: "label", Val: label},
		{Key: "id", Val: fmt.Sprintf("node%d", nodeID)},
		{Key: "fontsize", Val: fontSize},
		{Key: "shape", Val: dotwriter.ID(shape)},
		{Key: "tooltip", Val: fmt.Sprintf("%s (%s)", node.Info.PrintableName(), cumValue)},
		{Key: "color", Val: dotColor(cumScore, false)},
		{Key: "fillcolor", Val: dotColor(cumScore, true)},
	}

	// Add on extra attributes if provided.
	if nodeAttrs != nil {
		if nodeAttrs.Bold {
			attrs = append(attrs, dotwriter.Attr{Key: "style", Val: "bold,filled"})
		}
		if nodeAttrs.Peripheries != 0 {
			attrs = append(attrs, dotwriter.Attr{Key: "peripheries", Val: nodeAttrs.Peripheries})
		}
		if nodeAttrs.URL != "" {
			// target="_blank" forces the link to open in a new tab.
			attrs = append(attrs, dotwriter.Attr{Key: "URL", Val: nodeAttrs.URL}, dotwriter.Attr{Key: "target", Val: "_blank"})
		}
	}

	b.dw.Node(dotwriter.ID(fmt.Sprintf("N%d", nodeID)), attrs...)
}

// addNodelets generates the DOT boxes for the node tags if they exist.
func (b *builder) addNodelets(node *Node, nodeID int) bool {
	wrote := false

	// Populate two Tag slices, one for LabelTags and one for NumericTags.
	var ts []*Tag
	lnts := make(map[string][]*Tag)
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
	parentID := dotwriter.ID(fmt.Sprintf("N%d", nodeID))
	for i, t := range ts {
		w := t.CumValue()
		if flatTags {
			w = t.FlatValue()
		}
		if w == 0 {
			continue
		}
		wrote = true
		weight := b.config.FormatValue(w)
		id := dotwriter.ID(fmt.Sprintf("N%d_%d", nodeID, i))
		label := dotwriter.LinesCenter(strings.Split(t.Name, "\n"))
		b.dw.Node(id,
			dotwriter.Attr{Key: "label", Val: label},
			dotwriter.Attr{Key: "id", Val: string(id)},
			dotwriter.Attr{Key: "fontsize", Val: 8},
			dotwriter.Attr{Key: "shape", Val: dotwriter.ID("box3d")},
			dotwriter.Attr{Key: "tooltip", Val: weight},
		)
		b.dw.Edge(parentID, id,
			dotwriter.Attr{Key: "label", Val: " " + weight},
			dotwriter.Attr{Key: "weight", Val: 100},
			dotwriter.Attr{Key: "tooltip", Val: weight},
			dotwriter.Attr{Key: "labeltooltip", Val: weight},
		)
		if nts := lnts[t.Name]; nts != nil {
			b.numericNodelets(nts, maxNodelets, flatTags, id)
		}
	}

	if nts := lnts[""]; nts != nil {
		if b.numericNodelets(nts, maxNodelets, flatTags, parentID) {
			wrote = true
		}
	}

	return wrote
}

func (b *builder) numericNodelets(nts []*Tag, maxNumNodelets int, flatTags bool, source dotwriter.ID) bool {
	wrote := false
	// Collapse numeric labels into maxNumNodelets buckets, of the form:
	// 1MB..2MB, 3MB..5MB, ...
	for j, t := range b.collapsedTags(nts, maxNumNodelets, flatTags) {
		w, dotted := t.CumValue(), true
		if flatTags || t.FlatValue() == t.CumValue() {
			w, dotted = t.FlatValue(), false
		}
		if w == 0 {
			continue
		}
		wrote = true
		weight := b.config.FormatValue(w)
		id := dotwriter.ID(fmt.Sprintf("N%s_%d", source, j))
		b.dw.Node(id,
			dotwriter.Attr{Key: "label", Val: t.Name},
			dotwriter.Attr{Key: "id", Val: string(id)},
			dotwriter.Attr{Key: "fontsize", Val: 8},
			dotwriter.Attr{Key: "shape", Val: dotwriter.ID("box3d")},
			dotwriter.Attr{Key: "tooltip", Val: weight},
		)
		edgeAttrs := []dotwriter.Attr{
			{Key: "label", Val: " " + weight},
			{Key: "weight", Val: 100},
			{Key: "tooltip", Val: weight},
			{Key: "labeltooltip", Val: weight},
		}
		if dotted {
			edgeAttrs = append(edgeAttrs, dotwriter.Attr{Key: "style", Val: "dotted"})
		}
		b.dw.Edge(source, id, edgeAttrs...)
	}
	return wrote
}

// addEdge generates a graph edge in DOT format.
func (b *builder) addEdge(edge *Edge, from, to int, hasNodelets bool) {
	w := b.config.FormatValue(edge.WeightValue())
	label := dotwriter.Escaped(" " + w)
	if edge.Inline {
		label = dotwriter.JoinLinesCenter(label, dotwriter.Escaped(" (inline)"))
	}
	attrs := []dotwriter.Attr{{Key: "label", Val: label}}
	if b.config.Total != 0 {
		// Note: edge.weight > b.config.Total is possible for profile diffs.
		if weight := 1 + int(min64(abs64(edge.WeightValue()*100/b.config.Total), 100)); weight > 1 {
			attrs = append(attrs, dotwriter.Attr{Key: "weight", Val: weight})
		}
		if width := 1 + int(min64(abs64(edge.WeightValue()*5/b.config.Total), 5)); width > 1 {
			attrs = append(attrs, dotwriter.Attr{Key: "penwidth", Val: width})
		}
		color := dotColor(float64(edge.WeightValue())/float64(abs64(b.config.Total)), false)
		attrs = append(attrs, dotwriter.Attr{Key: "color", Val: color})
	}
	arrow := "->"
	if edge.Residual {
		arrow = "..."
	}
	tooltip := fmt.Sprintf("%s %s %s (%s)",
		edge.Src.Info.PrintableName(), arrow, edge.Dest.Info.PrintableName(), w)
	attrs = append(attrs, dotwriter.Attr{Key: "tooltip", Val: tooltip}, dotwriter.Attr{Key: "labeltooltip", Val: tooltip})

	if edge.Residual {
		attrs = append(attrs, dotwriter.Attr{Key: "style", Val: "dotted"})
	}
	if hasNodelets {
		// Separate children further if source has tags.
		attrs = append(attrs, dotwriter.Attr{Key: "minlen", Val: 2})
	}

	b.dw.Edge(dotwriter.ID(fmt.Sprintf("N%d", from)), dotwriter.ID(fmt.Sprintf("N%d", to)), attrs...)
}

// dotColor returns a color for the given score (between -1.0 and
// 1.0), with -1.0 colored blue, 0.0 colored grey, and 1.0 colored
// red. If isBackground is true, then a light (low-saturation)
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
		b = value
		r = value * (1 + saturation*score)
		g = value * (1 + saturation*score)
	} else {
		r = value
		g = value * (1 - saturation*score)
		b = value * (1 - saturation)
	}
	return fmt.Sprintf("#%02x%02x%02x", uint8(r*255.0), uint8(g*255.0), uint8(b*255.0))
}

func multilinePrintableName(info *NodeInfo) string {
	infoCopy := *info
	infoCopy.Name = ShortenFunctionName(infoCopy.Name)
	if infoCopy.File != "" {
		infoCopy.File = filepath.Base(infoCopy.File)
	}
	if infoCopy.Objfile != "" {
		infoCopy.Objfile = filepath.Base(infoCopy.Objfile)
	}
	var lines []string
	for _, c := range infoCopy.NameComponents() {
		if infoCopy.Name != "" && c == infoCopy.Name {
			lines = append(lines, splitFunctionName(c)...)
		} else {
			lines = append(lines, c)
		}
	}
	return strings.Join(lines, "\n") + "\n"
}

func splitFunctionName(name string) []string {
	// Go type parameters are reported as "[...]" by Go pprof profiles.
	// Keep this ellipsis rather than splitting on its dots below.
	name = strings.ReplaceAll(name, "[...]", "[…]")
	var parts []string
	for _, seg := range strings.Split(name, "::") {
		parts = append(parts, strings.Split(seg, ".")...)
	}
	return parts
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
	if len(g) == 1 {
		t := g[0]
		return measurement.Label(t.Value, t.Unit), t.FlatValue(), t.CumValue()
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

	// Tags are not scaled with the selected output unit because tags are often
	// much smaller than other values which appear, so the range of tag sizes
	// sometimes would appear to be "0..0" when scaled to the selected output unit.
	return measurement.Label(min.Value, min.Unit) + ".." + measurement.Label(max.Value, max.Unit), f, c
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
