// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package report summarizes a performance profile into a
// human-readable report.
package report

import (
	"fmt"
	"io"
	"math"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"cmd/pprof/internal/plugin"
	"internal/pprof/profile"
)

// Generate generates a report as directed by the Report.
func Generate(w io.Writer, rpt *Report, obj plugin.ObjTool) error {
	o := rpt.options

	switch o.OutputFormat {
	case Dot:
		return printDOT(w, rpt)
	case Tree:
		return printTree(w, rpt)
	case Text:
		return printText(w, rpt)
	case Raw:
		fmt.Fprint(w, rpt.prof.String())
		return nil
	case Tags:
		return printTags(w, rpt)
	case Proto:
		return rpt.prof.Write(w)
	case Dis:
		return printAssembly(w, rpt, obj)
	case List:
		return printSource(w, rpt)
	case WebList:
		return printWebSource(w, rpt, obj)
	case Callgrind:
		return printCallgrind(w, rpt)
	}
	return fmt.Errorf("unexpected output format")
}

// printAssembly prints an annotated assembly listing.
func printAssembly(w io.Writer, rpt *Report, obj plugin.ObjTool) error {
	g, err := newGraph(rpt)
	if err != nil {
		return err
	}

	o := rpt.options
	prof := rpt.prof

	// If the regexp source can be parsed as an address, also match
	// functions that land on that address.
	var address *uint64
	if hex, err := strconv.ParseUint(o.Symbol.String(), 0, 64); err == nil {
		address = &hex
	}

	fmt.Fprintln(w, "Total:", rpt.formatValue(rpt.total))
	symbols := symbolsFromBinaries(prof, g, o.Symbol, address, obj)
	symNodes := nodesPerSymbol(g.ns, symbols)
	// Sort function names for printing.
	var syms objSymbols
	for s := range symNodes {
		syms = append(syms, s)
	}
	sort.Sort(syms)

	// Correlate the symbols from the binary with the profile samples.
	for _, s := range syms {
		sns := symNodes[s]

		// Gather samples for this symbol.
		flatSum, cumSum := sumNodes(sns)

		// Get the function assembly.
		insns, err := obj.Disasm(s.sym.File, s.sym.Start, s.sym.End)
		if err != nil {
			return err
		}

		ns := annotateAssembly(insns, sns, s.base)

		fmt.Fprintf(w, "ROUTINE ======================== %s\n", s.sym.Name[0])
		for _, name := range s.sym.Name[1:] {
			fmt.Fprintf(w, "    AKA ======================== %s\n", name)
		}
		fmt.Fprintf(w, "%10s %10s (flat, cum) %s of Total\n",
			rpt.formatValue(flatSum), rpt.formatValue(cumSum),
			percentage(cumSum, rpt.total))

		for _, n := range ns {
			fmt.Fprintf(w, "%10s %10s %10x: %s\n", valueOrDot(n.flat, rpt), valueOrDot(n.cum, rpt), n.info.address, n.info.name)
		}
	}
	return nil
}

// symbolsFromBinaries examines the binaries listed on the profile
// that have associated samples, and identifies symbols matching rx.
func symbolsFromBinaries(prof *profile.Profile, g graph, rx *regexp.Regexp, address *uint64, obj plugin.ObjTool) []*objSymbol {
	hasSamples := make(map[string]bool)
	// Only examine mappings that have samples that match the
	// regexp. This is an optimization to speed up pprof.
	for _, n := range g.ns {
		if name := n.info.prettyName(); rx.MatchString(name) && n.info.objfile != "" {
			hasSamples[n.info.objfile] = true
		}
	}

	// Walk all mappings looking for matching functions with samples.
	var objSyms []*objSymbol
	for _, m := range prof.Mapping {
		if !hasSamples[m.File] {
			if address == nil || !(m.Start <= *address && *address <= m.Limit) {
				continue
			}
		}

		f, err := obj.Open(m.File, m.Start)
		if err != nil {
			fmt.Printf("%v\n", err)
			continue
		}

		// Find symbols in this binary matching the user regexp.
		var addr uint64
		if address != nil {
			addr = *address
		}
		msyms, err := f.Symbols(rx, addr)
		base := f.Base()
		f.Close()
		if err != nil {
			continue
		}
		for _, ms := range msyms {
			objSyms = append(objSyms,
				&objSymbol{
					sym:  ms,
					base: base,
				},
			)
		}
	}

	return objSyms
}

// objSym represents a symbol identified from a binary. It includes
// the SymbolInfo from the disasm package and the base that must be
// added to correspond to sample addresses
type objSymbol struct {
	sym  *plugin.Sym
	base uint64
}

// objSymbols is a wrapper type to enable sorting of []*objSymbol.
type objSymbols []*objSymbol

func (o objSymbols) Len() int {
	return len(o)
}

func (o objSymbols) Less(i, j int) bool {
	if namei, namej := o[i].sym.Name[0], o[j].sym.Name[0]; namei != namej {
		return namei < namej
	}
	return o[i].sym.Start < o[j].sym.Start
}

func (o objSymbols) Swap(i, j int) {
	o[i], o[j] = o[j], o[i]
}

// nodesPerSymbol classifies nodes into a group of symbols.
func nodesPerSymbol(ns nodes, symbols []*objSymbol) map[*objSymbol]nodes {
	symNodes := make(map[*objSymbol]nodes)
	for _, s := range symbols {
		// Gather samples for this symbol.
		for _, n := range ns {
			address := n.info.address - s.base
			if address >= s.sym.Start && address < s.sym.End {
				symNodes[s] = append(symNodes[s], n)
			}
		}
	}
	return symNodes
}

// annotateAssembly annotates a set of assembly instructions with a
// set of samples. It returns a set of nodes to display.  base is an
// offset to adjust the sample addresses.
func annotateAssembly(insns []plugin.Inst, samples nodes, base uint64) nodes {
	// Add end marker to simplify printing loop.
	insns = append(insns, plugin.Inst{
		Addr: ^uint64(0),
	})

	// Ensure samples are sorted by address.
	samples.sort(addressOrder)

	var s int
	var asm nodes
	for ix, in := range insns[:len(insns)-1] {
		n := node{
			info: nodeInfo{
				address: in.Addr,
				name:    in.Text,
				file:    trimPath(in.File),
				lineno:  in.Line,
			},
		}

		// Sum all the samples until the next instruction (to account
		// for samples attributed to the middle of an instruction).
		for next := insns[ix+1].Addr; s < len(samples) && samples[s].info.address-base < next; s++ {
			n.flat += samples[s].flat
			n.cum += samples[s].cum
			if samples[s].info.file != "" {
				n.info.file = trimPath(samples[s].info.file)
				n.info.lineno = samples[s].info.lineno
			}
		}
		asm = append(asm, &n)
	}

	return asm
}

// valueOrDot formats a value according to a report, intercepting zero
// values.
func valueOrDot(value int64, rpt *Report) string {
	if value == 0 {
		return "."
	}
	return rpt.formatValue(value)
}

// printTags collects all tags referenced in the profile and prints
// them in a sorted table.
func printTags(w io.Writer, rpt *Report) error {
	p := rpt.prof

	// Hashtable to keep accumulate tags as key,value,count.
	tagMap := make(map[string]map[string]int64)
	for _, s := range p.Sample {
		for key, vals := range s.Label {
			for _, val := range vals {
				if valueMap, ok := tagMap[key]; ok {
					valueMap[val] = valueMap[val] + s.Value[0]
					continue
				}
				valueMap := make(map[string]int64)
				valueMap[val] = s.Value[0]
				tagMap[key] = valueMap
			}
		}
		for key, vals := range s.NumLabel {
			for _, nval := range vals {
				val := scaledValueLabel(nval, key, "auto")
				if valueMap, ok := tagMap[key]; ok {
					valueMap[val] = valueMap[val] + s.Value[0]
					continue
				}
				valueMap := make(map[string]int64)
				valueMap[val] = s.Value[0]
				tagMap[key] = valueMap
			}
		}
	}

	tagKeys := make(tags, 0, len(tagMap))
	for key := range tagMap {
		tagKeys = append(tagKeys, &tag{name: key})
	}
	sort.Sort(tagKeys)

	for _, tagKey := range tagKeys {
		var total int64
		key := tagKey.name
		tags := make(tags, 0, len(tagMap[key]))
		for t, c := range tagMap[key] {
			total += c
			tags = append(tags, &tag{name: t, weight: c})
		}

		sort.Sort(tags)
		fmt.Fprintf(w, "%s: Total %d\n", key, total)
		for _, t := range tags {
			if total > 0 {
				fmt.Fprintf(w, "  %8d (%s): %s\n", t.weight,
					percentage(t.weight, total), t.name)
			} else {
				fmt.Fprintf(w, "  %8d: %s\n", t.weight, t.name)
			}
		}
		fmt.Fprintln(w)
	}
	return nil
}

// printText prints a flat text report for a profile.
func printText(w io.Writer, rpt *Report) error {
	g, err := newGraph(rpt)
	if err != nil {
		return err
	}

	origCount, droppedNodes, _ := g.preprocess(rpt)
	fmt.Fprintln(w, strings.Join(legendDetailLabels(rpt, g, origCount, droppedNodes, 0), "\n"))

	fmt.Fprintf(w, "%10s %5s%% %5s%% %10s %5s%%\n",
		"flat", "flat", "sum", "cum", "cum")

	var flatSum int64
	for _, n := range g.ns {
		name, flat, cum := n.info.prettyName(), n.flat, n.cum

		flatSum += flat
		fmt.Fprintf(w, "%10s %s %s %10s %s  %s\n",
			rpt.formatValue(flat),
			percentage(flat, rpt.total),
			percentage(flatSum, rpt.total),
			rpt.formatValue(cum),
			percentage(cum, rpt.total),
			name)
	}
	return nil
}

// printCallgrind prints a graph for a profile on callgrind format.
func printCallgrind(w io.Writer, rpt *Report) error {
	g, err := newGraph(rpt)
	if err != nil {
		return err
	}

	o := rpt.options
	rpt.options.NodeFraction = 0
	rpt.options.EdgeFraction = 0
	rpt.options.NodeCount = 0

	g.preprocess(rpt)

	fmt.Fprintln(w, "positions: instr line")
	fmt.Fprintln(w, "events:", o.SampleType+"("+o.OutputUnit+")")

	objfiles := make(map[string]int)
	files := make(map[string]int)
	names := make(map[string]int)

	// prevInfo points to the previous nodeInfo.
	// It is used to group cost lines together as much as possible.
	var prevInfo *nodeInfo
	for _, n := range g.ns {
		if prevInfo == nil || n.info.objfile != prevInfo.objfile || n.info.file != prevInfo.file || n.info.name != prevInfo.name {
			fmt.Fprintln(w)
			fmt.Fprintln(w, "ob="+callgrindName(objfiles, n.info.objfile))
			fmt.Fprintln(w, "fl="+callgrindName(files, n.info.file))
			fmt.Fprintln(w, "fn="+callgrindName(names, n.info.name))
		}

		addr := callgrindAddress(prevInfo, n.info.address)
		sv, _ := ScaleValue(n.flat, o.SampleUnit, o.OutputUnit)
		fmt.Fprintf(w, "%s %d %d\n", addr, n.info.lineno, int(sv))

		// Print outgoing edges.
		for _, out := range sortedEdges(n.out) {
			c, _ := ScaleValue(out.weight, o.SampleUnit, o.OutputUnit)
			callee := out.dest
			fmt.Fprintln(w, "cfl="+callgrindName(files, callee.info.file))
			fmt.Fprintln(w, "cfn="+callgrindName(names, callee.info.name))
			fmt.Fprintf(w, "calls=%d %s %d\n", int(c), callgrindAddress(prevInfo, callee.info.address), callee.info.lineno)
			// TODO: This address may be in the middle of a call
			// instruction. It would be best to find the beginning
			// of the instruction, but the tools seem to handle
			// this OK.
			fmt.Fprintf(w, "* * %d\n", int(c))
		}

		prevInfo = &n.info
	}

	return nil
}

// callgrindName implements the callgrind naming compression scheme.
// For names not previously seen returns "(N) name", where N is a
// unique index. For names previously seen returns "(N)" where N is
// the index returned the first time.
func callgrindName(names map[string]int, name string) string {
	if name == "" {
		return ""
	}
	if id, ok := names[name]; ok {
		return fmt.Sprintf("(%d)", id)
	}
	id := len(names) + 1
	names[name] = id
	return fmt.Sprintf("(%d) %s", id, name)
}

// callgrindAddress implements the callgrind subposition compression scheme if
// possible. If prevInfo != nil, it contains the previous address. The current
// address can be given relative to the previous address, with an explicit +/-
// to indicate it is relative, or * for the same address.
func callgrindAddress(prevInfo *nodeInfo, curr uint64) string {
	abs := fmt.Sprintf("%#x", curr)
	if prevInfo == nil {
		return abs
	}

	prev := prevInfo.address
	if prev == curr {
		return "*"
	}

	diff := int64(curr - prev)
	relative := fmt.Sprintf("%+d", diff)

	// Only bother to use the relative address if it is actually shorter.
	if len(relative) < len(abs) {
		return relative
	}

	return abs
}

// printTree prints a tree-based report in text form.
func printTree(w io.Writer, rpt *Report) error {
	const separator = "----------------------------------------------------------+-------------"
	const legend = "      flat  flat%   sum%        cum   cum%   calls calls% + context 	 	 "

	g, err := newGraph(rpt)
	if err != nil {
		return err
	}

	origCount, droppedNodes, _ := g.preprocess(rpt)
	fmt.Fprintln(w, strings.Join(legendDetailLabels(rpt, g, origCount, droppedNodes, 0), "\n"))

	fmt.Fprintln(w, separator)
	fmt.Fprintln(w, legend)
	var flatSum int64

	rx := rpt.options.Symbol
	for _, n := range g.ns {
		name, flat, cum := n.info.prettyName(), n.flat, n.cum

		// Skip any entries that do not match the regexp (for the "peek" command).
		if rx != nil && !rx.MatchString(name) {
			continue
		}

		fmt.Fprintln(w, separator)
		// Print incoming edges.
		inEdges := sortedEdges(n.in)
		inSum := inEdges.sum()
		for _, in := range inEdges {
			fmt.Fprintf(w, "%50s %s |   %s\n", rpt.formatValue(in.weight),
				percentage(in.weight, inSum), in.src.info.prettyName())
		}

		// Print current node.
		flatSum += flat
		fmt.Fprintf(w, "%10s %s %s %10s %s                | %s\n",
			rpt.formatValue(flat),
			percentage(flat, rpt.total),
			percentage(flatSum, rpt.total),
			rpt.formatValue(cum),
			percentage(cum, rpt.total),
			name)

		// Print outgoing edges.
		outEdges := sortedEdges(n.out)
		outSum := outEdges.sum()
		for _, out := range outEdges {
			fmt.Fprintf(w, "%50s %s |   %s\n", rpt.formatValue(out.weight),
				percentage(out.weight, outSum), out.dest.info.prettyName())
		}
	}
	if len(g.ns) > 0 {
		fmt.Fprintln(w, separator)
	}
	return nil
}

// printDOT prints an annotated callgraph in DOT format.
func printDOT(w io.Writer, rpt *Report) error {
	g, err := newGraph(rpt)
	if err != nil {
		return err
	}

	origCount, droppedNodes, droppedEdges := g.preprocess(rpt)

	prof := rpt.prof
	graphname := "unnamed"
	if len(prof.Mapping) > 0 {
		graphname = filepath.Base(prof.Mapping[0].File)
	}
	fmt.Fprintln(w, `digraph "`+graphname+`" {`)
	fmt.Fprintln(w, `node [style=filled fillcolor="#f8f8f8"]`)
	fmt.Fprintln(w, dotLegend(rpt, g, origCount, droppedNodes, droppedEdges))

	if len(g.ns) == 0 {
		fmt.Fprintln(w, "}")
		return nil
	}

	// Make sure nodes have a unique consistent id.
	nodeIndex := make(map[*node]int)
	maxFlat := float64(g.ns[0].flat)
	for i, n := range g.ns {
		nodeIndex[n] = i + 1
		if float64(n.flat) > maxFlat {
			maxFlat = float64(n.flat)
		}
	}
	var edges edgeList
	for _, n := range g.ns {
		node := dotNode(rpt, maxFlat, nodeIndex[n], n)
		fmt.Fprintln(w, node)
		if nodelets := dotNodelets(rpt, nodeIndex[n], n); nodelets != "" {
			fmt.Fprint(w, nodelets)
		}

		// Collect outgoing edges.
		for _, e := range n.out {
			edges = append(edges, e)
		}
	}
	// Sort edges by frequency as a hint to the graph layout engine.
	sort.Sort(edges)
	for _, e := range edges {
		fmt.Fprintln(w, dotEdge(rpt, nodeIndex[e.src], nodeIndex[e.dest], e))
	}
	fmt.Fprintln(w, "}")
	return nil
}

// percentage computes the percentage of total of a value, and encodes
// it as a string. At least two digits of precision are printed.
func percentage(value, total int64) string {
	var ratio float64
	if total != 0 {
		ratio = float64(value) / float64(total) * 100
	}
	switch {
	case ratio >= 99.95:
		return "  100%"
	case ratio >= 1.0:
		return fmt.Sprintf("%5.2f%%", ratio)
	default:
		return fmt.Sprintf("%5.2g%%", ratio)
	}
}

// dotLegend generates the overall graph label for a report in DOT format.
func dotLegend(rpt *Report, g graph, origCount, droppedNodes, droppedEdges int) string {
	label := legendLabels(rpt)
	label = append(label, legendDetailLabels(rpt, g, origCount, droppedNodes, droppedEdges)...)
	return fmt.Sprintf(`subgraph cluster_L { L [shape=box fontsize=32 label="%s\l"] }`, strings.Join(label, `\l`))
}

// legendLabels generates labels exclusive to graph visualization.
func legendLabels(rpt *Report) []string {
	prof := rpt.prof
	o := rpt.options
	var label []string
	if len(prof.Mapping) > 0 {
		if prof.Mapping[0].File != "" {
			label = append(label, "File: "+filepath.Base(prof.Mapping[0].File))
		}
		if prof.Mapping[0].BuildID != "" {
			label = append(label, "Build ID: "+prof.Mapping[0].BuildID)
		}
	}
	if o.SampleType != "" {
		label = append(label, "Type: "+o.SampleType)
	}
	if prof.TimeNanos != 0 {
		const layout = "Jan 2, 2006 at 3:04pm (MST)"
		label = append(label, "Time: "+time.Unix(0, prof.TimeNanos).Format(layout))
	}
	if prof.DurationNanos != 0 {
		label = append(label, fmt.Sprintf("Duration: %v", time.Duration(prof.DurationNanos)))
	}
	return label
}

// legendDetailLabels generates labels common to graph and text visualization.
func legendDetailLabels(rpt *Report, g graph, origCount, droppedNodes, droppedEdges int) []string {
	nodeFraction := rpt.options.NodeFraction
	edgeFraction := rpt.options.EdgeFraction
	nodeCount := rpt.options.NodeCount

	label := []string{}

	var flatSum int64
	for _, n := range g.ns {
		flatSum = flatSum + n.flat
	}

	label = append(label, fmt.Sprintf("%s of %s total (%s)", rpt.formatValue(flatSum), rpt.formatValue(rpt.total), percentage(flatSum, rpt.total)))

	if rpt.total > 0 {
		if droppedNodes > 0 {
			label = append(label, genLabel(droppedNodes, "node", "cum",
				rpt.formatValue(int64(float64(rpt.total)*nodeFraction))))
		}
		if droppedEdges > 0 {
			label = append(label, genLabel(droppedEdges, "edge", "freq",
				rpt.formatValue(int64(float64(rpt.total)*edgeFraction))))
		}
		if nodeCount > 0 && nodeCount < origCount {
			label = append(label, fmt.Sprintf("Showing top %d nodes out of %d (cum >= %s)",
				nodeCount, origCount,
				rpt.formatValue(g.ns[len(g.ns)-1].cum)))
		}
	}
	return label
}

func genLabel(d int, n, l, f string) string {
	if d > 1 {
		n = n + "s"
	}
	return fmt.Sprintf("Dropped %d %s (%s <= %s)", d, n, l, f)
}

// dotNode generates a graph node in DOT format.
func dotNode(rpt *Report, maxFlat float64, rIndex int, n *node) string {
	flat, cum := n.flat, n.cum

	labels := strings.Split(n.info.prettyName(), "::")
	label := strings.Join(labels, `\n`) + `\n`

	flatValue := rpt.formatValue(flat)
	if flat > 0 {
		label = label + fmt.Sprintf(`%s(%s)`,
			flatValue,
			strings.TrimSpace(percentage(flat, rpt.total)))
	} else {
		label = label + "0"
	}
	cumValue := flatValue
	if cum != flat {
		if flat > 0 {
			label = label + `\n`
		} else {
			label = label + " "
		}
		cumValue = rpt.formatValue(cum)
		label = label + fmt.Sprintf(`of %s(%s)`,
			cumValue,
			strings.TrimSpace(percentage(cum, rpt.total)))
	}

	// Scale font sizes from 8 to 24 based on percentage of flat frequency.
	// Use non linear growth to emphasize the size difference.
	baseFontSize, maxFontGrowth := 8, 16.0
	fontSize := baseFontSize
	if maxFlat > 0 && flat > 0 && float64(flat) <= maxFlat {
		fontSize += int(math.Ceil(maxFontGrowth * math.Sqrt(float64(flat)/maxFlat)))
	}
	return fmt.Sprintf(`N%d [label="%s" fontsize=%d shape=box tooltip="%s (%s)"]`,
		rIndex,
		label,
		fontSize, n.info.prettyName(), cumValue)
}

// dotEdge generates a graph edge in DOT format.
func dotEdge(rpt *Report, from, to int, e *edgeInfo) string {
	w := rpt.formatValue(e.weight)
	attr := fmt.Sprintf(`label=" %s"`, w)
	if rpt.total > 0 {
		if weight := 1 + int(e.weight*100/rpt.total); weight > 1 {
			attr = fmt.Sprintf(`%s weight=%d`, attr, weight)
		}
		if width := 1 + int(e.weight*5/rpt.total); width > 1 {
			attr = fmt.Sprintf(`%s penwidth=%d`, attr, width)
		}
	}
	arrow := "->"
	if e.residual {
		arrow = "..."
	}
	tooltip := fmt.Sprintf(`"%s %s %s (%s)"`,
		e.src.info.prettyName(), arrow, e.dest.info.prettyName(), w)
	attr = fmt.Sprintf(`%s tooltip=%s labeltooltip=%s`,
		attr, tooltip, tooltip)

	if e.residual {
		attr = attr + ` style="dotted"`
	}

	if len(e.src.tags) > 0 {
		// Separate children further if source has tags.
		attr = attr + " minlen=2"
	}
	return fmt.Sprintf("N%d -> N%d [%s]", from, to, attr)
}

// dotNodelets generates the DOT boxes for the node tags.
func dotNodelets(rpt *Report, rIndex int, n *node) (dot string) {
	const maxNodelets = 4    // Number of nodelets for alphanumeric labels
	const maxNumNodelets = 4 // Number of nodelets for numeric labels

	var ts, nts tags
	for _, t := range n.tags {
		if t.unit == "" {
			ts = append(ts, t)
		} else {
			nts = append(nts, t)
		}
	}

	// Select the top maxNodelets alphanumeric labels by weight
	sort.Sort(ts)
	if len(ts) > maxNodelets {
		ts = ts[:maxNodelets]
	}
	for i, t := range ts {
		weight := rpt.formatValue(t.weight)
		dot += fmt.Sprintf(`N%d_%d [label = "%s" fontsize=8 shape=box3d tooltip="%s"]`+"\n", rIndex, i, t.name, weight)
		dot += fmt.Sprintf(`N%d -> N%d_%d [label=" %s" weight=100 tooltip="\L" labeltooltip="\L"]`+"\n", rIndex, rIndex, i, weight)
	}

	// Collapse numeric labels into maxNumNodelets buckets, of the form:
	// 1MB..2MB, 3MB..5MB, ...
	nts = collapseTags(nts, maxNumNodelets)
	sort.Sort(nts)
	for i, t := range nts {
		weight := rpt.formatValue(t.weight)
		dot += fmt.Sprintf(`NN%d_%d [label = "%s" fontsize=8 shape=box3d tooltip="%s"]`+"\n", rIndex, i, t.name, weight)
		dot += fmt.Sprintf(`N%d -> NN%d_%d [label=" %s" weight=100 tooltip="\L" labeltooltip="\L"]`+"\n", rIndex, rIndex, i, weight)
	}

	return dot
}

// graph summarizes a performance profile into a format that is
// suitable for visualization.
type graph struct {
	ns nodes
}

// nodes is an ordered collection of graph nodes.
type nodes []*node

// tags represent sample annotations
type tags []*tag
type tagMap map[string]*tag

type tag struct {
	name   string
	unit   string // Describe the value, "" for non-numeric tags
	value  int64
	weight int64
}

func (t tags) Len() int      { return len(t) }
func (t tags) Swap(i, j int) { t[i], t[j] = t[j], t[i] }
func (t tags) Less(i, j int) bool {
	if t[i].weight == t[j].weight {
		return t[i].name < t[j].name
	}
	return t[i].weight > t[j].weight
}

// node is an entry on a profiling report. It represents a unique
// program location. It can include multiple names to represent
// inlined functions.
type node struct {
	info nodeInfo // Information associated to this entry.

	// values associated to this node.
	// flat is exclusive to this node, cum includes all descendents.
	flat, cum int64

	// in and out contains the nodes immediately reaching or reached by this nodes.
	in, out edgeMap

	// tags provide additional information about subsets of a sample.
	tags tagMap
}

type nodeInfo struct {
	name              string
	origName          string
	address           uint64
	file              string
	startLine, lineno int
	inline            bool
	lowPriority       bool
	objfile           string
	parent            *node // Used only if creating a calltree
}

func (n *node) addTags(s *profile.Sample, weight int64) {
	// Add a tag with all string labels
	var labels []string
	for key, vals := range s.Label {
		for _, v := range vals {
			labels = append(labels, key+":"+v)
		}
	}
	if len(labels) > 0 {
		sort.Strings(labels)
		l := n.tags.findOrAddTag(strings.Join(labels, `\n`), "", 0)
		l.weight += weight
	}

	for key, nvals := range s.NumLabel {
		for _, v := range nvals {
			label := scaledValueLabel(v, key, "auto")
			l := n.tags.findOrAddTag(label, key, v)
			l.weight += weight
		}
	}
}

func (m tagMap) findOrAddTag(label, unit string, value int64) *tag {
	if l := m[label]; l != nil {
		return l
	}
	l := &tag{
		name:  label,
		unit:  unit,
		value: value,
	}
	m[label] = l
	return l
}

// collapseTags reduces the number of entries in a tagMap by merging
// adjacent nodes into ranges. It uses a greedy approach to merge
// starting with the entries with the lowest weight.
func collapseTags(ts tags, count int) tags {
	if len(ts) <= count {
		return ts
	}

	sort.Sort(ts)
	tagGroups := make([]tags, count)
	for i, t := range ts[:count] {
		tagGroups[i] = tags{t}
	}
	for _, t := range ts[count:] {
		g, d := 0, tagDistance(t, tagGroups[0][0])
		for i := 1; i < count; i++ {
			if nd := tagDistance(t, tagGroups[i][0]); nd < d {
				g, d = i, nd
			}
		}
		tagGroups[g] = append(tagGroups[g], t)
	}

	var nts tags
	for _, g := range tagGroups {
		l, w := tagGroupLabel(g)
		nts = append(nts, &tag{
			name:   l,
			weight: w,
		})
	}
	return nts
}

func tagDistance(t, u *tag) float64 {
	v, _ := ScaleValue(u.value, u.unit, t.unit)
	if v < float64(t.value) {
		return float64(t.value) - v
	}
	return v - float64(t.value)
}

func tagGroupLabel(g tags) (string, int64) {
	if len(g) == 1 {
		t := g[0]
		return scaledValueLabel(t.value, t.unit, "auto"), t.weight
	}
	min := g[0]
	max := g[0]
	w := min.weight
	for _, t := range g[1:] {
		if v, _ := ScaleValue(t.value, t.unit, min.unit); int64(v) < min.value {
			min = t
		}
		if v, _ := ScaleValue(t.value, t.unit, max.unit); int64(v) > max.value {
			max = t
		}
		w += t.weight
	}
	return scaledValueLabel(min.value, min.unit, "auto") + ".." +
		scaledValueLabel(max.value, max.unit, "auto"), w
}

// sumNodes adds the flat and sum values on a report.
func sumNodes(ns nodes) (flat int64, cum int64) {
	for _, n := range ns {
		flat += n.flat
		cum += n.cum
	}
	return
}

type edgeMap map[*node]*edgeInfo

// edgeInfo contains any attributes to be represented about edges in a graph/
type edgeInfo struct {
	src, dest *node
	// The summary weight of the edge
	weight int64
	// residual edges connect nodes that were connected through a
	// separate node, which has been removed from the report.
	residual bool
}

// bumpWeight increases the weight of an edge. If there isn't such an
// edge in the map one is created.
func bumpWeight(from, to *node, w int64, residual bool) {
	if from.out[to] != to.in[from] {
		panic(fmt.Errorf("asymmetric edges %v %v", *from, *to))
	}

	if n := from.out[to]; n != nil {
		n.weight += w
		if n.residual && !residual {
			n.residual = false
		}
		return
	}

	info := &edgeInfo{src: from, dest: to, weight: w, residual: residual}
	from.out[to] = info
	to.in[from] = info
}

// Output formats.
const (
	Proto = iota
	Dot
	Tags
	Tree
	Text
	Raw
	Dis
	List
	WebList
	Callgrind
)

// Options are the formatting and filtering options used to generate a
// profile.
type Options struct {
	OutputFormat int

	CumSort        bool
	CallTree       bool
	PrintAddresses bool
	DropNegative   bool
	Ratio          float64

	NodeCount    int
	NodeFraction float64
	EdgeFraction float64

	SampleType string
	SampleUnit string // Unit for the sample data from the profile.
	OutputUnit string // Units for data formatting in report.

	Symbol *regexp.Regexp // Symbols to include on disassembly report.
}

// newGraph summarizes performance data from a profile into a graph.
func newGraph(rpt *Report) (g graph, err error) {
	prof := rpt.prof
	o := rpt.options

	// Generate a tree for graphical output if requested.
	buildTree := o.CallTree && o.OutputFormat == Dot

	locations := make(map[uint64][]nodeInfo)
	for _, l := range prof.Location {
		locations[l.ID] = newLocInfo(l)
	}

	nm := make(nodeMap)
	for _, sample := range prof.Sample {
		if sample.Location == nil {
			continue
		}

		// Construct list of node names for sample.
		var stack []nodeInfo
		for _, loc := range sample.Location {
			id := loc.ID
			stack = append(stack, locations[id]...)
		}

		// Upfront pass to update the parent chains, to prevent the
		// merging of nodes with different parents.
		if buildTree {
			var nn *node
			for i := len(stack); i > 0; i-- {
				n := &stack[i-1]
				n.parent = nn
				nn = nm.findOrInsertNode(*n)
			}
		}

		leaf := nm.findOrInsertNode(stack[0])
		weight := rpt.sampleValue(sample)
		leaf.addTags(sample, weight)

		// Aggregate counter data.
		leaf.flat += weight
		seen := make(map[*node]bool)
		var nn *node
		for _, s := range stack {
			n := nm.findOrInsertNode(s)
			if !seen[n] {
				seen[n] = true
				n.cum += weight

				if nn != nil {
					bumpWeight(n, nn, weight, false)
				}
			}
			nn = n
		}
	}

	// Collect new nodes into a report.
	ns := make(nodes, 0, len(nm))
	for _, n := range nm {
		if rpt.options.DropNegative && n.flat < 0 {
			continue
		}
		ns = append(ns, n)
	}

	return graph{ns}, nil
}

// Create a slice of formatted names for a location.
func newLocInfo(l *profile.Location) []nodeInfo {
	var objfile string

	if m := l.Mapping; m != nil {
		objfile = m.File
	}

	if len(l.Line) == 0 {
		return []nodeInfo{
			{
				address: l.Address,
				objfile: objfile,
			},
		}
	}
	var info []nodeInfo
	numInlineFrames := len(l.Line) - 1
	for li, line := range l.Line {
		ni := nodeInfo{
			address: l.Address,
			lineno:  int(line.Line),
			inline:  li < numInlineFrames,
			objfile: objfile,
		}

		if line.Function != nil {
			ni.name = line.Function.Name
			ni.origName = line.Function.SystemName
			ni.file = line.Function.Filename
			ni.startLine = int(line.Function.StartLine)
		}

		info = append(info, ni)
	}
	return info
}

// nodeMap maps from a node info struct to a node. It is used to merge
// report entries with the same info.
type nodeMap map[nodeInfo]*node

func (m nodeMap) findOrInsertNode(info nodeInfo) *node {
	rr := m[info]
	if rr == nil {
		rr = &node{
			info: info,
			in:   make(edgeMap),
			out:  make(edgeMap),
			tags: make(map[string]*tag),
		}
		m[info] = rr
	}
	return rr
}

// preprocess does any required filtering/sorting according to the
// report options. Returns the mapping from each node to any nodes
// removed by path compression and statistics on the nodes/edges removed.
func (g *graph) preprocess(rpt *Report) (origCount, droppedNodes, droppedEdges int) {
	o := rpt.options

	// Compute total weight of current set of nodes.
	// This is <= rpt.total because of node filtering.
	var totalValue int64
	for _, n := range g.ns {
		totalValue += n.flat
	}

	// Remove nodes with value <= total*nodeFraction
	if nodeFraction := o.NodeFraction; nodeFraction > 0 {
		var removed nodes
		minValue := int64(float64(totalValue) * nodeFraction)
		kept := make(nodes, 0, len(g.ns))
		for _, n := range g.ns {
			if n.cum < minValue {
				removed = append(removed, n)
			} else {
				kept = append(kept, n)
				tagsKept := make(map[string]*tag)
				for s, t := range n.tags {
					if t.weight >= minValue {
						tagsKept[s] = t
					}
				}
				n.tags = tagsKept
			}
		}
		droppedNodes = len(removed)
		removeNodes(removed, false, false)
		g.ns = kept
	}

	// Remove edges below minimum frequency.
	if edgeFraction := o.EdgeFraction; edgeFraction > 0 {
		minEdge := int64(float64(totalValue) * edgeFraction)
		for _, n := range g.ns {
			for src, e := range n.in {
				if e.weight < minEdge {
					delete(n.in, src)
					delete(src.out, n)
					droppedEdges++
				}
			}
		}
	}

	sortOrder := flatName
	if o.CumSort {
		// Force cum sorting for graph output, to preserve connectivity.
		sortOrder = cumName
	}

	// Nodes that have flat==0 and a single in/out do not provide much
	// information. Give them first chance to be removed. Do not consider edges
	// from/to nodes that are expected to be removed.
	maxNodes := o.NodeCount
	if o.OutputFormat == Dot {
		if maxNodes > 0 && maxNodes < len(g.ns) {
			sortOrder = cumName
			g.ns.sort(cumName)
			cumCutoff := g.ns[maxNodes].cum
			for _, n := range g.ns {
				if n.flat == 0 {
					if count := countEdges(n.out, cumCutoff); count > 1 {
						continue
					}
					if count := countEdges(n.in, cumCutoff); count != 1 {
						continue
					}
					n.info.lowPriority = true
				}
			}
		}
	}

	g.ns.sort(sortOrder)
	if maxNodes > 0 {
		origCount = len(g.ns)
		for index, nodes := 0, 0; index < len(g.ns); index++ {
			nodes++
			// For DOT output, count the tags as nodes since we will draw
			// boxes for them.
			if o.OutputFormat == Dot {
				nodes += len(g.ns[index].tags)
			}
			if nodes > maxNodes {
				// Trim to the top n nodes. Create dotted edges to bridge any
				// broken connections.
				removeNodes(g.ns[index:], true, true)
				g.ns = g.ns[:index]
				break
			}
		}
	}
	removeRedundantEdges(g.ns)

	// Select best unit for profile output.
	// Find the appropriate units for the smallest non-zero sample
	if o.OutputUnit == "minimum" && len(g.ns) > 0 {
		var maxValue, minValue int64

		for _, n := range g.ns {
			if n.flat > 0 && (minValue == 0 || n.flat < minValue) {
				minValue = n.flat
			}
			if n.cum > maxValue {
				maxValue = n.cum
			}
		}
		if r := o.Ratio; r > 0 && r != 1 {
			minValue = int64(float64(minValue) * r)
			maxValue = int64(float64(maxValue) * r)
		}

		_, minUnit := ScaleValue(minValue, o.SampleUnit, "minimum")
		_, maxUnit := ScaleValue(maxValue, o.SampleUnit, "minimum")

		unit := minUnit
		if minUnit != maxUnit && minValue*100 < maxValue && o.OutputFormat != Callgrind {
			// Minimum and maximum values have different units. Scale
			// minimum by 100 to use larger units, allowing minimum value to
			// be scaled down to 0.01, except for callgrind reports since
			// they can only represent integer values.
			_, unit = ScaleValue(100*minValue, o.SampleUnit, "minimum")
		}

		if unit != "" {
			o.OutputUnit = unit
		} else {
			o.OutputUnit = o.SampleUnit
		}
	}
	return
}

// countEdges counts the number of edges below the specified cutoff.
func countEdges(el edgeMap, cutoff int64) int {
	count := 0
	for _, e := range el {
		if e.weight > cutoff {
			count++
		}
	}
	return count
}

// removeNodes removes nodes from a report, optionally bridging
// connections between in/out edges and spreading out their weights
// proportionally. residual marks new bridge edges as residual
// (dotted).
func removeNodes(toRemove nodes, bridge, residual bool) {
	for _, n := range toRemove {
		for ei := range n.in {
			delete(ei.out, n)
		}
		if bridge {
			for ei, wi := range n.in {
				for eo, wo := range n.out {
					var weight int64
					if n.cum != 0 {
						weight = int64(float64(wo.weight) * (float64(wi.weight) / float64(n.cum)))
					}
					bumpWeight(ei, eo, weight, residual)
				}
			}
		}
		for eo := range n.out {
			delete(eo.in, n)
		}
	}
}

// removeRedundantEdges removes residual edges if the destination can
// be reached through another path. This is done to simplify the graph
// while preserving connectivity.
func removeRedundantEdges(ns nodes) {
	// Walk the nodes and outgoing edges in reverse order to prefer
	// removing edges with the lowest weight.
	for i := len(ns); i > 0; i-- {
		n := ns[i-1]
		in := sortedEdges(n.in)
		for j := len(in); j > 0; j-- {
			if e := in[j-1]; e.residual && isRedundant(e) {
				delete(e.src.out, e.dest)
				delete(e.dest.in, e.src)
			}
		}
	}
}

// isRedundant determines if an edge can be removed without impacting
// connectivity of the whole graph. This is implemented by checking if the
// nodes have a common ancestor after removing the edge.
func isRedundant(e *edgeInfo) bool {
	destPred := predecessors(e, e.dest)
	if len(destPred) == 1 {
		return false
	}
	srcPred := predecessors(e, e.src)

	for n := range srcPred {
		if destPred[n] && n != e.dest {
			return true
		}
	}
	return false
}

// predecessors collects all the predecessors to node n, excluding edge e.
func predecessors(e *edgeInfo, n *node) map[*node]bool {
	seen := map[*node]bool{n: true}
	queue := []*node{n}
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		for _, ie := range n.in {
			if e == ie || seen[ie.src] {
				continue
			}
			seen[ie.src] = true
			queue = append(queue, ie.src)
		}
	}
	return seen
}

// nodeSorter is a mechanism used to allow a report to be sorted
// in different ways.
type nodeSorter struct {
	rs   nodes
	less func(i, j int) bool
}

func (s nodeSorter) Len() int           { return len(s.rs) }
func (s nodeSorter) Swap(i, j int)      { s.rs[i], s.rs[j] = s.rs[j], s.rs[i] }
func (s nodeSorter) Less(i, j int) bool { return s.less(i, j) }

type nodeOrder int

const (
	flatName nodeOrder = iota
	flatCumName
	cumName
	nameOrder
	fileOrder
	addressOrder
)

// sort reorders the entries in a report based on the specified
// ordering criteria. The result is sorted in decreasing order for
// numeric quantities, alphabetically for text, and increasing for
// addresses.
func (ns nodes) sort(o nodeOrder) error {
	var s nodeSorter

	switch o {
	case flatName:
		s = nodeSorter{ns,
			func(i, j int) bool {
				if iv, jv := ns[i].flat, ns[j].flat; iv != jv {
					return iv > jv
				}
				if ns[i].info.prettyName() != ns[j].info.prettyName() {
					return ns[i].info.prettyName() < ns[j].info.prettyName()
				}
				iv, jv := ns[i].cum, ns[j].cum
				return iv > jv
			},
		}
	case flatCumName:
		s = nodeSorter{ns,
			func(i, j int) bool {
				if iv, jv := ns[i].flat, ns[j].flat; iv != jv {
					return iv > jv
				}
				if iv, jv := ns[i].cum, ns[j].cum; iv != jv {
					return iv > jv
				}
				return ns[i].info.prettyName() < ns[j].info.prettyName()
			},
		}
	case cumName:
		s = nodeSorter{ns,
			func(i, j int) bool {
				if ns[i].info.lowPriority != ns[j].info.lowPriority {
					return ns[j].info.lowPriority
				}
				if iv, jv := ns[i].cum, ns[j].cum; iv != jv {
					return iv > jv
				}
				if ns[i].info.prettyName() != ns[j].info.prettyName() {
					return ns[i].info.prettyName() < ns[j].info.prettyName()
				}
				iv, jv := ns[i].flat, ns[j].flat
				return iv > jv
			},
		}
	case nameOrder:
		s = nodeSorter{ns,
			func(i, j int) bool {
				return ns[i].info.name < ns[j].info.name
			},
		}
	case fileOrder:
		s = nodeSorter{ns,
			func(i, j int) bool {
				return ns[i].info.file < ns[j].info.file
			},
		}
	case addressOrder:
		s = nodeSorter{ns,
			func(i, j int) bool {
				return ns[i].info.address < ns[j].info.address
			},
		}
	default:
		return fmt.Errorf("report: unrecognized sort ordering: %d", o)
	}
	sort.Sort(s)
	return nil
}

type edgeList []*edgeInfo

// sortedEdges return a slice of the edges in the map, sorted for
// visualization. The sort order is first based on the edge weight
// (higher-to-lower) and then by the node names to avoid flakiness.
func sortedEdges(edges map[*node]*edgeInfo) edgeList {
	el := make(edgeList, 0, len(edges))
	for _, w := range edges {
		el = append(el, w)
	}

	sort.Sort(el)
	return el
}

func (el edgeList) Len() int {
	return len(el)
}

func (el edgeList) Less(i, j int) bool {
	if el[i].weight != el[j].weight {
		return el[i].weight > el[j].weight
	}

	from1 := el[i].src.info.prettyName()
	from2 := el[j].src.info.prettyName()
	if from1 != from2 {
		return from1 < from2
	}

	to1 := el[i].dest.info.prettyName()
	to2 := el[j].dest.info.prettyName()

	return to1 < to2
}

func (el edgeList) Swap(i, j int) {
	el[i], el[j] = el[j], el[i]
}

func (el edgeList) sum() int64 {
	var ret int64
	for _, e := range el {
		ret += e.weight
	}
	return ret
}

// ScaleValue reformats a value from a unit to a different unit.
func ScaleValue(value int64, fromUnit, toUnit string) (sv float64, su string) {
	// Avoid infinite recursion on overflow.
	if value < 0 && -value > 0 {
		v, u := ScaleValue(-value, fromUnit, toUnit)
		return -v, u
	}
	if m, u, ok := memoryLabel(value, fromUnit, toUnit); ok {
		return m, u
	}
	if t, u, ok := timeLabel(value, fromUnit, toUnit); ok {
		return t, u
	}
	// Skip non-interesting units.
	switch toUnit {
	case "count", "sample", "unit", "minimum":
		return float64(value), ""
	default:
		return float64(value), toUnit
	}
}

func scaledValueLabel(value int64, fromUnit, toUnit string) string {
	v, u := ScaleValue(value, fromUnit, toUnit)

	sv := strings.TrimSuffix(fmt.Sprintf("%.2f", v), ".00")
	if sv == "0" || sv == "-0" {
		return "0"
	}
	return sv + u
}

func memoryLabel(value int64, fromUnit, toUnit string) (v float64, u string, ok bool) {
	fromUnit = strings.TrimSuffix(strings.ToLower(fromUnit), "s")
	toUnit = strings.TrimSuffix(strings.ToLower(toUnit), "s")

	switch fromUnit {
	case "byte", "b":
	case "kilobyte", "kb":
		value *= 1024
	case "megabyte", "mb":
		value *= 1024 * 1024
	case "gigabyte", "gb":
		value *= 1024 * 1024 * 1024
	default:
		return 0, "", false
	}

	if toUnit == "minimum" || toUnit == "auto" {
		switch {
		case value < 1024:
			toUnit = "b"
		case value < 1024*1024:
			toUnit = "kb"
		case value < 1024*1024*1024:
			toUnit = "mb"
		default:
			toUnit = "gb"
		}
	}

	var output float64
	switch toUnit {
	default:
		output, toUnit = float64(value), "B"
	case "kb", "kbyte", "kilobyte":
		output, toUnit = float64(value)/1024, "kB"
	case "mb", "mbyte", "megabyte":
		output, toUnit = float64(value)/(1024*1024), "MB"
	case "gb", "gbyte", "gigabyte":
		output, toUnit = float64(value)/(1024*1024*1024), "GB"
	}
	return output, toUnit, true
}

func timeLabel(value int64, fromUnit, toUnit string) (v float64, u string, ok bool) {
	fromUnit = strings.ToLower(fromUnit)
	if len(fromUnit) > 2 {
		fromUnit = strings.TrimSuffix(fromUnit, "s")
	}

	toUnit = strings.ToLower(toUnit)
	if len(toUnit) > 2 {
		toUnit = strings.TrimSuffix(toUnit, "s")
	}

	var d time.Duration
	switch fromUnit {
	case "nanosecond", "ns":
		d = time.Duration(value) * time.Nanosecond
	case "microsecond":
		d = time.Duration(value) * time.Microsecond
	case "millisecond", "ms":
		d = time.Duration(value) * time.Millisecond
	case "second", "sec":
		d = time.Duration(value) * time.Second
	case "cycle":
		return float64(value), "", true
	default:
		return 0, "", false
	}

	if toUnit == "minimum" || toUnit == "auto" {
		switch {
		case d < 1*time.Microsecond:
			toUnit = "ns"
		case d < 1*time.Millisecond:
			toUnit = "us"
		case d < 1*time.Second:
			toUnit = "ms"
		case d < 1*time.Minute:
			toUnit = "sec"
		case d < 1*time.Hour:
			toUnit = "min"
		case d < 24*time.Hour:
			toUnit = "hour"
		case d < 15*24*time.Hour:
			toUnit = "day"
		case d < 120*24*time.Hour:
			toUnit = "week"
		default:
			toUnit = "year"
		}
	}

	var output float64
	dd := float64(d)
	switch toUnit {
	case "ns", "nanosecond":
		output, toUnit = dd/float64(time.Nanosecond), "ns"
	case "us", "microsecond":
		output, toUnit = dd/float64(time.Microsecond), "us"
	case "ms", "millisecond":
		output, toUnit = dd/float64(time.Millisecond), "ms"
	case "min", "minute":
		output, toUnit = dd/float64(time.Minute), "mins"
	case "hour", "hr":
		output, toUnit = dd/float64(time.Hour), "hrs"
	case "day":
		output, toUnit = dd/float64(24*time.Hour), "days"
	case "week", "wk":
		output, toUnit = dd/float64(7*24*time.Hour), "wks"
	case "year", "yr":
		output, toUnit = dd/float64(365*7*24*time.Hour), "yrs"
	default:
		fallthrough
	case "sec", "second", "s":
		output, toUnit = dd/float64(time.Second), "s"
	}
	return output, toUnit, true
}

// prettyName determines the printable name to be used for a node.
func (info *nodeInfo) prettyName() string {
	var name string
	if info.address != 0 {
		name = fmt.Sprintf("%016x", info.address)
	}

	if info.name != "" {
		name = name + " " + info.name
	}

	if info.file != "" {
		name += " " + trimPath(info.file)
		if info.lineno != 0 {
			name += fmt.Sprintf(":%d", info.lineno)
		}
	}

	if info.inline {
		name = name + " (inline)"
	}

	if name = strings.TrimSpace(name); name == "" && info.objfile != "" {
		name = "[" + filepath.Base(info.objfile) + "]"
	}
	return name
}

// New builds a new report indexing the sample values interpreting the
// samples with the provided function.
func New(prof *profile.Profile, options Options, value func(s *profile.Sample) int64, unit string) *Report {
	o := &options
	if o.SampleUnit == "" {
		o.SampleUnit = unit
	}
	format := func(v int64) string {
		if r := o.Ratio; r > 0 && r != 1 {
			fv := float64(v) * r
			v = int64(fv)
		}
		return scaledValueLabel(v, o.SampleUnit, o.OutputUnit)
	}
	return &Report{prof, computeTotal(prof, value), o, value, format}
}

// NewDefault builds a new report indexing the sample values with the
// last value available.
func NewDefault(prof *profile.Profile, options Options) *Report {
	index := len(prof.SampleType) - 1
	o := &options
	if o.SampleUnit == "" {
		o.SampleUnit = strings.ToLower(prof.SampleType[index].Unit)
	}
	value := func(s *profile.Sample) int64 {
		return s.Value[index]
	}
	format := func(v int64) string {
		if r := o.Ratio; r > 0 && r != 1 {
			fv := float64(v) * r
			v = int64(fv)
		}
		return scaledValueLabel(v, o.SampleUnit, o.OutputUnit)
	}
	return &Report{prof, computeTotal(prof, value), o, value, format}
}

func computeTotal(prof *profile.Profile, value func(s *profile.Sample) int64) int64 {
	var ret int64
	for _, sample := range prof.Sample {
		ret += value(sample)
	}
	return ret
}

// Report contains the data and associated routines to extract a
// report from a profile.
type Report struct {
	prof        *profile.Profile
	total       int64
	options     *Options
	sampleValue func(*profile.Sample) int64
	formatValue func(int64) string
}
