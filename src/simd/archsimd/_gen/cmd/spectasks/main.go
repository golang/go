// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Spectasks renders SPEC-TRANSITION.md's task dependencies as a Graphviz
// graph, and checks the document's copies of those dependencies against each
// other.
//
//	go run ./cmd/spectasks -w        # write SPEC-TASKS.dot
//	go run ./cmd/spectasks           # check only; write the graph to stdout
//
// SPEC-TRANSITION.md states its dependencies three times: once in Part 2's
// table, which this tool treats as the source of truth; once per task in
// Part 3's "Needs:"/"Prefers:" header lines; and a third time, reversed, in
// Part 3's "Blocks:" lines. go test ./cmd/spectasks fails if they disagree, and
// also fails if SPEC-TASKS.dot has not been regenerated since. Adding the graph
// as a fourth hand-maintained copy would have made that worse, so the graph is
// generated.
//
// Two things the graph shows that the table cannot. The critical path is
// computed, not asserted: a task is on it when delaying it delays the whole
// project, which is the set of tasks whose longest-chain-in plus
// longest-chain-out spans the graph. And milestones -- the points worth
// announcing -- are drawn as nodes, so it is visible which tasks feed them.
// Milestone state is derived from the task checkboxes and is never written
// down separately.
package main

import (
	"bytes"
	"cmp"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"simd/archsimd/_gen/gentools"
	"simd/archsimd/_gen/specgen"
)

// paths returns the plan document and the graph generated from it.
func paths(goroot string) (md, dot string) {
	specDir := specgen.MustFindSpecDir(goroot)
	genDir := filepath.Join(filepath.Clean(filepath.Join(specDir, "..", "..")), "archsimd", "_gen")
	return filepath.Join(genDir, "SPEC-TRANSITION.md"), filepath.Join(genDir, "SPEC-TASKS.dot")
}

// load parses the plan document.
func load(md string) *doc {
	data, err := os.ReadFile(md)
	if err != nil {
		log.Fatal(err)
	}
	return parse(string(data))
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("spectasks: ")
	genOpts := gentools.RegisterFlags(nil)
	flag.Parse()

	md, _ := paths(genOpts.GOROOT)
	d := load(md)

	// The test is where inconsistency is reported properly; this is only a
	// guard, so that a graph is never written from a document that disagrees
	// with itself.
	if probs := d.problems(); len(probs) > 0 {
		for _, s := range probs {
			fmt.Fprintf(os.Stderr, "spectasks: %s\n", s)
		}
		log.Fatalf("%s disagrees with itself; go test ./cmd/spectasks reports this properly", filepath.Base(md))
	}

	var files gentools.Files
	defer files.FlushOrExit()

	buf := files.NewRawFile("simd/archsimd/_gen/SPEC-TASKS.dot")
	d.dot(buf)
}

// A dep is one edge: the id of the task depended on.
type dep struct {
	id string
}

type task struct {
	id             string
	done           bool
	track          string
	title          string
	needs, prefers []dep

	critical bool // on some longest chain through the graph
}

type track struct {
	letter, name string
	tasks        []string
}

type milestone struct {
	name  string
	needs []dep
}

type doc struct {
	tasks      map[string]*task
	order      []string // task ids, in table order, which must be topological
	tracks     []track
	milestones []milestone

	// part4 is what Part 4's header lines claim, for cross-checking.
	part4 map[string]struct{ needs, prefers, blocks []dep }
}

// --- parsing ---------------------------------------------------------------

// A markdown table, as a header row and body rows of trimmed cells.
type mdTable struct {
	head []string
	rows [][]string
}

// cells splits one "| a | b |" line. Leading and trailing empties from the
// bounding pipes are dropped.
func cells(line string) []string {
	s := strings.TrimSpace(line)
	if !strings.HasPrefix(s, "|") || !strings.HasSuffix(s, "|") {
		return nil
	}
	parts := strings.Split(s[1:len(s)-1], "|")
	for i, p := range parts {
		parts[i] = strings.TrimSpace(p)
	}
	return parts
}

// findTable returns the first table in lines whose header row equals head.
// The header must match exactly: this is a format we control, and a silent
// mismatch would silently drop rows.
func findTable(lines []string, head ...string) *mdTable {
	for i, line := range lines {
		got := cells(line)
		if !slices.Equal(got, head) {
			continue
		}
		if i+1 >= len(lines) || !strings.HasPrefix(strings.TrimSpace(lines[i+1]), "|-") {
			log.Fatalf("table %q at line %d has no separator row", head, i+1)
		}
		t := &mdTable{head: got}
		for _, row := range lines[i+2:] {
			c := cells(row)
			if c == nil {
				break
			}
			if len(c) != len(head) {
				log.Fatalf("line %d: got %d cells, want %d", i+1, len(c), len(head))
			}
			t.rows = append(t.rows, c)
		}
		return t
	}
	log.Fatalf("no table with header %q", head)
	return nil
}

var (
	// A dependency list entry: a task id.
	depRE = regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`)
	// A Part 4 task heading: "### `spec-common` — Spec the common operations".
	headingRE = regexp.MustCompile("^### `([a-z0-9]+(?:-[a-z0-9]+)*)` — ")
	// A Part 4 header field: "**Needs:** 3, 5".
	fieldRE = regexp.MustCompile(`\*\*(Needs|Prefers|Blocks):\*\* ([^·]*)`)
	// Markdown emphasis and code spans, for extracting plain text.
	markupRE = regexp.MustCompile("[`*]")
)

// parseDeps parses a dependency cell: "—" for none, else a comma-separated
// list of dep entries.
func parseDeps(s string) []dep {
	s = strings.TrimSpace(markupRE.ReplaceAllString(s, ""))
	if s == "" || s == "—" || s == "-" {
		return nil
	}
	var out []dep
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if !depRE.MatchString(f) {
			log.Fatalf("cannot parse dependency %q in %q; want a task id", f, s)
		}
		out = append(out, dep{f})
	}
	// Sorted so that the three copies of the edge set compare equal
	// regardless of the order each happens to list them in.
	slices.SortFunc(out, func(a, b dep) int { return cmp.Compare(a.id, b.id) })
	return out
}

func parse(text string) *doc {
	lines := strings.Split(text, "\n")
	d := &doc{
		tasks: make(map[string]*task),
		part4: make(map[string]struct{ needs, prefers, blocks []dep }),
	}

	for _, r := range findTable(lines, "", "Track", "What it is", "Tasks").rows {
		var tasks []string
		for _, dep := range parseDeps(r[3]) {
			tasks = append(tasks, dep.id)
		}
		d.tracks = append(d.tracks, track{
			letter: markupRE.ReplaceAllString(r[0], ""),
			name:   r[1],
			tasks:  tasks,
		})
	}

	for _, r := range findTable(lines, "✓", "T", "Task", "What it is", "Needs", "Prefers").rows {
		id := markupRE.ReplaceAllString(r[2], "")
		if !depRE.MatchString(id) {
			log.Fatalf("bad task id %q; want lowercase words joined by hyphens", r[2])
		}
		if d.tasks[id] != nil {
			log.Fatalf("task %q listed twice", id)
		}
		d.tasks[id] = &task{
			id:      id,
			done:    strings.Contains(r[0], "x"),
			track:   r[1],
			title:   markupRE.ReplaceAllString(r[3], ""),
			needs:   parseDeps(r[4]),
			prefers: parseDeps(r[5]),
		}
		d.order = append(d.order, id)
	}

	for _, r := range findTable(lines, "Milestone", "Reached after", "What it means").rows {
		d.milestones = append(d.milestones, milestone{
			name:  markupRE.ReplaceAllString(r[0], ""),
			needs: parseDeps(r[1]),
		})
	}

	// Part 4 header lines, for cross-checking.
	cur := ""
	for _, line := range lines {
		if m := headingRE.FindStringSubmatch(line); m != nil {
			cur = m[1]
			continue
		}
		if cur == "" || !strings.HasPrefix(line, "**Done:** [") {
			continue
		}
		var e struct{ needs, prefers, blocks []dep }
		for _, m := range fieldRE.FindAllStringSubmatch(line, -1) {
			switch m[1] {
			case "Needs":
				e.needs = parseDeps(m[2])
			case "Prefers":
				e.prefers = parseDeps(m[2])
			case "Blocks":
				e.blocks = parseDeps(m[2])
			}
		}
		d.part4[cur] = e
		cur = ""
	}
	return d
}

// --- checking --------------------------------------------------------------

func ids(ds []dep) []string {
	out := make([]string, len(ds))
	for i, d := range ds {
		out[i] = d.id
	}
	return out
}

func fmtIDs(xs []string) string {
	if len(xs) == 0 {
		return "—"
	}
	return strings.Join(xs, ", ")
}

// problems returns everything inconsistent about the document, as one message
// per problem, in a stable order. It reports rather than fails, so that the
// test can attribute each one and the tool can use it as a guard.
func (d *doc) problems() []string {
	var probs []string
	fail := func(format string, args ...any) {
		probs = append(probs, fmt.Sprintf(format, args...))
	}

	// Every task belongs to exactly one track, and the two lists agree.
	seen := make(map[string]string)
	for _, tr := range d.tracks {
		for _, id := range tr.tasks {
			if prev, ok := seen[id]; ok {
				fail("task %s is in tracks %s and %s", id, prev, tr.letter)
			}
			seen[id] = tr.letter
			if t := d.tasks[id]; t == nil {
				fail("track %s lists task %s, which the dependency table does not have", tr.letter, id)
			} else if t.track != tr.letter {
				fail("task %s is in track %s per the track table, %s per the dependency table", id, tr.letter, t.track)
			}
		}
	}

	// The table is in topological order, so that reading it top to bottom is
	// a legal order to do the work in. This is what task ids buy: identity no
	// longer encodes position, so rows can be moved and inserted freely.
	pos := make(map[string]int, len(d.order))
	for i, id := range d.order {
		pos[id] = i
	}

	// Edges point at real tasks; Part 4 agrees with the table; Blocks is the
	// exact reverse of Needs.
	blocks := make(map[string][]string)
	for _, id := range d.order {
		t := d.tasks[id]
		if seen[id] == "" {
			fail("task %s is in no track", id)
		}
		for _, e := range slices.Concat(t.needs, t.prefers) {
			if d.tasks[e.id] == nil {
				fail("task %s depends on task %s, which does not exist", id, e.id)
			}
		}
		for _, e := range t.needs {
			if d.tasks[e.id] != nil && pos[e.id] > pos[id] {
				fail("task %s needs %s but comes before it; the table must be in topological order", id, e.id)
			}
			blocks[e.id] = append(blocks[e.id], id)
		}
		for _, e := range t.prefers {
			if d.tasks[e.id] != nil && pos[e.id] > pos[id] {
				fail("task %s prefers %s but comes before it; the table must be in topological order", id, e.id)
			}
		}
		p4, ok := d.part4[id]
		if !ok {
			fail("task %s has no Part 4 entry", id)
			continue
		}
		if !slices.Equal(ids(t.needs), ids(p4.needs)) {
			fail("task %s Needs: table says %s, Part 4 says %s", id, fmtIDs(ids(t.needs)), fmtIDs(ids(p4.needs)))
		}
		if !slices.Equal(ids(t.prefers), ids(p4.prefers)) {
			fail("task %s Prefers: table says %s, Part 4 says %s", id, fmtIDs(ids(t.prefers)), fmtIDs(ids(p4.prefers)))
		}
	}
	for _, id := range d.order {
		want := blocks[id]
		slices.Sort(want)
		if got := ids(d.part4[id].blocks); !slices.Equal(got, want) {
			fail("task %s Blocks: header says %s, reversing Needs gives %s", id, fmtIDs(got), fmtIDs(want))
		}
	}

	// Unrelated tasks sort by track: if task A (track X) precedes task B
	// (track Y) and X > Y, then B must transitively depend on A through
	// Needs or Prefers edges.
	tdeps := make(map[string]map[string]bool, len(d.order))
	var computeDeps func(string) map[string]bool
	computeDeps = func(id string) map[string]bool {
		if r, ok := tdeps[id]; ok {
			return r
		}
		r := make(map[string]bool)
		tdeps[id] = r
		for _, e := range slices.Concat(d.tasks[id].needs, d.tasks[id].prefers) {
			if d.tasks[e.id] != nil {
				r[e.id] = true
				for dep := range computeDeps(e.id) {
					r[dep] = true
				}
			}
		}
		return r
	}
	for _, id := range d.order {
		computeDeps(id)
	}
	for i, id := range d.order {
		for _, id2 := range d.order[i+1:] {
			if d.tasks[id].track > d.tasks[id2].track && !tdeps[id2][id] {
				fail("task %s (track %s) precedes %s (track %s) but %s does not depend on %s; unrelated tasks sort by track",
					id, d.tasks[id].track, id2, d.tasks[id2].track, id2, id)
			}
		}
	}

	for _, m := range d.milestones {
		for _, e := range m.needs {
			if d.tasks[e.id] == nil {
				fail("milestone %q depends on task %s, which does not exist", m.name, e.id)
			}
		}
	}

	return probs
}

// mark sets task.critical on every task that lies on some longest chain of
// hard dependencies: delaying it delays the whole project. Ties are kept --
// there is usually more than one longest chain, and picking one of them
// arbitrarily is how a hand-written "critical path" understates the work.
func (d *doc) mark() {
	var into, outOf func(string) int
	memo := func(f func(string) int) func(string) int {
		m := make(map[string]int)
		return func(id string) int {
			if v, ok := m[id]; ok {
				return v
			}
			m[id] = 1 // cycles fall back to 1 rather than recursing forever
			v := f(id)
			m[id] = v
			return v
		}
	}
	// Longest chain ending at id, counted in tasks.
	into = memo(func(id string) int {
		best := 0
		for _, e := range d.tasks[id].needs {
			if d.tasks[e.id] != nil {
				best = max(best, into(e.id))
			}
		}
		return best + 1
	})
	// Longest chain starting at id.
	outOf = memo(func(id string) int {
		best := 0
		for _, other := range d.order {
			for _, e := range d.tasks[other].needs {
				if e.id == id {
					best = max(best, outOf(other))
				}
			}
		}
		return best + 1
	})

	span := 0
	for _, id := range d.order {
		span = max(span, into(id))
	}
	for _, id := range d.order {
		d.tasks[id].critical = into(id)+outOf(id)-1 == span
	}
}

// reached reports whether every task feeding m is ticked.
func (d *doc) reached(m milestone) bool {
	for _, e := range m.needs {
		if !d.tasks[e.id].done {
			return false
		}
	}
	return len(m.needs) > 0
}

// --- rendering -------------------------------------------------------------

// Node fills, in track order. Light enough to read black text on.
var trackFill = []string{
	"#dce9f7", "#d9f2e2", "#fbe6cd", "#ebdcf7", "#f7dce3", "#d8f0f0", "#e8e8e8",
}

// wrap breaks a title into Graphviz label lines of at most width runes,
// without breaking words.
func wrap(s string, width int) string {
	var lines []string
	line := ""
	for _, w := range strings.Fields(s) {
		switch {
		case line == "":
			line = w
		case len([]rune(line))+1+len([]rune(w)) <= width:
			line += " " + w
		default:
			lines = append(lines, line)
			line = w
		}
	}
	return strings.Join(append(lines, line), `\n`)
}

// node is a task id as a Graphviz node name. Ids contain hyphens, which are
// not legal in a bare Graphviz ID, so they are always quoted.
func node(id string) string { return quote(id) }

func quote(s string) string {
	return `"` + strings.NewReplacer(`"`, `\"`, "\n", `\n`).Replace(s) + `"`
}

func (d *doc) dot(w *bytes.Buffer) {
	d.mark()
	p := func(format string, args ...any) { fmt.Fprintf(w, format+"\n", args...) }

	p(`// Task graph for SPEC-TRANSITION.md. Do not edit.`)
	p(`// Generated by archsimd/_gen/cmd/spectasks; regenerate with:`)
	p(`//     go run ./cmd/spectasks -w`)
	p(`// Render with, e.g.:`)
	p(`//     dot -Tsvg SPEC-TASKS.dot -o SPEC-TASKS.svg`)
	p(`//`)
	p(`// The key in the rendered graph explains the notation.`)
	p(``)
	p(`digraph spectransition {`)
	p(`  rankdir=LR;`)
	p(`  newrank=true;`)
	p(`  ranksep=0.55; nodesep=0.22;`)
	p(`  fontname="Helvetica"; fontsize=11;`)
	p(`  node [shape=box, style="rounded,filled", fillcolor=white,`)
	p(`        fontname="Helvetica", fontsize=10, margin="0.10,0.06"];`)
	p(`  edge [color="#666666", arrowsize=0.7];`)
	p(``)

	// Tracks are shown as node colour, not as clusters: a cluster has to be
	// laid out contiguously, which fights the topological ranking and drags
	// long edges across the whole graph. Colour costs a legend and keeps the
	// left-to-right reading of dependency order.
	emit := func(t *task, fill string) {
		attrs := []string{
			"label=" + quote(t.id+`\n`+wrap(t.title, 22)),
			"fillcolor=" + quote(fill),
			`color="#999999"`,
		}
		if t.done {
			attrs = append(attrs, `fillcolor="#d0d0d0"`, `fontcolor="#777777"`)
		}
		p(`  %s [%s];`, node(t.id), strings.Join(attrs, ", "))
	}
	// Emitted track by track: declaration order is one of the things dot breaks
	// ties on, and grouping by track lays out better than topological order
	// does. Any task no track claims is emitted afterwards rather than dropped,
	// so a document that problems() would reject still renders.
	seen := make(map[string]bool, len(d.order))
	for i, tr := range d.tracks {
		for _, id := range tr.tasks {
			if t := d.tasks[id]; t != nil && !seen[id] {
				seen[id] = true
				emit(t, trackFill[i%len(trackFill)])
			}
		}
	}
	for _, id := range d.order {
		if !seen[id] {
			emit(d.tasks[id], "white")
		}
	}

	p(``)

	for _, id := range d.order {
		t := d.tasks[id]
		for _, e := range t.needs {
			var attrs []string
			if t.critical && d.tasks[e.id].critical {
				attrs = append(attrs, `color="#c0392b"`, `penwidth=1.8`)
			}
			p(`  %s -> %s%s;`, node(e.id), node(id), attrList(attrs))
		}
		for _, e := range t.prefers {
			p(`  %s -> %s [style=dotted, color="#aaaaaa", arrowsize=0.5];`, node(e.id), node(id))
		}
	}
	p(``)

	for i, m := range d.milestones {
		fill := "#fff8dc"
		if d.reached(m) {
			fill = "#d7d7d7"
		}
		p(`  m%d [label=%s, shape=box, style="filled,bold", fillcolor=%s, color="#8a6d3b"];`,
			i, quote(wrap(m.name, 22)), quote(fill))
		for _, e := range m.needs {
			p(`  %s -> m%d [color="#8a6d3b", style=bold, arrowsize=0.7];`, node(e.id), i)
		}
	}

	// The key is a single node with an HTML label rather than a cluster of
	// nodes: a cluster chained together spans as many ranks as it has entries,
	// and dot then aligns those ranks with the graph proper, which distorts the
	// layout it is supposed to annotate.
	p(``)
	p(`  key [shape=plaintext, style="", label=<`)
	p(`    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="3" BGCOLOR="#fafafa">`)
	p(`    <TR><TD ALIGN="LEFT" COLSPAN="2"><B>Tracks</B></TD></TR>`)
	for i, tr := range d.tracks {
		p(`    <TR><TD BGCOLOR="%s" WIDTH="18"> </TD><TD ALIGN="LEFT">%s</TD></TR>`,
			trackFill[i%len(trackFill)], html(tr.letter+" — "+tr.name))
	}
	p(`    <TR><TD COLSPAN="2"><FONT POINT-SIZE="5"> </FONT></TD></TR>`)
	p(`    <TR><TD ALIGN="LEFT" COLSPAN="2"><B>Reading the graph</B></TD></TR>`)
	// Each row shows the thing it names, rather than describing it: an arrow in
	// the edge's own colour, a swatch in the node's own fill.
	arrow := func(color, glyph string) string {
		return fmt.Sprintf(`<FONT COLOR="%s" POINT-SIZE="14">%s</FONT>`, color, glyph)
	}
	swatch := func(fill, border string) string {
		return fmt.Sprintf(`<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" `+
			`COLOR="%s" BGCOLOR="%s"><TR><TD WIDTH="16" HEIGHT="9"> </TD></TR></TABLE>`, border, fill)
	}
	for _, row := range [][2]string{
		{arrow("#666666", "&#8594;"), "must be done first"},
		{arrow("#aaaaaa", "&#8674;"), "better done first, but not required"},
		{arrow("#c0392b", "&#8594;"), "on a longest chain: slipping it slips the project"},
		{swatch("#fff8dc", "#8a6d3b"), "milestone: reached, not worked on"},
		{swatch("#d0d0d0", "#999999"), "done"},
	} {
		p(`    <TR><TD ALIGN="RIGHT">%s</TD><TD ALIGN="LEFT">%s</TD></TR>`, row[0], html(row[1]))
	}
	p(`    </TABLE>`)
	p(`  >];`)
	p(`}`)
}

// html escapes text for a Graphviz HTML-like label.
func html(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;").Replace(s)
}

func attrList(attrs []string) string {
	if len(attrs) == 0 {
		return ""
	}
	return " [" + strings.Join(attrs, ", ") + "]"
}
