// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package main // import "golang.org/x/tools/cmd/digraph"

// TODO(adonovan):
// - support input files other than stdin
// - support alternative formats (AT&T GraphViz, CSV, etc),
//   a comment syntax, etc.
// - allow queries to nest, like Blaze query language.

import (
	"bufio"
	"bytes"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

func usage() {
	// Extract the content of the /* ... */ comment in doc.go.
	_, after, _ := strings.Cut(doc, "/*")
	doc, _, _ := strings.Cut(after, "*/")
	io.WriteString(flag.CommandLine.Output(), doc)
	flag.PrintDefaults()

	os.Exit(2)
}

//go:embed doc.go
var doc string

func main() {
	flag.Usage = usage
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		usage()
	}

	if err := digraph(args[0], args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "digraph: %s\n", err)
		os.Exit(1)
	}
}

type nodelist []string

func (l nodelist) println(sep string) {
	for i, node := range l {
		if i > 0 {
			fmt.Fprint(stdout, sep)
		}
		fmt.Fprint(stdout, node)
	}
	fmt.Fprintln(stdout)
}

type nodeset map[string]bool

func (s nodeset) sort() nodelist {
	nodes := make(nodelist, len(s))
	var i int
	for node := range s {
		nodes[i] = node
		i++
	}
	sort.Strings(nodes)
	return nodes
}

func (s nodeset) addAll(x nodeset) {
	for node := range x {
		s[node] = true
	}
}

// A graph maps nodes to the non-nil set of their immediate successors.
type graph map[string]nodeset

func (g graph) addNode(node string) nodeset {
	edges := g[node]
	if edges == nil {
		edges = make(nodeset)
		g[node] = edges
	}
	return edges
}

func (g graph) addEdges(from string, to ...string) {
	edges := g.addNode(from)
	for _, to := range to {
		g.addNode(to)
		edges[to] = true
	}
}

func (g graph) nodelist() nodelist {
	nodes := make(nodeset)
	for node := range g {
		nodes[node] = true
	}
	return nodes.sort()
}

func (g graph) reachableFrom(roots nodeset) nodeset {
	seen := make(nodeset)
	var visit func(node string)
	visit = func(node string) {
		if !seen[node] {
			seen[node] = true
			for e := range g[node] {
				visit(e)
			}
		}
	}
	for root := range roots {
		visit(root)
	}
	return seen
}

func (g graph) transpose() graph {
	rev := make(graph)
	for node, edges := range g {
		rev.addNode(node)
		for succ := range edges {
			rev.addEdges(succ, node)
		}
	}
	return rev
}

func (g graph) sccs() []nodeset {
	// Kosaraju's algorithm---Tarjan is overkill here.

	// Forward pass.
	S := make(nodelist, 0, len(g)) // postorder stack
	seen := make(nodeset)
	var visit func(node string)
	visit = func(node string) {
		if !seen[node] {
			seen[node] = true
			for e := range g[node] {
				visit(e)
			}
			S = append(S, node)
		}
	}
	for node := range g {
		visit(node)
	}

	// Reverse pass.
	rev := g.transpose()
	var scc nodeset
	seen = make(nodeset)
	var rvisit func(node string)
	rvisit = func(node string) {
		if !seen[node] {
			seen[node] = true
			scc[node] = true
			for e := range rev[node] {
				rvisit(e)
			}
		}
	}
	var sccs []nodeset
	for len(S) > 0 {
		top := S[len(S)-1]
		S = S[:len(S)-1] // pop
		if !seen[top] {
			scc = make(nodeset)
			rvisit(top)
			if len(scc) == 1 && !g[top][top] {
				continue
			}
			sccs = append(sccs, scc)
		}
	}
	return sccs
}

func (g graph) allpaths(from, to string) error {
	// Mark all nodes to "to".
	seen := make(nodeset) // value of seen[x] indicates whether x is on some path to "to"
	var visit func(node string) bool
	visit = func(node string) bool {
		reachesTo, ok := seen[node]
		if !ok {
			reachesTo = node == to
			seen[node] = reachesTo
			for e := range g[node] {
				if visit(e) {
					reachesTo = true
				}
			}
			if reachesTo && node != to {
				seen[node] = true
			}
		}
		return reachesTo
	}
	visit(from)

	// For each marked node, collect its marked successors.
	var edges []string
	for n := range seen {
		for succ := range g[n] {
			if seen[succ] {
				edges = append(edges, n+" "+succ)
			}
		}
	}

	// Sort (so that this method is deterministic) and print edges.
	sort.Strings(edges)
	for _, e := range edges {
		fmt.Fprintln(stdout, e)
	}

	return nil
}

func (g graph) somepath(from, to string) error {
	// Search breadth-first so that we return a minimal path.

	// A path is a linked list whose head is a candidate "to" node
	// and whose tail is the path ending in the "from" node.
	type path struct {
		node string
		tail *path
	}

	seen := nodeset{from: true}

	var queue []*path
	queue = append(queue, &path{node: from, tail: nil})
	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]

		if p.node == to {
			// Found a path. Print, tail first.
			var print func(p *path)
			print = func(p *path) {
				if p.tail != nil {
					print(p.tail)
					fmt.Fprintln(stdout, p.tail.node+" "+p.node)
				}
			}
			print(p)
			return nil
		}

		for succ := range g[p.node] {
			if !seen[succ] {
				seen[succ] = true
				queue = append(queue, &path{node: succ, tail: p})
			}
		}
	}
	return fmt.Errorf("no path from %q to %q", from, to)
}

func (g graph) toDot(w *bytes.Buffer) {
	fmt.Fprintln(w, "digraph {")
	for _, src := range g.nodelist() {
		for _, dst := range g[src].sort() {
			// Dot's quoting rules appear to align with Go's for escString,
			// which is the syntax of node IDs. Labels require significantly
			// more quoting, but that appears not to be necessary if the node ID
			// is implicitly used as the label.
			fmt.Fprintf(w, "\t%q -> %q;\n", src, dst)
		}
	}
	fmt.Fprintln(w, "}")
}

func parse(rd io.Reader) (graph, error) {
	g := make(graph)

	var linenum int
	// We avoid bufio.Scanner as it imposes a (configurable) limit
	// on line length, whereas Reader.ReadString does not.
	in := bufio.NewReader(rd)
	for {
		linenum++
		line, err := in.ReadString('\n')
		eof := false
		if err == io.EOF {
			eof = true
		} else if err != nil {
			return nil, err
		}
		// Split into words, honoring double-quotes per Go spec.
		words, err := split(line)
		if err != nil {
			return nil, fmt.Errorf("at line %d: %v", linenum, err)
		}
		if len(words) > 0 {
			g.addEdges(words[0], words[1:]...)
		}
		if eof {
			break
		}
	}
	return g, nil
}

// Overridable for redirection.
var stdin io.Reader = os.Stdin
var stdout io.Writer = os.Stdout

func digraph(cmd string, args []string) error {
	// Parse the input graph.
	g, err := parse(stdin)
	if err != nil {
		return err
	}

	// Parse the command line.
	switch cmd {
	case "nodes":
		if len(args) != 0 {
			return fmt.Errorf("usage: digraph nodes")
		}
		g.nodelist().println("\n")

	case "degree":
		if len(args) != 0 {
			return fmt.Errorf("usage: digraph degree")
		}
		nodes := make(nodeset)
		for node := range g {
			nodes[node] = true
		}
		rev := g.transpose()
		for _, node := range nodes.sort() {
			fmt.Fprintf(stdout, "%d\t%d\t%s\n", len(rev[node]), len(g[node]), node)
		}

	case "transpose":
		if len(args) != 0 {
			return fmt.Errorf("usage: digraph transpose")
		}
		var revEdges []string
		for node, succs := range g.transpose() {
			for succ := range succs {
				revEdges = append(revEdges, fmt.Sprintf("%s %s", node, succ))
			}
		}
		sort.Strings(revEdges) // make output deterministic
		for _, e := range revEdges {
			fmt.Fprintln(stdout, e)
		}

	case "succs", "preds":
		if len(args) == 0 {
			return fmt.Errorf("usage: digraph %s <node> ... ", cmd)
		}
		g := g
		if cmd == "preds" {
			g = g.transpose()
		}
		result := make(nodeset)
		for _, root := range args {
			edges := g[root]
			if edges == nil {
				return fmt.Errorf("no such node %q", root)
			}
			result.addAll(edges)
		}
		result.sort().println("\n")

	case "forward", "reverse":
		if len(args) == 0 {
			return fmt.Errorf("usage: digraph %s <node> ... ", cmd)
		}
		roots := make(nodeset)
		for _, root := range args {
			if g[root] == nil {
				return fmt.Errorf("no such node %q", root)
			}
			roots[root] = true
		}
		g := g
		if cmd == "reverse" {
			g = g.transpose()
		}
		g.reachableFrom(roots).sort().println("\n")

	case "somepath":
		if len(args) != 2 {
			return fmt.Errorf("usage: digraph somepath <from> <to>")
		}
		from, to := args[0], args[1]
		if g[from] == nil {
			return fmt.Errorf("no such 'from' node %q", from)
		}
		if g[to] == nil {
			return fmt.Errorf("no such 'to' node %q", to)
		}
		if err := g.somepath(from, to); err != nil {
			return err
		}

	case "allpaths":
		if len(args) != 2 {
			return fmt.Errorf("usage: digraph allpaths <from> <to>")
		}
		from, to := args[0], args[1]
		if g[from] == nil {
			return fmt.Errorf("no such 'from' node %q", from)
		}
		if g[to] == nil {
			return fmt.Errorf("no such 'to' node %q", to)
		}
		if err := g.allpaths(from, to); err != nil {
			return err
		}

	case "sccs":
		if len(args) != 0 {
			return fmt.Errorf("usage: digraph sccs")
		}
		buf := new(bytes.Buffer)
		oldStdout := stdout
		stdout = buf
		for _, scc := range g.sccs() {
			scc.sort().println(" ")
		}
		lines := strings.SplitAfter(buf.String(), "\n")
		sort.Strings(lines)
		stdout = oldStdout
		io.WriteString(stdout, strings.Join(lines, ""))

	case "scc":
		if len(args) != 1 {
			return fmt.Errorf("usage: digraph scc <node>")
		}
		node := args[0]
		if g[node] == nil {
			return fmt.Errorf("no such node %q", node)
		}
		for _, scc := range g.sccs() {
			if scc[node] {
				scc.sort().println("\n")
				break
			}
		}

	case "focus":
		if len(args) != 1 {
			return fmt.Errorf("usage: digraph focus <node>")
		}
		node := args[0]
		if g[node] == nil {
			return fmt.Errorf("no such node %q", node)
		}

		edges := make(map[string]struct{})
		for from := range g.reachableFrom(nodeset{node: true}) {
			for to := range g[from] {
				edges[fmt.Sprintf("%s %s", from, to)] = struct{}{}
			}
		}

		gtrans := g.transpose()
		for from := range gtrans.reachableFrom(nodeset{node: true}) {
			for to := range gtrans[from] {
				edges[fmt.Sprintf("%s %s", to, from)] = struct{}{}
			}
		}

		edgesSorted := make([]string, 0, len(edges))
		for e := range edges {
			edgesSorted = append(edgesSorted, e)
		}
		sort.Strings(edgesSorted)
		fmt.Fprintln(stdout, strings.Join(edgesSorted, "\n"))

	case "to":
		if len(args) != 1 || args[0] != "dot" {
			return fmt.Errorf("usage: digraph to dot")
		}
		var b bytes.Buffer
		g.toDot(&b)
		stdout.Write(b.Bytes())

	default:
		return fmt.Errorf("no such command %q", cmd)
	}

	return nil
}

// -- Utilities --------------------------------------------------------

// split splits a line into words, which are generally separated by
// spaces, but Go-style double-quoted string literals are also supported.
// (This approximates the behaviour of the Bourne shell.)
//
//	`one "two three"` -> ["one" "two three"]
//	`a"\n"b` -> ["a\nb"]
func split(line string) ([]string, error) {
	var (
		words   []string
		inWord  bool
		current bytes.Buffer
	)

	for len(line) > 0 {
		r, size := utf8.DecodeRuneInString(line)
		if unicode.IsSpace(r) {
			if inWord {
				words = append(words, current.String())
				current.Reset()
				inWord = false
			}
		} else if r == '"' {
			var ok bool
			size, ok = quotedLength(line)
			if !ok {
				return nil, errors.New("invalid quotation")
			}
			s, err := strconv.Unquote(line[:size])
			if err != nil {
				return nil, err
			}
			current.WriteString(s)
			inWord = true
		} else {
			current.WriteRune(r)
			inWord = true
		}
		line = line[size:]
	}
	if inWord {
		words = append(words, current.String())
	}
	return words, nil
}

// quotedLength returns the length in bytes of the prefix of input that
// contain a possibly-valid double-quoted Go string literal.
//
// On success, n is at least two (""); input[:n] may be passed to
// strconv.Unquote to interpret its value, and input[n:] contains the
// rest of the input.
//
// On failure, quotedLength returns false, and the entire input can be
// passed to strconv.Unquote if an informative error message is desired.
//
// quotedLength does not and need not detect all errors, such as
// invalid hex or octal escape sequences, since it assumes
// strconv.Unquote will be applied to the prefix.  It guarantees only
// that if there is a prefix of input containing a valid string literal,
// its length is returned.
//
// TODO(adonovan): move this into a strconv-like utility package.
func quotedLength(input string) (n int, ok bool) {
	var offset int

	// next returns the rune at offset, or -1 on EOF.
	// offset advances to just after that rune.
	next := func() rune {
		if offset < len(input) {
			r, size := utf8.DecodeRuneInString(input[offset:])
			offset += size
			return r
		}
		return -1
	}

	if next() != '"' {
		return // error: not a quotation
	}

	for {
		r := next()
		if r == '\n' || r < 0 {
			return // error: string literal not terminated
		}
		if r == '"' {
			return offset, true // success
		}
		if r == '\\' {
			var skip int
			switch next() {
			case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', '"':
				skip = 0
			case '0', '1', '2', '3', '4', '5', '6', '7':
				skip = 2
			case 'x':
				skip = 2
			case 'u':
				skip = 4
			case 'U':
				skip = 8
			default:
				return // error: invalid escape
			}

			for i := 0; i < skip; i++ {
				next()
			}
		}
	}
}
