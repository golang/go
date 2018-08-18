// The digraph command performs queries over unlabelled directed graphs
// represented in text form.  It is intended to integrate nicely with
// typical UNIX command pipelines.
//
// Since directed graphs (import graphs, reference graphs, call graphs,
// etc) often arise during software tool development and debugging, this
// command is included in the go.tools repository.
//
// TODO(adonovan):
// - support input files other than stdin
// - support alternative formats (AT&T GraphViz, CSV, etc),
//   a comment syntax, etc.
// - allow queries to nest, like Blaze query language.
//
package main // import "golang.org/x/tools/cmd/digraph"

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"unicode"
	"unicode/utf8"
)

const Usage = `digraph: queries over directed graphs in text form.

Graph format:

  Each line contains zero or more words.  Words are separated by
  unquoted whitespace; words may contain Go-style double-quoted portions,
  allowing spaces and other characters to be expressed.

  Each field declares a node, and if there are more than one,
  an edge from the first to each subsequent one.
  The graph is provided on the standard input.

  For instance, the following (acyclic) graph specifies a partial order
  among the subtasks of getting dressed:

	% cat clothes.txt
	socks shoes
	"boxer shorts" pants
	pants belt shoes
	shirt tie sweater
	sweater jacket
	hat

  The line "shirt tie sweater" indicates the two edges shirt -> tie and
  shirt -> sweater, not shirt -> tie -> sweater.

Supported queries:

  nodes
	the set of all nodes
  degree
	the in-degree and out-degree of each node.
  preds <label> ...
	the set of immediate predecessors of the specified nodes
  succs <label> ...
	the set of immediate successors of the specified nodes
  forward <label> ...
	the set of nodes transitively reachable from the specified nodes
  reverse <label> ...
	the set of nodes that transitively reach the specified nodes
  somepath <label> <label>
	the list of nodes on some arbitrary path from the first node to the second
  allpaths <label> <label>
	the set of nodes on all paths from the first node to the second
  sccs
	all strongly connected components (one per line)
  scc <label>
	the set of nodes nodes strongly connected to the specified one

Example usage:

   Show the transitive closure of imports of the digraph tool itself:
   % go list -f '{{.ImportPath}}{{.Imports}}' ... | tr '[]' '  ' |
         digraph forward golang.org/x/tools/cmd/digraph

   Show which clothes (see above) must be donned before a jacket:
   %  digraph reverse jacket <clothes.txt

`

func main() {
	flag.Usage = func() { fmt.Fprintln(os.Stderr, Usage) }
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, Usage)
		return
	}

	if err := digraph(args[0], args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "digraph: %s\n", err)
		os.Exit(1)
	}
}

type nodelist []string

func (l nodelist) println(sep string) {
	for i, label := range l {
		if i > 0 {
			fmt.Fprint(stdout, sep)
		}
		fmt.Fprint(stdout, label)
	}
	fmt.Fprintln(stdout)
}

type nodeset map[string]bool

func (s nodeset) sort() nodelist {
	labels := make(nodelist, len(s))
	var i int
	for label := range s {
		labels[i] = label
		i++
	}
	sort.Strings(labels)
	return labels
}

func (s nodeset) addAll(x nodeset) {
	for label := range x {
		s[label] = true
	}
}

// A graph maps nodes to the non-nil set of their immediate successors.
type graph map[string]nodeset

func (g graph) addNode(label string) nodeset {
	edges := g[label]
	if edges == nil {
		edges = make(nodeset)
		g[label] = edges
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

func (g graph) reachableFrom(roots nodeset) nodeset {
	seen := make(nodeset)
	var visit func(label string)
	visit = func(label string) {
		if !seen[label] {
			seen[label] = true
			for e := range g[label] {
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
	for label, edges := range g {
		rev.addNode(label)
		for succ := range edges {
			rev.addEdges(succ, label)
		}
	}
	return rev
}

func (g graph) sccs() []nodeset {
	// Kosaraju's algorithm---Tarjan is overkill here.

	// Forward pass.
	S := make(nodelist, 0, len(g)) // postorder stack
	seen := make(nodeset)
	var visit func(label string)
	visit = func(label string) {
		if !seen[label] {
			seen[label] = true
			for e := range g[label] {
				visit(e)
			}
			S = append(S, label)
		}
	}
	for label := range g {
		visit(label)
	}

	// Reverse pass.
	rev := g.transpose()
	var scc nodeset
	seen = make(nodeset)
	var rvisit func(label string)
	rvisit = func(label string) {
		if !seen[label] {
			seen[label] = true
			scc[label] = true
			for e := range rev[label] {
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
			sccs = append(sccs, scc)
		}
	}
	return sccs
}

func parse(rd io.Reader) (graph, error) {
	g := make(graph)

	var linenum int
	in := bufio.NewScanner(rd)
	for in.Scan() {
		linenum++
		// Split into words, honoring double-quotes per Go spec.
		words, err := split(in.Text())
		if err != nil {
			return nil, fmt.Errorf("at line %d: %v", linenum, err)
		}
		if len(words) > 0 {
			g.addEdges(words[0], words[1:]...)
		}
	}
	if err := in.Err(); err != nil {
		return nil, err
	}
	return g, nil
}

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
		nodes := make(nodeset)
		for label := range g {
			nodes[label] = true
		}
		nodes.sort().println("\n")

	case "degree":
		if len(args) != 0 {
			return fmt.Errorf("usage: digraph degree")
		}
		nodes := make(nodeset)
		for label := range g {
			nodes[label] = true
		}
		rev := g.transpose()
		for _, label := range nodes.sort() {
			fmt.Fprintf(stdout, "%d\t%d\t%s\n", len(rev[label]), len(g[label]), label)
		}

	case "succs", "preds":
		if len(args) == 0 {
			return fmt.Errorf("usage: digraph %s <label> ...", cmd)
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
			return fmt.Errorf("usage: digraph %s <label> ...", cmd)
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

		seen := make(nodeset)
		var visit func(path nodelist, label string) bool
		visit = func(path nodelist, label string) bool {
			if !seen[label] {
				seen[label] = true
				if label == to {
					append(path, label).println("\n")
					return true // unwind
				}
				for e := range g[label] {
					if visit(append(path, label), e) {
						return true
					}
				}
			}
			return false
		}
		if !visit(make(nodelist, 0, 100), from) {
			return fmt.Errorf("no path from %q to %q", args[0], args[1])
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

		seen := make(nodeset) // value of seen[x] indicates whether x is on some path to 'to'
		var visit func(label string) bool
		visit = func(label string) bool {
			reachesTo, ok := seen[label]
			if !ok {
				reachesTo = label == to

				seen[label] = reachesTo
				for e := range g[label] {
					if visit(e) {
						reachesTo = true
					}
				}
				seen[label] = reachesTo
			}
			return reachesTo
		}
		if !visit(from) {
			return fmt.Errorf("no path from %q to %q", from, to)
		}
		for label, reachesTo := range seen {
			if !reachesTo {
				delete(seen, label)
			}
		}
		seen.sort().println("\n")

	case "sccs":
		if len(args) != 0 {
			return fmt.Errorf("usage: digraph sccs")
		}
		for _, scc := range g.sccs() {
			scc.sort().println(" ")
		}

	case "scc":
		if len(args) != 1 {
			return fmt.Errorf("usage: digraph scc <label>")
		}
		label := args[0]
		if g[label] == nil {
			return fmt.Errorf("no such node %q", label)
		}
		for _, scc := range g.sccs() {
			if scc[label] {
				scc.sort().println("\n")
				break
			}
		}

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
//   `one "two three"` -> ["one" "two three"]
//   `a"\n"b` -> ["a\nb"]
//
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
//
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
