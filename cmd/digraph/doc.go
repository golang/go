// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The digraph command performs queries over unlabelled directed graphs
represented in text form.  It is intended to integrate nicely with
typical UNIX command pipelines.

Usage:

	your-application | digraph [command]

The supported commands are:

	nodes
		the set of all nodes
	degree
		the in-degree and out-degree of each node
	transpose
		the reverse of the input edges
	preds <node> ...
		the set of immediate predecessors of the specified nodes
	succs <node> ...
		the set of immediate successors of the specified nodes
	forward <node> ...
		the set of nodes transitively reachable from the specified nodes
	reverse <node> ...
		the set of nodes that transitively reach the specified nodes
	somepath <node> <node>
		the list of nodes on some arbitrary path from the first node to the second
	allpaths <node> <node>
		the set of nodes on all paths from the first node to the second
	sccs
		all strongly connected components (one per line)
	scc <node>
		the set of nodes strongly connected to the specified one
	focus <node>
		the subgraph containing all directed paths that pass through the specified node
	to dot
		print the graph in Graphviz dot format (other formats may be supported in the future)

Input format:

Each line contains zero or more words. Words are separated by unquoted
whitespace; words may contain Go-style double-quoted portions, allowing spaces
and other characters to be expressed.

Each word declares a node, and if there are more than one, an edge from the
first to each subsequent one. The graph is provided on the standard input.

For instance, the following (acyclic) graph specifies a partial order among the
subtasks of getting dressed:

	$ cat clothes.txt
	socks shoes
	"boxer shorts" pants
	pants belt shoes
	shirt tie sweater
	sweater jacket
	hat

The line "shirt tie sweater" indicates the two edges shirt -> tie and
shirt -> sweater, not shirt -> tie -> sweater.

Example usage:

Show which clothes (see above) must be donned before a jacket:

	$ digraph reverse jacket

Many tools can be persuaded to produce output in digraph format,
as in the following examples.

Using an import graph produced by go list, show a path that indicates
why the gopls application depends on the cmp package:

	$ go list -f '{{.ImportPath}} {{join .Imports " "}}' -deps golang.org/x/tools/gopls |
		digraph somepath golang.org/x/tools/gopls github.com/google/go-cmp/cmp

Show which packages in x/tools depend, perhaps indirectly, on the callgraph package:

	$ go list -f '{{.ImportPath}} {{join .Imports " "}}' -deps golang.org/x/tools/... |
		digraph reverse golang.org/x/tools/go/callgraph

Visualize the package dependency graph of the current package:

	$ go list -f '{{.ImportPath}} {{join .Imports " "}}' -deps |
		digraph to dot | dot -Tpng -o x.png

Using a module graph produced by go mod, show all dependencies of the current module:

	$ go mod graph | digraph forward $(go list -m)
*/
package main
