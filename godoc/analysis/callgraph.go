// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysis

// This file computes the CALLERS and CALLEES relations from the call
// graph.  CALLERS/CALLEES information is displayed in the lower pane
// when a "func" token or ast.CallExpr.Lparen is clicked, respectively.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log"
	"math/big"
	"sort"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
)

// doCallgraph computes the CALLEES and CALLERS relations.
func (a *analysis) doCallgraph(cg *callgraph.Graph) {
	log.Print("Deleting synthetic nodes...")
	// TODO(adonovan): opt: DeleteSyntheticNodes is asymptotically
	// inefficient and can be (unpredictably) slow.
	cg.DeleteSyntheticNodes()
	log.Print("Synthetic nodes deleted")

	// Populate nodes of package call graphs (PCGs).
	for _, n := range cg.Nodes {
		a.pcgAddNode(n.Func)
	}
	// Within each PCG, sort funcs by name.
	for _, pcg := range a.pcgs {
		pcg.sortNodes()
	}

	calledFuncs := make(map[ssa.CallInstruction]map[*ssa.Function]bool)
	callingSites := make(map[*ssa.Function]map[ssa.CallInstruction]bool)
	for _, n := range cg.Nodes {
		for _, e := range n.Out {
			if e.Site == nil {
				continue // a call from a synthetic node such as <root>
			}

			// Add (site pos, callee) to calledFuncs.
			// (Dynamic calls only.)
			callee := e.Callee.Func

			a.pcgAddEdge(n.Func, callee)

			if callee.Synthetic != "" {
				continue // call of a package initializer
			}

			if e.Site.Common().StaticCallee() == nil {
				// dynamic call
				// (CALLEES information for static calls
				// is computed using SSA information.)
				lparen := e.Site.Common().Pos()
				if lparen != token.NoPos {
					fns := calledFuncs[e.Site]
					if fns == nil {
						fns = make(map[*ssa.Function]bool)
						calledFuncs[e.Site] = fns
					}
					fns[callee] = true
				}
			}

			// Add (callee, site) to callingSites.
			fns := callingSites[callee]
			if fns == nil {
				fns = make(map[ssa.CallInstruction]bool)
				callingSites[callee] = fns
			}
			fns[e.Site] = true
		}
	}

	// CALLEES.
	log.Print("Callees...")
	for site, fns := range calledFuncs {
		var funcs funcsByPos
		for fn := range fns {
			funcs = append(funcs, fn)
		}
		sort.Sort(funcs)

		a.addCallees(site, funcs)
	}

	// CALLERS
	log.Print("Callers...")
	for callee, sites := range callingSites {
		pos := funcToken(callee)
		if pos == token.NoPos {
			log.Printf("CALLERS: skipping %s: no pos", callee)
			continue
		}

		var this *types.Package // for relativizing names
		if callee.Pkg != nil {
			this = callee.Pkg.Pkg
		}

		// Compute sites grouped by parent, with text and URLs.
		sitesByParent := make(map[*ssa.Function]sitesByPos)
		for site := range sites {
			fn := site.Parent()
			sitesByParent[fn] = append(sitesByParent[fn], site)
		}
		var funcs funcsByPos
		for fn := range sitesByParent {
			funcs = append(funcs, fn)
		}
		sort.Sort(funcs)

		v := callersJSON{
			Callee:  callee.String(),
			Callers: []callerJSON{}, // (JS wants non-nil)
		}
		for _, fn := range funcs {
			caller := callerJSON{
				Func:  prettyFunc(this, fn),
				Sites: []anchorJSON{}, // (JS wants non-nil)
			}
			sites := sitesByParent[fn]
			sort.Sort(sites)
			for _, site := range sites {
				pos := site.Common().Pos()
				if pos != token.NoPos {
					caller.Sites = append(caller.Sites, anchorJSON{
						Text: fmt.Sprintf("%d", a.prog.Fset.Position(pos).Line),
						Href: a.posURL(pos, len("(")),
					})
				}
			}
			v.Callers = append(v.Callers, caller)
		}

		fi, offset := a.fileAndOffset(pos)
		fi.addLink(aLink{
			start:   offset,
			end:     offset + len("func"),
			title:   fmt.Sprintf("%d callers", len(sites)),
			onclick: fmt.Sprintf("onClickCallers(%d)", fi.addData(v)),
		})
	}

	// PACKAGE CALLGRAPH
	log.Print("Package call graph...")
	for pkg, pcg := range a.pcgs {
		// Maps (*ssa.Function).RelString() to index in JSON CALLGRAPH array.
		index := make(map[string]int)

		// Treat exported functions (and exported methods of
		// exported named types) as roots even if they aren't
		// actually called from outside the package.
		for i, n := range pcg.nodes {
			if i == 0 || n.fn.Object() == nil || !n.fn.Object().Exported() {
				continue
			}
			recv := n.fn.Signature.Recv()
			if recv == nil || deref(recv.Type()).(*types.Named).Obj().Exported() {
				roots := &pcg.nodes[0].edges
				roots.SetBit(roots, i, 1)
			}
			index[n.fn.RelString(pkg.Pkg)] = i
		}

		json := a.pcgJSON(pcg)

		// TODO(adonovan): pkg.Path() is not unique!
		// It is possible to declare a non-test package called x_test.
		a.result.pkgInfo(pkg.Pkg.Path()).setCallGraph(json, index)
	}
}

// addCallees adds client data and links for the facts that site calls fns.
func (a *analysis) addCallees(site ssa.CallInstruction, fns []*ssa.Function) {
	v := calleesJSON{
		Descr:   site.Common().Description(),
		Callees: []anchorJSON{}, // (JS wants non-nil)
	}
	var this *types.Package // for relativizing names
	if p := site.Parent().Package(); p != nil {
		this = p.Pkg
	}

	for _, fn := range fns {
		v.Callees = append(v.Callees, anchorJSON{
			Text: prettyFunc(this, fn),
			Href: a.posURL(funcToken(fn), len("func")),
		})
	}

	fi, offset := a.fileAndOffset(site.Common().Pos())
	fi.addLink(aLink{
		start:   offset,
		end:     offset + len("("),
		title:   fmt.Sprintf("%d callees", len(v.Callees)),
		onclick: fmt.Sprintf("onClickCallees(%d)", fi.addData(v)),
	})
}

// -- utilities --------------------------------------------------------

// stable order within packages but undefined across packages.
type funcsByPos []*ssa.Function

func (a funcsByPos) Less(i, j int) bool { return a[i].Pos() < a[j].Pos() }
func (a funcsByPos) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a funcsByPos) Len() int           { return len(a) }

type sitesByPos []ssa.CallInstruction

func (a sitesByPos) Less(i, j int) bool { return a[i].Common().Pos() < a[j].Common().Pos() }
func (a sitesByPos) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a sitesByPos) Len() int           { return len(a) }

func funcToken(fn *ssa.Function) token.Pos {
	switch syntax := fn.Syntax().(type) {
	case *ast.FuncLit:
		return syntax.Type.Func
	case *ast.FuncDecl:
		return syntax.Type.Func
	}
	return token.NoPos
}

// prettyFunc pretty-prints fn for the user interface.
// TODO(adonovan): return HTML so we have more markup freedom.
func prettyFunc(this *types.Package, fn *ssa.Function) string {
	if fn.Parent() != nil {
		return fmt.Sprintf("%s in %s",
			types.TypeString(fn.Signature, types.RelativeTo(this)),
			prettyFunc(this, fn.Parent()))
	}
	if fn.Synthetic != "" && fn.Name() == "init" {
		// (This is the actual initializer, not a declared 'func init').
		if fn.Pkg.Pkg == this {
			return "package initializer"
		}
		return fmt.Sprintf("%q package initializer", fn.Pkg.Pkg.Path())
	}
	return fn.RelString(this)
}

// -- intra-package callgraph ------------------------------------------

// pcgNode represents a node in the package call graph (PCG).
type pcgNode struct {
	fn     *ssa.Function
	pretty string  // cache of prettyFunc(fn)
	edges  big.Int // set of callee func indices
}

// A packageCallGraph represents the intra-package edges of the global call graph.
// The zeroth node indicates "all external functions".
type packageCallGraph struct {
	nodeIndex map[*ssa.Function]int // maps func to node index (a small int)
	nodes     []*pcgNode            // maps node index to node
}

// sortNodes populates pcg.nodes in name order and updates the nodeIndex.
func (pcg *packageCallGraph) sortNodes() {
	nodes := make([]*pcgNode, 0, len(pcg.nodeIndex))
	nodes = append(nodes, &pcgNode{fn: nil, pretty: "<external>"})
	for fn := range pcg.nodeIndex {
		nodes = append(nodes, &pcgNode{
			fn:     fn,
			pretty: prettyFunc(fn.Pkg.Pkg, fn),
		})
	}
	sort.Sort(pcgNodesByPretty(nodes[1:]))
	for i, n := range nodes {
		pcg.nodeIndex[n.fn] = i
	}
	pcg.nodes = nodes
}

func (pcg *packageCallGraph) addEdge(caller, callee *ssa.Function) {
	var callerIndex int
	if caller.Pkg == callee.Pkg {
		// intra-package edge
		callerIndex = pcg.nodeIndex[caller]
		if callerIndex < 1 {
			panic(caller)
		}
	}
	edges := &pcg.nodes[callerIndex].edges
	edges.SetBit(edges, pcg.nodeIndex[callee], 1)
}

func (a *analysis) pcgAddNode(fn *ssa.Function) {
	if fn.Pkg == nil {
		return
	}
	pcg, ok := a.pcgs[fn.Pkg]
	if !ok {
		pcg = &packageCallGraph{nodeIndex: make(map[*ssa.Function]int)}
		a.pcgs[fn.Pkg] = pcg
	}
	pcg.nodeIndex[fn] = -1
}

func (a *analysis) pcgAddEdge(caller, callee *ssa.Function) {
	if callee.Pkg != nil {
		a.pcgs[callee.Pkg].addEdge(caller, callee)
	}
}

// pcgJSON returns a new slice of callgraph JSON values.
func (a *analysis) pcgJSON(pcg *packageCallGraph) []*PCGNodeJSON {
	var nodes []*PCGNodeJSON
	for _, n := range pcg.nodes {

		// TODO(adonovan): why is there no good way to iterate
		// over the set bits of a big.Int?
		var callees []int
		nbits := n.edges.BitLen()
		for j := 0; j < nbits; j++ {
			if n.edges.Bit(j) == 1 {
				callees = append(callees, j)
			}
		}

		var pos token.Pos
		if n.fn != nil {
			pos = funcToken(n.fn)
		}
		nodes = append(nodes, &PCGNodeJSON{
			Func: anchorJSON{
				Text: n.pretty,
				Href: a.posURL(pos, len("func")),
			},
			Callees: callees,
		})
	}
	return nodes
}

type pcgNodesByPretty []*pcgNode

func (a pcgNodesByPretty) Less(i, j int) bool { return a[i].pretty < a[j].pretty }
func (a pcgNodesByPretty) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a pcgNodesByPretty) Len() int           { return len(a) }
