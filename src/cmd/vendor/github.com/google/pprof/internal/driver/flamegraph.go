// Copyright 2017 Google Inc. All Rights Reserved.
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

package driver

import (
	"encoding/json"
	"html/template"
	"net/http"
	"strings"

	"github.com/google/pprof/internal/graph"
	"github.com/google/pprof/internal/measurement"
	"github.com/google/pprof/internal/report"
)

type treeNode struct {
	Name      string      `json:"n"`
	FullName  string      `json:"f"`
	Cum       int64       `json:"v"`
	CumFormat string      `json:"l"`
	Percent   string      `json:"p"`
	Children  []*treeNode `json:"c"`
}

// flamegraph generates a web page containing a flamegraph.
func (ui *webInterface) flamegraph(w http.ResponseWriter, req *http.Request) {
	// Force the call tree so that the graph is a tree.
	// Also do not trim the tree so that the flame graph contains all functions.
	rpt, errList := ui.makeReport(w, req, []string{"svg"}, "call_tree", "true", "trim", "false")
	if rpt == nil {
		return // error already reported
	}

	// Generate dot graph.
	g, config := report.GetDOT(rpt)
	var nodes []*treeNode
	nroots := 0
	rootValue := int64(0)
	nodeArr := []string{}
	nodeMap := map[*graph.Node]*treeNode{}
	// Make all nodes and the map, collect the roots.
	for _, n := range g.Nodes {
		v := n.CumValue()
		fullName := n.Info.PrintableName()
		node := &treeNode{
			Name:      getNodeShortName(fullName),
			FullName:  fullName,
			Cum:       v,
			CumFormat: config.FormatValue(v),
			Percent:   strings.TrimSpace(measurement.Percentage(v, config.Total)),
		}
		nodes = append(nodes, node)
		if len(n.In) == 0 {
			nodes[nroots], nodes[len(nodes)-1] = nodes[len(nodes)-1], nodes[nroots]
			nroots++
			rootValue += v
		}
		nodeMap[n] = node
		// Get all node names into an array.
		nodeArr = append(nodeArr, n.Info.Name)
	}
	// Populate the child links.
	for _, n := range g.Nodes {
		node := nodeMap[n]
		for child := range n.Out {
			node.Children = append(node.Children, nodeMap[child])
		}
	}

	rootNode := &treeNode{
		Name:      "root",
		FullName:  "root",
		Cum:       rootValue,
		CumFormat: config.FormatValue(rootValue),
		Percent:   strings.TrimSpace(measurement.Percentage(rootValue, config.Total)),
		Children:  nodes[0:nroots],
	}

	// JSON marshalling flame graph
	b, err := json.Marshal(rootNode)
	if err != nil {
		http.Error(w, "error serializing flame graph", http.StatusInternalServerError)
		ui.options.UI.PrintErr(err)
		return
	}

	ui.render(w, "flamegraph", rpt, errList, config.Labels, webArgs{
		FlameGraph: template.JS(b),
		Nodes:      nodeArr,
	})
}

// getNodeShortName builds a short node name from fullName.
func getNodeShortName(name string) string {
	chunks := strings.SplitN(name, "(", 2)
	head := chunks[0]
	pathSep := strings.LastIndexByte(head, '/')
	if pathSep == -1 || pathSep+1 >= len(head) {
		return name
	}
	// Check if name is a stdlib package, i.e. doesn't have "." before "/"
	if dot := strings.IndexByte(head, '.'); dot == -1 || dot > pathSep {
		return name
	}
	// Trim package path prefix from node name
	return name[pathSep+1:]
}
