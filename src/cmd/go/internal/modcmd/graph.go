// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod graph

package modcmd

import (
	"bufio"
	"context"
	"os"
	"sort"

	"cmd/go/internal/base"
	"cmd/go/internal/modload"

	"golang.org/x/mod/module"
)

var cmdGraph = &base.Command{
	UsageLine: "go mod graph",
	Short:     "print module requirement graph",
	Long: `
Graph prints the module requirement graph (with replacements applied)
in text form. Each line in the output has two space-separated fields: a module
and one of its requirements. Each module is identified as a string of the form
path@version, except for the main module, which has no @version suffix.

See https://golang.org/ref/mod#go-mod-graph for more about 'go mod graph'.
	`,
	Run: runGraph,
}

func init() {
	base.AddModCommonFlags(&cmdGraph.Flag)
}

func runGraph(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) > 0 {
		base.Fatalf("go mod graph: graph takes no arguments")
	}
	modload.ForceUseModules = true
	modload.RootMode = modload.NeedRoot
	modload.LoadAllModules(ctx)

	reqs := modload.MinReqs()
	format := func(m module.Version) string {
		if m.Version == "" {
			return m.Path
		}
		return m.Path + "@" + m.Version
	}

	var out []string
	var deps int // index in out where deps start
	seen := map[module.Version]bool{modload.Target: true}
	queue := []module.Version{modload.Target}
	for len(queue) > 0 {
		var m module.Version
		m, queue = queue[0], queue[1:]
		list, _ := reqs.Required(m)
		for _, r := range list {
			if !seen[r] {
				queue = append(queue, r)
				seen[r] = true
			}
			out = append(out, format(m)+" "+format(r)+"\n")
		}
		if m == modload.Target {
			deps = len(out)
		}
	}

	sort.Slice(out[deps:], func(i, j int) bool {
		return out[deps+i][0] < out[deps+j][0]
	})

	w := bufio.NewWriter(os.Stdout)
	for _, line := range out {
		w.WriteString(line)
	}
	w.Flush()
}
