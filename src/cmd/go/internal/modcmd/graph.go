// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod graph

package modcmd

import (
	"bufio"
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/modload"
	"cmd/go/internal/toolchain"
	"context"
	"os"

	"golang.org/x/mod/module"
)

var cmdGraph = &base.Command{
	UsageLine: "go mod graph [-go=version] [-x] [-selectedVersions]",
	Short:     "print module requirement graph",
	Long: `
Graph prints the module requirement graph (with replacements applied)
in text form. Each line in the output has two space-separated fields: a module
and one of its requirements. Each module is identified as a string of the form
path@version, except for the main module, which has no @version suffix.

The -go flag causes graph to report the module graph as loaded by the
given Go version, instead of the version indicated by the 'go' directive
in the go.mod file.

The -x flag causes graph to print the commands graph executes.

The -selectedVersions flag cause graph to contains only packages with versions that were selected 
by the minimal version selected algorithm.

See https://golang.org/ref/mod#go-mod-graph for more about 'go mod graph'.
	`,
	Run: runGraph,
}

var (
	graphGo          goVersionFlag
	selectedVersions bool
)

func init() {
	cmdGraph.Flag.Var(&graphGo, "go", "")
	cmdGraph.Flag.BoolVar(&cfg.BuildX, "x", false, "")
	cmdGraph.Flag.BoolVar(&selectedVersions, "selectedVersions", false, "")
	base.AddChdirFlag(&cmdGraph.Flag)
	base.AddModCommonFlags(&cmdGraph.Flag)
}

func filterVersions(versions []module.Version, mg *modload.ModuleGraph) []module.Version {
	var filtered []module.Version

	// Iterating over the versions and filtering by selected versions
	for _, version := range versions {

		if version.Version == mg.Selected(version.Path) {
			filtered = append(filtered, version)
		}
	}

	return filtered
}

func walkOnSelectedVersions(version module.Version, walkNodes map[string]bool, w *bufio.Writer, mg *modload.ModuleGraph) {
	walkNodes[version.String()] = true

	reqs, ok := mg.RequiredBy(version)

	// filter out requirements versions that were not selected.
	reqs = filterVersions(reqs, mg)

	if ok {
		// print the dependencies of the selected package
		for _, req := range reqs {
			format(version, w)
			w.WriteByte(' ')
			format(req, w)
			w.WriteByte('\n')
		}

		// iterate on all the dependencies of the selected package
		for _, req := range reqs {
			if _, exists := walkNodes[req.String()]; !exists {
				walkOnSelectedVersions(req, walkNodes, w, mg)
			}
		}
	}
}

func format(m module.Version, w *bufio.Writer) {
	w.WriteString(m.Path)
	if m.Version != "" {
		w.WriteString("@")
		w.WriteString(m.Version)
	}
}

func runGraph(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()

	if len(args) > 0 {
		base.Fatalf("go: 'go mod graph' accepts no arguments")
	}
	modload.ForceUseModules = true
	modload.RootMode = modload.NeedRoot

	goVersion := graphGo.String()
	if goVersion != "" && gover.Compare(gover.Local(), goVersion) < 0 {
		toolchain.SwitchOrFatal(ctx, &gover.TooNewError{
			What:      "-go flag",
			GoVersion: goVersion,
		})
	}

	mg, err := modload.LoadModGraph(ctx, goVersion)
	if err != nil {
		base.Fatal(err)
	}
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()

	if selectedVersions {
		rootNode := mg.BuildList()[0]
		walkNodes := make(map[string]bool)
		walkOnSelectedVersions(rootNode, walkNodes, w, mg)
	} else {
		mg.WalkBreadthFirst(func(m module.Version) {
			reqs, _ := mg.RequiredBy(m)
			for _, r := range reqs {
				format(m, w)
				w.WriteByte(' ')
				format(r, w)
				w.WriteByte('\n')
			}
		})
	}
}
