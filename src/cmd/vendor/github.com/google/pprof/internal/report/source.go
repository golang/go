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

package report

// This file contains routines related to the generation of annotated
// source listings.

import (
	"bufio"
	"fmt"
	"html/template"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/google/pprof/internal/graph"
	"github.com/google/pprof/internal/plugin"
)

// printSource prints an annotated source listing, include all
// functions with samples that match the regexp rpt.options.symbol.
// The sources are sorted by function name and then by filename to
// eliminate potential nondeterminism.
func printSource(w io.Writer, rpt *Report) error {
	o := rpt.options
	g := rpt.newGraph(nil)

	// Identify all the functions that match the regexp provided.
	// Group nodes for each matching function.
	var functions graph.Nodes
	functionNodes := make(map[string]graph.Nodes)
	for _, n := range g.Nodes {
		if !o.Symbol.MatchString(n.Info.Name) {
			continue
		}
		if functionNodes[n.Info.Name] == nil {
			functions = append(functions, n)
		}
		functionNodes[n.Info.Name] = append(functionNodes[n.Info.Name], n)
	}
	functions.Sort(graph.NameOrder)

	sourcePath := o.SourcePath
	if sourcePath == "" {
		wd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("Could not stat current dir: %v", err)
		}
		sourcePath = wd
	}

	fmt.Fprintf(w, "Total: %s\n", rpt.formatValue(rpt.total))
	for _, fn := range functions {
		name := fn.Info.Name

		// Identify all the source files associated to this function.
		// Group nodes for each source file.
		var sourceFiles graph.Nodes
		fileNodes := make(map[string]graph.Nodes)
		for _, n := range functionNodes[name] {
			if n.Info.File == "" {
				continue
			}
			if fileNodes[n.Info.File] == nil {
				sourceFiles = append(sourceFiles, n)
			}
			fileNodes[n.Info.File] = append(fileNodes[n.Info.File], n)
		}

		if len(sourceFiles) == 0 {
			fmt.Fprintf(w, "No source information for %s\n", name)
			continue
		}

		sourceFiles.Sort(graph.FileOrder)

		// Print each file associated with this function.
		for _, fl := range sourceFiles {
			filename := fl.Info.File
			fns := fileNodes[filename]
			flatSum, cumSum := fns.Sum()

			fnodes, _, err := getSourceFromFile(filename, sourcePath, fns, 0, 0)
			fmt.Fprintf(w, "ROUTINE ======================== %s in %s\n", name, filename)
			fmt.Fprintf(w, "%10s %10s (flat, cum) %s of Total\n",
				rpt.formatValue(flatSum), rpt.formatValue(cumSum),
				percentage(cumSum, rpt.total))

			if err != nil {
				fmt.Fprintf(w, " Error: %v\n", err)
				continue
			}

			for _, fn := range fnodes {
				fmt.Fprintf(w, "%10s %10s %6d:%s\n", valueOrDot(fn.Flat, rpt), valueOrDot(fn.Cum, rpt), fn.Info.Lineno, fn.Info.Name)
			}
		}
	}
	return nil
}

// printWebSource prints an annotated source listing, include all
// functions with samples that match the regexp rpt.options.symbol.
func printWebSource(w io.Writer, rpt *Report, obj plugin.ObjTool) error {
	o := rpt.options
	g := rpt.newGraph(nil)

	// If the regexp source can be parsed as an address, also match
	// functions that land on that address.
	var address *uint64
	if hex, err := strconv.ParseUint(o.Symbol.String(), 0, 64); err == nil {
		address = &hex
	}

	sourcePath := o.SourcePath
	if sourcePath == "" {
		wd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("Could not stat current dir: %v", err)
		}
		sourcePath = wd
	}

	type fileFunction struct {
		fileName, functionName string
	}

	// Extract interesting symbols from binary files in the profile and
	// classify samples per symbol.
	symbols := symbolsFromBinaries(rpt.prof, g, o.Symbol, address, obj)
	symNodes := nodesPerSymbol(g.Nodes, symbols)

	// Identify sources associated to a symbol by examining
	// symbol samples. Classify samples per source file.
	fileNodes := make(map[fileFunction]graph.Nodes)
	if len(symNodes) == 0 {
		for _, n := range g.Nodes {
			if n.Info.File == "" || !o.Symbol.MatchString(n.Info.Name) {
				continue
			}
			ff := fileFunction{n.Info.File, n.Info.Name}
			fileNodes[ff] = append(fileNodes[ff], n)
		}
	} else {
		for _, nodes := range symNodes {
			for _, n := range nodes {
				if n.Info.File != "" {
					ff := fileFunction{n.Info.File, n.Info.Name}
					fileNodes[ff] = append(fileNodes[ff], n)
				}
			}
		}
	}

	if len(fileNodes) == 0 {
		return fmt.Errorf("No source information for %s\n", o.Symbol.String())
	}

	sourceFiles := make(graph.Nodes, 0, len(fileNodes))
	for _, nodes := range fileNodes {
		sNode := *nodes[0]
		sNode.Flat, sNode.Cum = nodes.Sum()
		sourceFiles = append(sourceFiles, &sNode)
	}
	sourceFiles.Sort(graph.FileOrder)

	// Print each file associated with this function.
	printHeader(w, rpt)
	for _, n := range sourceFiles {
		ff := fileFunction{n.Info.File, n.Info.Name}
		fns := fileNodes[ff]

		asm := assemblyPerSourceLine(symbols, fns, ff.fileName, obj)
		start, end := sourceCoordinates(asm)

		fnodes, path, err := getSourceFromFile(ff.fileName, sourcePath, fns, start, end)
		if err != nil {
			fnodes, path = getMissingFunctionSource(ff.fileName, asm, start, end)
		}

		printFunctionHeader(w, ff.functionName, path, n.Flat, n.Cum, rpt)
		for _, fn := range fnodes {
			printFunctionSourceLine(w, fn, asm[fn.Info.Lineno], rpt)
		}
		printFunctionClosing(w)
	}
	printPageClosing(w)
	return nil
}

// sourceCoordinates returns the lowest and highest line numbers from
// a set of assembly statements.
func sourceCoordinates(asm map[int][]assemblyInstruction) (start, end int) {
	for l := range asm {
		if start == 0 || l < start {
			start = l
		}
		if end == 0 || l > end {
			end = l
		}
	}
	return start, end
}

// assemblyPerSourceLine disassembles the binary containing a symbol
// and classifies the assembly instructions according to its
// corresponding source line, annotating them with a set of samples.
func assemblyPerSourceLine(objSyms []*objSymbol, rs graph.Nodes, src string, obj plugin.ObjTool) map[int][]assemblyInstruction {
	assembly := make(map[int][]assemblyInstruction)
	// Identify symbol to use for this collection of samples.
	o := findMatchingSymbol(objSyms, rs)
	if o == nil {
		return assembly
	}

	// Extract assembly for matched symbol
	insts, err := obj.Disasm(o.sym.File, o.sym.Start, o.sym.End)
	if err != nil {
		return assembly
	}

	srcBase := filepath.Base(src)
	anodes := annotateAssembly(insts, rs, o.base)
	var lineno = 0
	for _, an := range anodes {
		if filepath.Base(an.file) == srcBase {
			lineno = an.line
		}
		if lineno != 0 {
			assembly[lineno] = append(assembly[lineno], an)
		}
	}

	return assembly
}

// findMatchingSymbol looks for the symbol that corresponds to a set
// of samples, by comparing their addresses.
func findMatchingSymbol(objSyms []*objSymbol, ns graph.Nodes) *objSymbol {
	for _, n := range ns {
		for _, o := range objSyms {
			if filepath.Base(o.sym.File) == filepath.Base(n.Info.Objfile) &&
				o.sym.Start <= n.Info.Address-o.base &&
				n.Info.Address-o.base <= o.sym.End {
				return o
			}
		}
	}
	return nil
}

// printHeader prints the page header for a weblist report.
func printHeader(w io.Writer, rpt *Report) {
	fmt.Fprintln(w, weblistPageHeader)

	var labels []string
	for _, l := range ProfileLabels(rpt) {
		labels = append(labels, template.HTMLEscapeString(l))
	}

	fmt.Fprintf(w, `<div class="legend">%s<br>Total: %s</div>`,
		strings.Join(labels, "<br>\n"),
		rpt.formatValue(rpt.total),
	)
}

// printFunctionHeader prints a function header for a weblist report.
func printFunctionHeader(w io.Writer, name, path string, flatSum, cumSum int64, rpt *Report) {
	fmt.Fprintf(w, `<h1>%s</h1>%s
<pre onClick="pprof_toggle_asm(event)">
  Total:  %10s %10s (flat, cum) %s
`,
		template.HTMLEscapeString(name), template.HTMLEscapeString(path),
		rpt.formatValue(flatSum), rpt.formatValue(cumSum),
		percentage(cumSum, rpt.total))
}

// printFunctionSourceLine prints a source line and the corresponding assembly.
func printFunctionSourceLine(w io.Writer, fn *graph.Node, assembly []assemblyInstruction, rpt *Report) {
	if len(assembly) == 0 {
		fmt.Fprintf(w,
			"<span class=line> %6d</span> <span class=nop>  %10s %10s %s </span>\n",
			fn.Info.Lineno,
			valueOrDot(fn.Flat, rpt), valueOrDot(fn.Cum, rpt),
			template.HTMLEscapeString(fn.Info.Name))
		return
	}

	fmt.Fprintf(w,
		"<span class=line> %6d</span> <span class=deadsrc>  %10s %10s %s </span>",
		fn.Info.Lineno,
		valueOrDot(fn.Flat, rpt), valueOrDot(fn.Cum, rpt),
		template.HTMLEscapeString(fn.Info.Name))
	fmt.Fprint(w, "<span class=asm>")
	for _, an := range assembly {
		var fileline string
		class := "disasmloc"
		if an.file != "" {
			fileline = fmt.Sprintf("%s:%d", template.HTMLEscapeString(an.file), an.line)
			if an.line != fn.Info.Lineno {
				class = "unimportant"
			}
		}
		flat, cum := an.flat, an.cum
		if an.flatDiv != 0 {
			flat = flat / an.flatDiv
		}
		if an.cumDiv != 0 {
			cum = cum / an.cumDiv
		}
		fmt.Fprintf(w, " %8s %10s %10s %8x: %-48s <span class=%s>%s</span>\n", "",
			valueOrDot(flat, rpt), valueOrDot(cum, rpt),
			an.address,
			template.HTMLEscapeString(an.instruction),
			class,
			template.HTMLEscapeString(fileline))
	}
	fmt.Fprintln(w, "</span>")
}

// printFunctionClosing prints the end of a function in a weblist report.
func printFunctionClosing(w io.Writer) {
	fmt.Fprintln(w, "</pre>")
}

// printPageClosing prints the end of the page in a weblist report.
func printPageClosing(w io.Writer) {
	fmt.Fprintln(w, weblistPageClosing)
}

// getSourceFromFile collects the sources of a function from a source
// file and annotates it with the samples in fns. Returns the sources
// as nodes, using the info.name field to hold the source code.
func getSourceFromFile(file, sourcePath string, fns graph.Nodes, start, end int) (graph.Nodes, string, error) {
	file = trimPath(file)
	f, err := openSourceFile(file, sourcePath)
	if err != nil {
		return nil, file, err
	}

	lineNodes := make(map[int]graph.Nodes)
	// Collect source coordinates from profile.
	const margin = 5 // Lines before first/after last sample.
	if start == 0 {
		if fns[0].Info.StartLine != 0 {
			start = fns[0].Info.StartLine
		} else {
			start = fns[0].Info.Lineno - margin
		}
	} else {
		start -= margin
	}
	if end == 0 {
		end = fns[0].Info.Lineno
	}
	end += margin
	for _, n := range fns {
		lineno := n.Info.Lineno
		nodeStart := n.Info.StartLine
		if nodeStart == 0 {
			nodeStart = lineno - margin
		}
		nodeEnd := lineno + margin
		if nodeStart < start {
			start = nodeStart
		} else if nodeEnd > end {
			end = nodeEnd
		}
		lineNodes[lineno] = append(lineNodes[lineno], n)
	}

	var src graph.Nodes
	buf := bufio.NewReader(f)
	lineno := 1
	for {
		line, err := buf.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return nil, file, err
			}
			if line == "" {
				break
			}
		}
		if lineno >= start {
			flat, cum := lineNodes[lineno].Sum()

			src = append(src, &graph.Node{
				Info: graph.NodeInfo{
					Name:   strings.TrimRight(line, "\n"),
					Lineno: lineno,
				},
				Flat: flat,
				Cum:  cum,
			})
		}
		lineno++
		if lineno > end {
			break
		}
	}
	return src, file, nil
}

// getMissingFunctionSource creates a dummy function body to point to
// the source file and annotates it with the samples in asm.
func getMissingFunctionSource(filename string, asm map[int][]assemblyInstruction, start, end int) (graph.Nodes, string) {
	var fnodes graph.Nodes
	for i := start; i <= end; i++ {
		insts := asm[i]
		if len(insts) == 0 {
			continue
		}
		var group assemblyInstruction
		for _, insn := range insts {
			group.flat += insn.flat
			group.cum += insn.cum
			group.flatDiv += insn.flatDiv
			group.cumDiv += insn.cumDiv
		}
		flat := group.flatValue()
		cum := group.cumValue()
		fnodes = append(fnodes, &graph.Node{
			Info: graph.NodeInfo{
				Name:   "???",
				Lineno: i,
			},
			Flat: flat,
			Cum:  cum,
		})
	}
	return fnodes, filename
}

// openSourceFile opens a source file from a name encoded in a
// profile. File names in a profile after often relative paths, so
// search them in each of the paths in searchPath (or CWD by default),
// and their parents.
func openSourceFile(path, searchPath string) (*os.File, error) {
	if filepath.IsAbs(path) {
		f, err := os.Open(path)
		return f, err
	}

	// Scan each component of the path
	for _, dir := range strings.Split(searchPath, ":") {
		// Search up for every parent of each possible path.
		for {
			filename := filepath.Join(dir, path)
			if f, err := os.Open(filename); err == nil {
				return f, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	return nil, fmt.Errorf("Could not find file %s on path %s", path, searchPath)
}

// trimPath cleans up a path by removing prefixes that are commonly
// found on profiles.
func trimPath(path string) string {
	basePaths := []string{
		"/proc/self/cwd/./",
		"/proc/self/cwd/",
	}

	sPath := filepath.ToSlash(path)

	for _, base := range basePaths {
		if strings.HasPrefix(sPath, base) {
			return filepath.FromSlash(sPath[len(base):])
		}
	}
	return path
}
