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
	"github.com/google/pprof/internal/measurement"
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
			return fmt.Errorf("could not stat current dir: %v", err)
		}
		sourcePath = wd
	}
	reader := newSourceReader(sourcePath, o.TrimPath)

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

			fnodes, _, err := getSourceFromFile(filename, reader, fns, 0, 0)
			fmt.Fprintf(w, "ROUTINE ======================== %s in %s\n", name, filename)
			fmt.Fprintf(w, "%10s %10s (flat, cum) %s of Total\n",
				rpt.formatValue(flatSum), rpt.formatValue(cumSum),
				measurement.Percentage(cumSum, rpt.total))

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
	printHeader(w, rpt)
	if err := PrintWebList(w, rpt, obj, -1); err != nil {
		return err
	}
	printPageClosing(w)
	return nil
}

// PrintWebList prints annotated source listing of rpt to w.
func PrintWebList(w io.Writer, rpt *Report, obj plugin.ObjTool, maxFiles int) error {
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
			return fmt.Errorf("could not stat current dir: %v", err)
		}
		sourcePath = wd
	}
	reader := newSourceReader(sourcePath, o.TrimPath)

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
		return fmt.Errorf("no source information for %s", o.Symbol.String())
	}

	sourceFiles := make(graph.Nodes, 0, len(fileNodes))
	for _, nodes := range fileNodes {
		sNode := *nodes[0]
		sNode.Flat, sNode.Cum = nodes.Sum()
		sourceFiles = append(sourceFiles, &sNode)
	}

	// Limit number of files printed?
	if maxFiles < 0 {
		sourceFiles.Sort(graph.FileOrder)
	} else {
		sourceFiles.Sort(graph.FlatNameOrder)
		if maxFiles < len(sourceFiles) {
			sourceFiles = sourceFiles[:maxFiles]
		}
	}

	// Print each file associated with this function.
	for _, n := range sourceFiles {
		ff := fileFunction{n.Info.File, n.Info.Name}
		fns := fileNodes[ff]

		asm := assemblyPerSourceLine(symbols, fns, ff.fileName, obj)
		start, end := sourceCoordinates(asm)

		fnodes, path, err := getSourceFromFile(ff.fileName, reader, fns, start, end)
		if err != nil {
			fnodes, path = getMissingFunctionSource(ff.fileName, asm, start, end)
		}

		printFunctionHeader(w, ff.functionName, path, n.Flat, n.Cum, rpt)
		for _, fn := range fnodes {
			printFunctionSourceLine(w, fn, asm[fn.Info.Lineno], reader, rpt)
		}
		printFunctionClosing(w)
	}
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
	var prevline = 0
	for _, an := range anodes {
		// Do not rely solely on the line number produced by Disasm
		// since it is not what we want in the presence of inlining.
		//
		// E.g., suppose we are printing source code for F and this
		// instruction is from H where F called G called H and both
		// of those calls were inlined. We want to use the line
		// number from F, not from H (which is what Disasm gives us).
		//
		// So find the outer-most linenumber in the source file.
		found := false
		if frames, err := o.file.SourceLine(an.address + o.base); err == nil {
			for i := len(frames) - 1; i >= 0; i-- {
				if filepath.Base(frames[i].File) == srcBase {
					for j := i - 1; j >= 0; j-- {
						an.inlineCalls = append(an.inlineCalls, callID{frames[j].File, frames[j].Line})
					}
					lineno = frames[i].Line
					found = true
					break
				}
			}
		}
		if !found && filepath.Base(an.file) == srcBase {
			lineno = an.line
		}

		if lineno != 0 {
			if lineno != prevline {
				// This instruction starts a new block
				// of contiguous instructions on this line.
				an.startsBlock = true
			}
			prevline = lineno
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
	fmt.Fprintln(w, `
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Pprof listing</title>`)
	fmt.Fprintln(w, weblistPageCSS)
	fmt.Fprintln(w, weblistPageScript)
	fmt.Fprint(w, "</head>\n<body>\n\n")

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
	fmt.Fprintf(w, `<h2>%s</h2><p class="filename">%s</p>
<pre onClick="pprof_toggle_asm(event)">
  Total:  %10s %10s (flat, cum) %s
`,
		template.HTMLEscapeString(name), template.HTMLEscapeString(path),
		rpt.formatValue(flatSum), rpt.formatValue(cumSum),
		measurement.Percentage(cumSum, rpt.total))
}

// printFunctionSourceLine prints a source line and the corresponding assembly.
func printFunctionSourceLine(w io.Writer, fn *graph.Node, assembly []assemblyInstruction, reader *sourceReader, rpt *Report) {
	if len(assembly) == 0 {
		fmt.Fprintf(w,
			"<span class=line> %6d</span> <span class=nop>  %10s %10s %8s  %s </span>\n",
			fn.Info.Lineno,
			valueOrDot(fn.Flat, rpt), valueOrDot(fn.Cum, rpt),
			"", template.HTMLEscapeString(fn.Info.Name))
		return
	}

	fmt.Fprintf(w,
		"<span class=line> %6d</span> <span class=deadsrc>  %10s %10s %8s  %s </span>",
		fn.Info.Lineno,
		valueOrDot(fn.Flat, rpt), valueOrDot(fn.Cum, rpt),
		"", template.HTMLEscapeString(fn.Info.Name))
	srcIndent := indentation(fn.Info.Name)
	fmt.Fprint(w, "<span class=asm>")
	var curCalls []callID
	for i, an := range assembly {
		if an.startsBlock && i != 0 {
			// Insert a separator between discontiguous blocks.
			fmt.Fprintf(w, " %8s %28s\n", "", "⋮")
		}

		var fileline string
		if an.file != "" {
			fileline = fmt.Sprintf("%s:%d", template.HTMLEscapeString(an.file), an.line)
		}
		flat, cum := an.flat, an.cum
		if an.flatDiv != 0 {
			flat = flat / an.flatDiv
		}
		if an.cumDiv != 0 {
			cum = cum / an.cumDiv
		}

		// Print inlined call context.
		for j, c := range an.inlineCalls {
			if j < len(curCalls) && curCalls[j] == c {
				// Skip if same as previous instruction.
				continue
			}
			curCalls = nil
			fline, ok := reader.line(c.file, c.line)
			if !ok {
				fline = ""
			}
			text := strings.Repeat(" ", srcIndent+4+4*j) + strings.TrimSpace(fline)
			fmt.Fprintf(w, " %8s %10s %10s %8s  <span class=inlinesrc>%s</span> <span class=unimportant>%s:%d</span>\n",
				"", "", "", "",
				template.HTMLEscapeString(fmt.Sprintf("%-80s", text)),
				template.HTMLEscapeString(filepath.Base(c.file)), c.line)
		}
		curCalls = an.inlineCalls
		text := strings.Repeat(" ", srcIndent+4+4*len(curCalls)) + an.instruction
		fmt.Fprintf(w, " %8s %10s %10s %8x: %s <span class=unimportant>%s</span>\n",
			"", valueOrDot(flat, rpt), valueOrDot(cum, rpt), an.address,
			template.HTMLEscapeString(fmt.Sprintf("%-80s", text)),
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
func getSourceFromFile(file string, reader *sourceReader, fns graph.Nodes, start, end int) (graph.Nodes, string, error) {
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
	if start < 1 {
		start = 1
	}

	var src graph.Nodes
	for lineno := start; lineno <= end; lineno++ {
		line, ok := reader.line(file, lineno)
		if !ok {
			break
		}
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
	if err := reader.fileError(file); err != nil {
		return nil, file, err
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

// sourceReader provides access to source code with caching of file contents.
type sourceReader struct {
	// searchPath is a filepath.ListSeparator-separated list of directories where
	// source files should be searched.
	searchPath string

	// trimPath is a filepath.ListSeparator-separated list of paths to trim.
	trimPath string

	// files maps from path name to a list of lines.
	// files[*][0] is unused since line numbering starts at 1.
	files map[string][]string

	// errors collects errors encountered per file. These errors are
	// consulted before returning out of these module.
	errors map[string]error
}

func newSourceReader(searchPath, trimPath string) *sourceReader {
	return &sourceReader{
		searchPath,
		trimPath,
		make(map[string][]string),
		make(map[string]error),
	}
}

func (reader *sourceReader) fileError(path string) error {
	return reader.errors[path]
}

func (reader *sourceReader) line(path string, lineno int) (string, bool) {
	lines, ok := reader.files[path]
	if !ok {
		// Read and cache file contents.
		lines = []string{""} // Skip 0th line
		f, err := openSourceFile(path, reader.searchPath, reader.trimPath)
		if err != nil {
			reader.errors[path] = err
		} else {
			s := bufio.NewScanner(f)
			for s.Scan() {
				lines = append(lines, s.Text())
			}
			f.Close()
			if s.Err() != nil {
				reader.errors[path] = err
			}
		}
		reader.files[path] = lines
	}
	if lineno <= 0 || lineno >= len(lines) {
		return "", false
	}
	return lines[lineno], true
}

// openSourceFile opens a source file from a name encoded in a profile. File
// names in a profile after can be relative paths, so search them in each of
// the paths in searchPath and their parents. In case the profile contains
// absolute paths, additional paths may be configured to trim from the source
// paths in the profile. This effectively turns the path into a relative path
// searching it using searchPath as usual).
func openSourceFile(path, searchPath, trim string) (*os.File, error) {
	path = trimPath(path, trim, searchPath)
	// If file is still absolute, require file to exist.
	if filepath.IsAbs(path) {
		f, err := os.Open(path)
		return f, err
	}
	// Scan each component of the path.
	for _, dir := range filepath.SplitList(searchPath) {
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

	return nil, fmt.Errorf("could not find file %s on path %s", path, searchPath)
}

// trimPath cleans up a path by removing prefixes that are commonly
// found on profiles plus configured prefixes.
// TODO(aalexand): Consider optimizing out the redundant work done in this
// function if it proves to matter.
func trimPath(path, trimPath, searchPath string) string {
	// Keep path variable intact as it's used below to form the return value.
	sPath, searchPath := filepath.ToSlash(path), filepath.ToSlash(searchPath)
	if trimPath == "" {
		// If the trim path is not configured, try to guess it heuristically:
		// search for basename of each search path in the original path and, if
		// found, strip everything up to and including the basename. So, for
		// example, given original path "/some/remote/path/my-project/foo/bar.c"
		// and search path "/my/local/path/my-project" the heuristic will return
		// "/my/local/path/my-project/foo/bar.c".
		for _, dir := range filepath.SplitList(searchPath) {
			want := "/" + filepath.Base(dir) + "/"
			if found := strings.Index(sPath, want); found != -1 {
				return path[found+len(want):]
			}
		}
	}
	// Trim configured trim prefixes.
	trimPaths := append(filepath.SplitList(filepath.ToSlash(trimPath)), "/proc/self/cwd/./", "/proc/self/cwd/")
	for _, trimPath := range trimPaths {
		if !strings.HasSuffix(trimPath, "/") {
			trimPath += "/"
		}
		if strings.HasPrefix(sPath, trimPath) {
			return path[len(trimPath):]
		}
	}
	return path
}

func indentation(line string) int {
	column := 0
	for _, c := range line {
		if c == ' ' {
			column++
		} else if c == '\t' {
			column++
			for column%8 != 0 {
				column++
			}
		} else {
			break
		}
	}
	return column
}
