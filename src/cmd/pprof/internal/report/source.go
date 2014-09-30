// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	"sort"
	"strconv"
	"strings"

	"cmd/pprof/internal/plugin"
)

// printSource prints an annotated source listing, include all
// functions with samples that match the regexp rpt.options.symbol.
// The sources are sorted by function name and then by filename to
// eliminate potential nondeterminism.
func printSource(w io.Writer, rpt *Report) error {
	o := rpt.options
	g, err := newGraph(rpt)
	if err != nil {
		return err
	}

	// Identify all the functions that match the regexp provided.
	// Group nodes for each matching function.
	var functions nodes
	functionNodes := make(map[string]nodes)
	for _, n := range g.ns {
		if !o.Symbol.MatchString(n.info.name) {
			continue
		}
		if functionNodes[n.info.name] == nil {
			functions = append(functions, n)
		}
		functionNodes[n.info.name] = append(functionNodes[n.info.name], n)
	}
	functions.sort(nameOrder)

	fmt.Fprintf(w, "Total: %s\n", rpt.formatValue(rpt.total))
	for _, fn := range functions {
		name := fn.info.name

		// Identify all the source files associated to this function.
		// Group nodes for each source file.
		var sourceFiles nodes
		fileNodes := make(map[string]nodes)
		for _, n := range functionNodes[name] {
			if n.info.file == "" {
				continue
			}
			if fileNodes[n.info.file] == nil {
				sourceFiles = append(sourceFiles, n)
			}
			fileNodes[n.info.file] = append(fileNodes[n.info.file], n)
		}

		if len(sourceFiles) == 0 {
			fmt.Printf("No source information for %s\n", name)
			continue
		}

		sourceFiles.sort(fileOrder)

		// Print each file associated with this function.
		for _, fl := range sourceFiles {
			filename := fl.info.file
			fns := fileNodes[filename]
			flatSum, cumSum := sumNodes(fns)

			fnodes, path, err := getFunctionSource(name, filename, fns, 0, 0)
			fmt.Fprintf(w, "ROUTINE ======================== %s in %s\n", name, path)
			fmt.Fprintf(w, "%10s %10s (flat, cum) %s of Total\n",
				rpt.formatValue(flatSum), rpt.formatValue(cumSum),
				percentage(cumSum, rpt.total))

			if err != nil {
				fmt.Fprintf(w, " Error: %v\n", err)
				continue
			}

			for _, fn := range fnodes {
				fmt.Fprintf(w, "%10s %10s %6d:%s\n", valueOrDot(fn.flat, rpt), valueOrDot(fn.cum, rpt), fn.info.lineno, fn.info.name)
			}
		}
	}
	return nil
}

// printWebSource prints an annotated source listing, include all
// functions with samples that match the regexp rpt.options.symbol.
func printWebSource(w io.Writer, rpt *Report, obj plugin.ObjTool) error {
	o := rpt.options
	g, err := newGraph(rpt)
	if err != nil {
		return err
	}

	// If the regexp source can be parsed as an address, also match
	// functions that land on that address.
	var address *uint64
	if hex, err := strconv.ParseUint(o.Symbol.String(), 0, 64); err == nil {
		address = &hex
	}

	// Extract interesting symbols from binary files in the profile and
	// classify samples per symbol.
	symbols := symbolsFromBinaries(rpt.prof, g, o.Symbol, address, obj)
	symNodes := nodesPerSymbol(g.ns, symbols)

	// Sort symbols for printing.
	var syms objSymbols
	for s := range symNodes {
		syms = append(syms, s)
	}
	sort.Sort(syms)

	if len(syms) == 0 {
		return fmt.Errorf("no samples found on routines matching: %s", o.Symbol.String())
	}

	printHeader(w, rpt)
	for _, s := range syms {
		name := s.sym.Name[0]
		// Identify sources associated to a symbol by examining
		// symbol samples. Classify samples per source file.
		var sourceFiles nodes
		fileNodes := make(map[string]nodes)
		for _, n := range symNodes[s] {
			if n.info.file == "" {
				continue
			}
			if fileNodes[n.info.file] == nil {
				sourceFiles = append(sourceFiles, n)
			}
			fileNodes[n.info.file] = append(fileNodes[n.info.file], n)
		}

		if len(sourceFiles) == 0 {
			fmt.Printf("No source information for %s\n", name)
			continue
		}

		sourceFiles.sort(fileOrder)

		// Print each file associated with this function.
		for _, fl := range sourceFiles {
			filename := fl.info.file
			fns := fileNodes[filename]

			asm := assemblyPerSourceLine(symbols, fns, filename, obj)
			start, end := sourceCoordinates(asm)

			fnodes, path, err := getFunctionSource(name, filename, fns, start, end)
			if err != nil {
				fnodes, path = getMissingFunctionSource(filename, asm, start, end)
			}

			flatSum, cumSum := sumNodes(fnodes)
			printFunctionHeader(w, name, path, flatSum, cumSum, rpt)
			for _, fn := range fnodes {
				printFunctionSourceLine(w, fn, asm[fn.info.lineno], rpt)
			}
			printFunctionClosing(w)
		}
	}
	printPageClosing(w)
	return nil
}

// sourceCoordinates returns the lowest and highest line numbers from
// a set of assembly statements.
func sourceCoordinates(asm map[int]nodes) (start, end int) {
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
func assemblyPerSourceLine(objSyms []*objSymbol, rs nodes, src string, obj plugin.ObjTool) map[int]nodes {
	assembly := make(map[int]nodes)
	// Identify symbol to use for this collection of samples.
	o := findMatchingSymbol(objSyms, rs)
	if o == nil {
		return assembly
	}

	// Extract assembly for matched symbol
	insns, err := obj.Disasm(o.sym.File, o.sym.Start, o.sym.End)
	if err != nil {
		return assembly
	}

	srcBase := filepath.Base(src)
	anodes := annotateAssembly(insns, rs, o.base)
	var lineno = 0
	for _, an := range anodes {
		if filepath.Base(an.info.file) == srcBase {
			lineno = an.info.lineno
		}
		if lineno != 0 {
			assembly[lineno] = append(assembly[lineno], an)
		}
	}

	return assembly
}

// findMatchingSymbol looks for the symbol that corresponds to a set
// of samples, by comparing their addresses.
func findMatchingSymbol(objSyms []*objSymbol, ns nodes) *objSymbol {
	for _, n := range ns {
		for _, o := range objSyms {
			if filepath.Base(o.sym.File) == n.info.objfile &&
				o.sym.Start <= n.info.address-o.base &&
				n.info.address-o.base <= o.sym.End {
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
	for _, l := range legendLabels(rpt) {
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
<pre onClick="pprof_toggle_asm()">
  Total:  %10s %10s (flat, cum) %s
`,
		template.HTMLEscapeString(name), template.HTMLEscapeString(path),
		rpt.formatValue(flatSum), rpt.formatValue(cumSum),
		percentage(cumSum, rpt.total))
}

// printFunctionSourceLine prints a source line and the corresponding assembly.
func printFunctionSourceLine(w io.Writer, fn *node, assembly nodes, rpt *Report) {
	if len(assembly) == 0 {
		fmt.Fprintf(w,
			"<span class=line> %6d</span> <span class=nop>  %10s %10s %s </span>\n",
			fn.info.lineno,
			valueOrDot(fn.flat, rpt), valueOrDot(fn.cum, rpt),
			template.HTMLEscapeString(fn.info.name))
		return
	}

	fmt.Fprintf(w,
		"<span class=line> %6d</span> <span class=deadsrc>  %10s %10s %s </span>",
		fn.info.lineno,
		valueOrDot(fn.flat, rpt), valueOrDot(fn.cum, rpt),
		template.HTMLEscapeString(fn.info.name))
	fmt.Fprint(w, "<span class=asm>")
	for _, an := range assembly {
		var fileline string
		class := "disasmloc"
		if an.info.file != "" {
			fileline = fmt.Sprintf("%s:%d", template.HTMLEscapeString(an.info.file), an.info.lineno)
			if an.info.lineno != fn.info.lineno {
				class = "unimportant"
			}
		}
		fmt.Fprintf(w, " %8s %10s %10s %8x: %-48s <span class=%s>%s</span>\n", "",
			valueOrDot(an.flat, rpt), valueOrDot(an.cum, rpt),
			an.info.address,
			template.HTMLEscapeString(an.info.name),
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

// getFunctionSource collects the sources of a function from a source
// file and annotates it with the samples in fns. Returns the sources
// as nodes, using the info.name field to hold the source code.
func getFunctionSource(fun, file string, fns nodes, start, end int) (nodes, string, error) {
	f, file, err := adjustSourcePath(file)
	if err != nil {
		return nil, file, err
	}

	lineNodes := make(map[int]nodes)

	// Collect source coordinates from profile.
	const margin = 5 // Lines before first/after last sample.
	if start == 0 {
		if fns[0].info.startLine != 0 {
			start = fns[0].info.startLine
		} else {
			start = fns[0].info.lineno - margin
		}
	} else {
		start -= margin
	}
	if end == 0 {
		end = fns[0].info.lineno
	}
	end += margin
	for _, n := range fns {
		lineno := n.info.lineno
		nodeStart := n.info.startLine
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

	var src nodes
	buf := bufio.NewReader(f)
	lineno := 1
	for {
		line, err := buf.ReadString('\n')
		if err != nil {
			if line == "" || err != io.EOF {
				return nil, file, err
			}
		}
		if lineno >= start {
			flat, cum := sumNodes(lineNodes[lineno])

			src = append(src, &node{
				info: nodeInfo{
					name:   strings.TrimRight(line, "\n"),
					lineno: lineno,
				},
				flat: flat,
				cum:  cum,
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
func getMissingFunctionSource(filename string, asm map[int]nodes, start, end int) (nodes, string) {
	var fnodes nodes
	for i := start; i <= end; i++ {
		lrs := asm[i]
		if len(lrs) == 0 {
			continue
		}
		flat, cum := sumNodes(lrs)
		fnodes = append(fnodes, &node{
			info: nodeInfo{
				name:   "???",
				lineno: i,
			},
			flat: flat,
			cum:  cum,
		})
	}
	return fnodes, filename
}

// adjustSourcePath adjusts the pathe for a source file by trimmming
// known prefixes and searching for the file on all parents of the
// current working dir.
func adjustSourcePath(path string) (*os.File, string, error) {
	path = trimPath(path)
	f, err := os.Open(path)
	if err == nil {
		return f, path, nil
	}

	if dir, wderr := os.Getwd(); wderr == nil {
		for {
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			if f, err := os.Open(filepath.Join(parent, path)); err == nil {
				return f, filepath.Join(parent, path), nil
			}

			dir = parent
		}
	}

	return nil, path, err
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
