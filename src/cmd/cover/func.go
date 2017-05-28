// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the visitor that computes the (line, column)-(line-column) range for each function.

package main

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"text/tabwriter"
)

// funcOutput takes two file names as arguments, a coverage profile to read as input and an output
// file to write ("" means to write to standard output). The function reads the profile and produces
// as output the coverage data broken down by function, like this:
//
//	fmt/format.go:30:	init			100.0%
//	fmt/format.go:57:	clearflags		100.0%
//	...
//	fmt/scan.go:1046:	doScan			100.0%
//	fmt/scan.go:1075:	advance			96.2%
//	fmt/scan.go:1119:	doScanf			96.8%
//	total:		(statements)			91.9%

func funcOutput(profile, outputFile string) error {
	profiles, err := ParseProfiles(profile)
	if err != nil {
		return err
	}

	var out *bufio.Writer
	if outputFile == "" {
		out = bufio.NewWriter(os.Stdout)
	} else {
		fd, err := os.Create(outputFile)
		if err != nil {
			return err
		}
		defer fd.Close()
		out = bufio.NewWriter(fd)
	}
	defer out.Flush()

	tabber := tabwriter.NewWriter(out, 1, 8, 1, '\t', 0)
	defer tabber.Flush()

	var total, covered int64
	for _, profile := range profiles {
		fn := profile.FileName
		file, err := findFile(fn)
		if err != nil {
			return err
		}
		funcs, err := findFuncs(file)
		if err != nil {
			return err
		}
		// Now match up functions and profile blocks.
		for _, f := range funcs {
			c, t := f.coverage(profile)
			fmt.Fprintf(tabber, "%s:%d:\t%s\t%.1f%%\n", fn, f.startLine, f.name, percent(c, t))
			total += t
			covered += c
		}
	}
	fmt.Fprintf(tabber, "total:\t(statements)\t%.1f%%\n", percent(covered, total))

	return nil
}

// findFuncs parses the file and returns a slice of FuncExtent descriptors.
func findFuncs(name string) ([]*FuncExtent, error) {
	fset := token.NewFileSet()
	parsedFile, err := parser.ParseFile(fset, name, nil, 0)
	if err != nil {
		return nil, err
	}
	visitor := &FuncVisitor{
		fset:    fset,
		name:    name,
		astFile: parsedFile,
	}
	ast.Walk(visitor, visitor.astFile)
	return visitor.funcs, nil
}

// FuncExtent describes a function's extent in the source by file and position.
type FuncExtent struct {
	name      string
	startLine int
	startCol  int
	endLine   int
	endCol    int
}

// FuncVisitor implements the visitor that builds the function position list for a file.
type FuncVisitor struct {
	fset    *token.FileSet
	name    string // Name of file.
	astFile *ast.File
	funcs   []*FuncExtent
}

// Visit implements the ast.Visitor interface.
func (v *FuncVisitor) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.FuncDecl:
		start := v.fset.Position(n.Pos())
		end := v.fset.Position(n.End())
		fe := &FuncExtent{
			name:      n.Name.Name,
			startLine: start.Line,
			startCol:  start.Column,
			endLine:   end.Line,
			endCol:    end.Column,
		}
		v.funcs = append(v.funcs, fe)
	}
	return v
}

// coverage returns the fraction of the statements in the function that were covered, as a numerator and denominator.
func (f *FuncExtent) coverage(profile *Profile) (num, den int64) {
	// We could avoid making this n^2 overall by doing a single scan and annotating the functions,
	// but the sizes of the data structures is never very large and the scan is almost instantaneous.
	var covered, total int64
	// The blocks are sorted, so we can stop counting as soon as we reach the end of the relevant block.
	for _, b := range profile.Blocks {
		if b.StartLine > f.endLine || (b.StartLine == f.endLine && b.StartCol >= f.endCol) {
			// Past the end of the function.
			break
		}
		if b.EndLine < f.startLine || (b.EndLine == f.startLine && b.EndCol <= f.startCol) {
			// Before the beginning of the function
			continue
		}
		total += int64(b.NumStmt)
		if b.Count > 0 {
			covered += int64(b.NumStmt)
		}
	}
	return covered, total
}

// findFile finds the location of the named file in GOROOT, GOPATH etc.
func findFile(file string) (string, error) {
	dir, file := filepath.Split(file)
	pkg, err := build.Import(dir, ".", build.FindOnly)
	if err != nil {
		return "", fmt.Errorf("can't find %q: %v", file, err)
	}
	return filepath.Join(pkg.Dir, file), nil
}

func percent(covered, total int64) float64 {
	if total == 0 {
		total = 1 // Avoid zero denominator.
	}
	return 100.0 * float64(covered) / float64(total)
}
