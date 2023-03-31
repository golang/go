// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The nodecount program illustrates the use of packages.Load to print
// the frequency of occurrence of each type of syntax node among the
// selected packages.
//
// Example usage:
//
//	$ nodecount golang.org/x/tools/... std
//
// A typical distribution is 40% identifiers, 10% literals, 8%
// selectors, and 6% calls; around 3% each of BinaryExpr, BlockStmt,
// AssignStmt, Field, and Comment; and the rest accounting for 20%.
package main

import (
	"flag"
	"fmt"
	"go/ast"
	"log"
	"reflect"
	"sort"

	"golang.org/x/tools/go/packages"
)

func main() {
	flag.Parse()

	// Parse specified packages.
	config := packages.Config{
		Mode:  packages.NeedSyntax | packages.NeedFiles,
		Tests: true,
	}
	pkgs, err := packages.Load(&config, flag.Args()...)
	if err != nil {
		log.Fatal(err)
	}

	// Count each type of syntax node.
	var (
		byType = make(map[reflect.Type]int)
		total  int
	)
	packages.Visit(pkgs, nil, func(p *packages.Package) {
		for _, f := range p.Syntax {
			ast.Inspect(f, func(n ast.Node) bool {
				if n != nil {
					byType[reflect.TypeOf(n)]++
					total++
				}
				return true
			})
		}
	})

	// Print results (percent, count, type) in descending order.
	var types []reflect.Type
	for t := range byType {
		types = append(types, t)
	}
	sort.Slice(types, func(i, j int) bool {
		return byType[types[i]] > byType[types[j]]
	})
	for _, t := range types {
		percent := 100 * float64(byType[t]) / float64(total)
		fmt.Printf("%6.2f%%\t%8d\t%s\n", percent, byType[t], t)
	}
}
