// The nilness command applies the golang.org/x/tools/go/analysis/passes/lostcancel
// analysis to the specified packages of Go source code.
package main

import (
	"golang.org/x/tools/go/analysis/passes/lostcancel"
	"golang.org/x/tools/go/analysis/singlechecker"
)

func main() { singlechecker.Main(lostcancel.Analyzer) }
