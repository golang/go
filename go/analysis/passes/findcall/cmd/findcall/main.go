// The findcall command runs the findcall analyzer.
package main

import (
	"golang.org/x/tools/go/analysis/passes/findcall"
	"golang.org/x/tools/go/analysis/singlechecker"
)

func main() { singlechecker.Main(findcall.Analyzer) }
