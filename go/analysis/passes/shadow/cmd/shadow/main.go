// The shadow command runs the shadow analyzer.
package main

import (
	"golang.org/x/tools/go/analysis/passes/shadow"
	"golang.org/x/tools/go/analysis/singlechecker"
)

func main() { singlechecker.Main(shadow.Analyzer) }
