package pkgfact_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/pkgfact"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, pkgfact.Analyzer,
		"c", // loads testdata/src/c/c.go.
	)
}
