package buildtag_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/buildtag"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	// loads testdata/src/a/a.go
	analysistest.Run(t, testdata, buildtag.Analyzer, "a")
}
