package composite_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/composite"
)

func TestFromFileSystem(t *testing.T) {
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, composite.Analyzer, "a")
}
