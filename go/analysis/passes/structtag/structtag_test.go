package structtag_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/structtag"
)

func TestFromFileSystem(t *testing.T) {
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, structtag.Analyzer, "a")
}
