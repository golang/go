package tests_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/tests"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	// Loads "a", "a [a.test]", and "a.test".
	analysistest.Run(t, testdata, tests.Analyzer, "a")
}
