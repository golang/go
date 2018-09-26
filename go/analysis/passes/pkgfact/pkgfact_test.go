package pkgfact_test

import (
	"log"
	"os"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/pkgfact"
)

func init() {
	// This test currently requires GOPATH mode.
	// Explicitly disabling module mode should suffix, but
	// we'll also turn off GOPROXY just for good measure.
	if err := os.Setenv("GO111MODULE", "off"); err != nil {
		log.Fatal(err)
	}
	if err := os.Setenv("GOPROXY", "off"); err != nil {
		log.Fatal(err)
	}
}

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, pkgfact.Analyzer,
		"c", // loads testdata/src/c/c.go.
	)
}
