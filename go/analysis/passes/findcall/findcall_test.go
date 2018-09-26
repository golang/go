package findcall_test

import (
	"log"
	"os"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/findcall"
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

// TestFromStringLiterals demonstrates how to test an analysis using
// a table of string literals for each test case.
//
// Such tests are typically quite compact.
func TestFromStringLiterals(t *testing.T) {
	for _, test := range [...]struct {
		desc    string
		pkgpath string
		files   map[string]string
	}{
		{
			desc:    "SimpleTest",
			pkgpath: "main",
			files: map[string]string{"main/main.go": `package main

func main() {
	println("hello") // want "call of println"
	print("goodbye") // not a call of println
}`,
			},
		},
	} {
		t.Run(test.desc, func(t *testing.T) {
			dir, cleanup, err := analysistest.WriteFiles(test.files)
			if err != nil {
				t.Fatal(err)
			}
			defer cleanup()
			analysistest.Run(t, dir, findcall.Analyzer, test.pkgpath)
		})
	}
}

// TestFromFileSystem demonstrates how to test an analysis using input
// files stored in the file system.
//
// These tests have the advantages that test data can be edited
// directly, and that files named in error messages can be opened.
// However, they tend to spread a small number of lines of text across a
// rather deep directory hierarchy, and obscure similarities among
// related tests, especially when tests involve multiple packages, or
// multiple variants of a single scenario.
func TestFromFileSystem(t *testing.T) {
	testdata := analysistest.TestData()
	analysistest.Run(t, testdata, findcall.Analyzer,
		"a", // loads testdata/src/a/a.go.
	)
}
