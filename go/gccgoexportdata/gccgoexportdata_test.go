package gccgoexportdata_test

import (
	"go/types"
	"os"
	"testing"

	"golang.org/x/tools/go/gccgoexportdata"
)

// Test ensures this package can read gccgo export data from the
// .go_export section of an ELF file.
func Test(t *testing.T) {
	f, err := os.Open("testdata/errors.gox")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r, err := gccgoexportdata.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	imports := make(map[string]*types.Package)
	pkg, err := gccgoexportdata.Read(r, nil, imports, "errors")
	if err != nil {

		t.Fatal(err)
	}

	// Check type of errors.New.
	got := pkg.Scope().Lookup("New").Type().String()
	want := "func(text string) error"
	if got != want {
		t.Errorf("New.Type = %s, want %s", got, want)
	}
}
