package gccgoexportdata_test

import (
	"go/types"
	"os"
	"testing"

	"golang.org/x/tools/go/gccgoexportdata"
)

// Test ensures this package can read gccgo export data from the
// .go_export from a standalone ELF file or such a file in an archive
// library.
//
// The testdata/{short,long}.a ELF archive files were produced by:
//
//   $ echo 'package foo; func F()' > foo.go
//   $ gccgo -c -fgo-pkgpath blah foo.go
//   $ objcopy -j .go_export foo.o foo.gox
//   $ ar q short.a foo.gox
//   $ objcopy -j .go_export foo.o name-longer-than-16-bytes.gox
//   $ ar q long.a name-longer-than-16-bytes.gox
//
// The file long.a contains an archive string table.
//
// The errors.gox file (an ELF object file) comes from the toolchain's
// standard library.
func Test(t *testing.T) {
	for _, test := range []struct {
		filename, path, member, wantType string
	}{
		{"testdata/errors.gox", "errors", "New", "func(text string) error"},
		{"testdata/short.a", "short", "F", "func()"},
		{"testdata/long.a", "long", "F", "func()"},
	} {
		t.Logf("filename = %s", test.filename)
		f, err := os.Open(test.filename)
		if err != nil {
			t.Error(err)
			continue
		}
		defer f.Close()
		r, err := gccgoexportdata.NewReader(f)
		if err != nil {
			t.Error(err)
			continue
		}

		imports := make(map[string]*types.Package)
		pkg, err := gccgoexportdata.Read(r, nil, imports, test.path)
		if err != nil {
			t.Error(err)
			continue
		}

		// Check type of designated package member.
		obj := pkg.Scope().Lookup(test.member)
		if obj == nil {
			t.Errorf("%s.%s not found", test.path, test.member)
			continue
		}
		if obj.Type().String() != test.wantType {
			t.Errorf("%s.%s.Type = %s, want %s",
				test.path, test.member, obj.Type(), test.wantType)
		}
	}
}
