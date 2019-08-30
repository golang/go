package imports

import (
	"go/build"
	"os"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	testenv.ExitIfSmallMachine()
	os.Exit(m.Run())
}

// TestNilOpts tests that process does not crash with nil opts.
func TestNilOpts(t *testing.T) {
	var testOpts = []struct {
		name string
		opt  *Options
	}{
		{
			name: "nil",
			opt:  nil,
		},
		{
			name: "nil env",
			opt:  &Options{Comments: true, TabIndent: true, TabWidth: 8},
		},
		{
			name: "default",
			opt: &Options{
				Env: &ProcessEnv{
					GOPATH: build.Default.GOPATH,
					GOROOT: build.Default.GOROOT,
				},
				Comments:  true,
				TabIndent: true,
				TabWidth:  8,
			},
		},
	}

	input := `package p

func _() {
	fmt.Println()
}
`
	want := `package p

import "fmt"

func _() {
	fmt.Println()
}
`
	for _, test := range testOpts {
		// Test Process
		got, err := Process("", []byte(input), test.opt)
		if err != nil {
			t.Errorf("%s: %s", test.name, err.Error())
		}
		if string(got) != want {
			t.Errorf("%s: Process: Got:\n%s\nWant:\n%s\n", test.name, string(got), want)
		}

		// Test FixImports and ApplyFixes
		fixes, err := FixImports("", []byte(input), test.opt)
		if err != nil {
			t.Errorf("%s: %s", test.name, err.Error())
		}

		got, err = ApplyFixes(fixes, "", []byte(input), test.opt)
		if err != nil {
			t.Errorf("%s: %s", test.name, err.Error())
		}
		if string(got) != want {
			t.Errorf("%s: ApplyFix: Got:\n%s\nWant:\n%s\n", test.name, string(got), want)
		}
	}
}
