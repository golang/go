package load

import (
	"cmd/go/internal/cfg"
	"testing"
)

func TestPkgDefaultExecName(t *testing.T) {
	oldModulesEnabled := cfg.ModulesEnabled
	defer func() { cfg.ModulesEnabled = oldModulesEnabled }()
	for _, tt := range []struct {
		in         string
		files      []string
		wantMod    string
		wantGopath string
	}{
		{"example.com/mycmd", []string{}, "mycmd", "mycmd"},
		{"example.com/mycmd/v0", []string{}, "v0", "v0"},
		{"example.com/mycmd/v1", []string{}, "v1", "v1"},
		{"example.com/mycmd/v2", []string{}, "mycmd", "v2"}, // Semantic import versioning, use second last element in module mode.
		{"example.com/mycmd/v3", []string{}, "mycmd", "v3"}, // Semantic import versioning, use second last element in module mode.
		{"mycmd", []string{}, "mycmd", "mycmd"},
		{"mycmd/v0", []string{}, "v0", "v0"},
		{"mycmd/v1", []string{}, "v1", "v1"},
		{"mycmd/v2", []string{}, "mycmd", "v2"}, // Semantic import versioning, use second last element in module mode.
		{"v0", []string{}, "v0", "v0"},
		{"v1", []string{}, "v1", "v1"},
		{"v2", []string{}, "v2", "v2"},
		{"command-line-arguments", []string{"output.go", "foo.go"}, "output", "output"},
	} {
		{
			cfg.ModulesEnabled = true
			pkg := new(Package)
			pkg.ImportPath = tt.in
			pkg.GoFiles = tt.files
			pkg.Internal.CmdlineFiles = len(tt.files) > 0
			gotMod := pkg.DefaultExecName()
			if gotMod != tt.wantMod {
				t.Errorf("pkg.DefaultExecName with ImportPath = %q in module mode = %v; want %v", tt.in, gotMod, tt.wantMod)
			}
		}
		{
			cfg.ModulesEnabled = false
			pkg := new(Package)
			pkg.ImportPath = tt.in
			pkg.GoFiles = tt.files
			pkg.Internal.CmdlineFiles = len(tt.files) > 0
			gotGopath := pkg.DefaultExecName()
			if gotGopath != tt.wantGopath {
				t.Errorf("pkg.DefaultExecName with ImportPath = %q in gopath mode = %v; want %v", tt.in, gotGopath, tt.wantGopath)
			}
		}
	}
}

func TestIsVersionElement(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		in   string
		want bool
	}{
		{"v0", false},
		{"v05", false},
		{"v1", false},
		{"v2", true},
		{"v3", true},
		{"v9", true},
		{"v10", true},
		{"v11", true},
		{"v", false},
		{"vx", false},
	} {
		got := isVersionElement(tt.in)
		if got != tt.want {
			t.Errorf("isVersionElement(%q) = %v; want %v", tt.in, got, tt.want)
		}
	}
}
