// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/vcs"

	"context"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
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

func Test_setBuildInfo(t *testing.T) {
	sharedPackage := Package{
		PackagePublic: PackagePublic{
			Dir:        "/home/user/go/pkg/mod/example.com/shared/pkg@v2.28.4/src",
			Root:       "/home/user/go/pkg/mod/example.com/shared/pkg@v2.28.4",
			ImportPath: "example.com/shared/pkg",
			Name:       "pkg",
			Module: &modinfo.ModulePublic{
				Path:    "example.com/shared/pkg",
				Version: "v2.28.4",
			},
		},
	}

	sanitize := func(input string) string {
		re := regexp.MustCompile(`build\s*GO([A-Z]+64|ARM|386|MIPS|WASM).*=.*\s*`)
		input = re.ReplaceAllString(input, ``)
		return input
	}

	for _, tt := range []struct {
		name         string
		buildContext map[string]string
		pkg          Package
		autoVCS      bool
		want         string
	}{
		{
			name: "basic",
			buildContext: map[string]string{
				"arch":      "foo",
				"os":        "bar",
				"compiler":  "baz",
				"cgo":       "false",
				"buildmode": "hello",
			},
			pkg:     Package{},
			autoVCS: false,
			want: `build	-buildmode=hello
build	-compiler=baz
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "CGO enabled, no trim path",
			buildContext: map[string]string{
				"arch":         "foo",
				"os":           "bar",
				"compiler":     "baz",
				"cgo":          "true",
				"cgo_cflags":   "-my-c-flag",
				"cgo_cppflags": "-my-first-cpp-flag --flag2 true",
				"cgo_cxxflags": "-flag",
				"cgo_ldflags":  "-X -Z \"test\"",
			},
			pkg:     Package{},
			autoVCS: false,
			want: `build	-buildmode=
build	-compiler=baz
build	CGO_ENABLED=1
build	CGO_CFLAGS=-my-c-flag
build	CGO_CPPFLAGS="-my-first-cpp-flag --flag2 true"
build	CGO_CXXFLAGS=-flag
build	CGO_LDFLAGS="-X -Z \"test\""
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "CGO enabled, with trim path",
			buildContext: map[string]string{
				"arch":     "foo",
				"os":       "bar",
				"compiler": "baz",
				"cgo":      "true",
				"trimpath": "true",
			},
			pkg:     Package{},
			autoVCS: false,
			want: `build	-buildmode=
build	-compiler=baz
build	-trimpath=true
build	CGO_ENABLED=1
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "extra build settings",
			buildContext: map[string]string{
				"arch":          "foo",
				"os":            "bar",
				"compiler":      "baz",
				"cgo":           "false",
				"buildcover":    "true",
				"buildasan":     "true",
				"buildmsan":     "true",
				"buildrace":     "true",
				"buildasmflags": "-flag1, -flag2",
			},
			pkg:     Package{},
			autoVCS: false,
			want: `build	-asan=true
build	-asmflags="-flag1, -flag2"
build	-buildmode=
build	-compiler=baz
build	-cover=true
build	-msan=true
build	-race=true
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "tags",
			buildContext: map[string]string{
				"arch":     "foo",
				"os":       "bar",
				"compiler": "baz",
				"cgo":      "false",
				"tags":     "foo,bar,baz",
			},
			pkg:     Package{},
			autoVCS: false,
			want: `build	-buildmode=
build	-compiler=baz
build	-tags=foo,bar,baz
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{
			name: "ldflags simple",
			buildContext: map[string]string{
				"arch":     "foo",
				"os":       "bar",
				"compiler": "baz",
				"cgo":      "false",
				"ldflags":  "-flag1 -flag2",
			},
			pkg:     Package{},
			autoVCS: true,
			want: `build	-buildmode=
build	-compiler=baz
build	-ldflags="-flag1 -flag2"
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{
			name: "ldflags with trimpath",
			buildContext: map[string]string{
				"arch":     "foo",
				"os":       "bar",
				"compiler": "baz",
				"cgo":      "false",
				"ldflags":  "-flag1 -flag2",
				"trimpath": "true",
			},
			pkg:     Package{},
			autoVCS: true,
			want: `build	-buildmode=
build	-compiler=baz
build	-ldflags="-flag1 -flag2"
build	-trimpath=true
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "default build mode exe",
			buildContext: map[string]string{
				"arch":      "foo",
				"os":        "bar",
				"compiler":  "baz",
				"cgo":       "false",
				"buildmode": "default",
			},
			pkg: Package{
				PackagePublic: PackagePublic{
					Name: "main",
				},
			},
			autoVCS: false,
			want: `build	-buildmode=exe
build	-compiler=baz
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "default build mode exe",
			buildContext: map[string]string{
				"arch":      "foo",
				"os":        "bar",
				"compiler":  "baz",
				"cgo":       "false",
				"buildmode": "default",
			},
			pkg: Package{
				PackagePublic: PackagePublic{
					Name: "test",
				},
			},
			autoVCS: false,
			want: `build	-buildmode=archive
build	-compiler=baz
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "pkg path import",
			buildContext: map[string]string{
				"arch":     "foo",
				"os":       "bar",
				"compiler": "baz",
				"cgo":      "false",
			},
			pkg: Package{
				PackagePublic: PackagePublic{
					ImportPath: "github.com/foo/bar",
				},
			},
			autoVCS: false,
			want: `path	github.com/foo/bar
build	-buildmode=
build	-compiler=baz
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "pkg path from file",
			buildContext: map[string]string{
				"arch":     "foo",
				"os":       "bar",
				"compiler": "baz",
				"cgo":      "false",
			},
			pkg: Package{
				PackagePublic: PackagePublic{
					ImportPath: "github.com/foo/bar",
				},
				Internal: PackageInternal{
					CmdlineFiles: true,
				},
			},
			autoVCS: false,
			want: `path	command-line-arguments
build	-buildmode=
build	-compiler=baz
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "imports",
			buildContext: map[string]string{
				"arch":     "foo",
				"os":       "bar",
				"compiler": "baz",
				"cgo":      "false",
			},
			pkg: Package{
				PackagePublic: PackagePublic{
					Dir:        "/home/user/module",
					ImportPath: "github.com/foo/bar",
					Name:       "main",
					Imports: []string{
						"os",
						"fmt",
						"github.com/foo/bar",
						"github.com/abc/def",
					},
					Module: &modinfo.ModulePublic{
						Path:    "github.com/foo/bar",
						Version: "",
						Main:    true,
					},
				},
				Internal: PackageInternal{
					Imports: []*Package{
						&Package{
							PackagePublic: PackagePublic{
								Dir:        "/gopath/src/os",
								Root:       "/gopath",
								ImportPath: "os",
								Name:       "os",
							},
						},
						&Package{
							PackagePublic: PackagePublic{
								Dir:        "/home/user/go/pkg/mod/github.com/baz/hello@v1.2.3/src",
								Root:       "/home/user/go/pkg/mod/github.com/baz/hello@v1.2.3",
								ImportPath: "github.com/baz/hello",
								Name:       "hello",
								Module: &modinfo.ModulePublic{
									Path:    "github.com/baz/hello",
									Version: "v1.2.3",
								},
							},
							Internal: PackageInternal{
								Imports: []*Package{
									&sharedPackage,
								},
							},
						},
						&sharedPackage,
					},
				},
			},
			autoVCS: false,
			want: `path	github.com/foo/bar
mod	github.com/foo/bar	(devel)	
dep	example.com/shared/pkg	v2.28.4	
dep	github.com/baz/hello	v1.2.3	
build	-buildmode=
build	-compiler=baz
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
`,
		},
		{

			name: "vcs",
			buildContext: map[string]string{
				"arch":     "foo",
				"os":       "bar",
				"compiler": "baz",
				"cgo":      "false",
				"vcs":      "true",
			},
			pkg: Package{
				PackagePublic: PackagePublic{
					Dir:        "/home/user/module",
					ImportPath: "github.com/foo/bar",
					Name:       "main",
					Imports: []string{
						"os",
						"fmt",
						"github.com/foo/bar",
						"github.com/abc/def",
					},
					Module: &modinfo.ModulePublic{
						Path:    "github.com/foo/bar",
						Version: "",
						Main:    true,
					},
				},
				Internal: PackageInternal{
					Imports: []*Package{},
				},
			},
			autoVCS: true,
			want: `path	github.com/foo/bar
mod	github.com/foo/bar	(devel)	
build	-buildmode=
build	-compiler=baz
build	CGO_ENABLED=0
build	GOARCH=foo
build	GOOS=bar
build	vcs=git
build	vcs.revision=abcd
build	vcs.time=2012-11-10T09:06:04.000000003Z
build	vcs.modified=false
`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var currentCgoCFlags, currentCgoCPPFlags, currentCgoCXXFlags, currentCgoLDFlags string

			cfg.BuildContext.GOARCH = tt.buildContext["arch"]
			cfg.BuildContext.GOOS = tt.buildContext["os"]
			cfg.BuildContext.Compiler = tt.buildContext["compiler"]
			cfg.BuildBuildmode = tt.buildContext["buildmode"]

			var b bool
			var err error
			if tt.buildContext["vcs"] != "" {
				oldValue := cfg.BuildBuildvcs
				oldFn := FromDir
				cfg.BuildBuildvcs = tt.buildContext["vcs"]
				FromDir = func(dir, srcRoot string, allowNesting bool) (repoDir string, vcsCmd *vcs.Cmd, err error) {
					return "git", &vcs.Cmd{
						Name: "git",
						Cmd:  "git",
						Status: func(v *vcs.Cmd, rootDir string) (vcs.Status, error) {
							return vcs.Status{
								Revision:   "abcd",
								CommitTime: time.Date(2012, 11, 10, 9, 6, 4, 3, time.UTC),
							}, nil
						},
					}, nil
				}
				defer func() {
					cfg.BuildBuildvcs = oldValue
					FromDir = oldFn
				}()
			}

			if tt.buildContext["buildcover"] != "" {
				if b, err = strconv.ParseBool(tt.buildContext["buildcover"]); err == nil {
					oldValue := cfg.BuildCover
					cfg.BuildCover = b
					defer func() { cfg.BuildCover = oldValue }()
				}
			}

			if tt.buildContext["buildasan"] != "" {
				if b, err = strconv.ParseBool(tt.buildContext["buildcover"]); err == nil {
					oldValue := cfg.BuildASan
					cfg.BuildASan = b
					defer func() { cfg.BuildASan = oldValue }()
				}
			}

			if tt.buildContext["buildmsan"] != "" {
				if b, err = strconv.ParseBool(tt.buildContext["buildmsan"]); err == nil {
					oldValue := cfg.BuildMSan
					cfg.BuildMSan = b
					defer func() { cfg.BuildMSan = oldValue }()
				}
			}

			if tt.buildContext["buildrace"] != "" {
				if b, err = strconv.ParseBool(tt.buildContext["buildrace"]); err == nil {
					oldValue := cfg.BuildRace
					cfg.BuildRace = b
					defer func() { cfg.BuildRace = oldValue }()
				}
			}

			if tt.buildContext["trimpath"] != "" {
				if b, err = strconv.ParseBool(tt.buildContext["trimpath"]); err == nil {
					oldValue := cfg.BuildTrimpath
					cfg.BuildTrimpath = b
					defer func() { cfg.BuildTrimpath = oldValue }()
				}
			}

			if tt.buildContext["cgo"] != "" {
				if b, err = strconv.ParseBool(tt.buildContext["cgo"]); err == nil {
					oldValue := cfg.BuildContext.CgoEnabled
					cfg.BuildContext.CgoEnabled = b
					defer func() { cfg.BuildContext.CgoEnabled = oldValue }()
				}
			}

			if tt.buildContext["tags"] != "" {
				oldValue := cfg.BuildContext.BuildTags
				cfg.BuildContext.BuildTags = strings.Split(tt.buildContext["tags"], ",")
				defer func() { cfg.BuildContext.BuildTags = oldValue }()
			}

			if tt.buildContext["buildasmflags"] != "" {
				oldValue := BuildAsmflags
				BuildAsmflags = PerPackageFlag{
					present: true,
					raw:     "-flag1, -flag2",
				}
				defer func() { BuildAsmflags = oldValue }()
			}

			if tt.buildContext["ldflags"] != "" {
				oldValue := BuildLdflags
				BuildLdflags = PerPackageFlag{
					present: true,
					raw:     tt.buildContext["ldflags"],
				}
				defer func() { BuildLdflags = oldValue }()
			}

			if tt.buildContext["cgo_cflags"] != "" {
				currentCgoCFlags = os.Getenv("CGO_CFLAGS")
				os.Setenv("CGO_CFLAGS", tt.buildContext["cgo_cflags"])
				defer func() { os.Setenv("CGO_CFLAGS", currentCgoCFlags) }()
			}

			if tt.buildContext["cgo_cppflags"] != "" {
				currentCgoCPPFlags = os.Getenv("CGO_CPPFLAGS")
				os.Setenv("CGO_CPPFLAGS", tt.buildContext["cgo_cppflags"])
				defer func() { os.Setenv("CGO_CPPFLAGS", currentCgoCPPFlags) }()
			}

			if tt.buildContext["cgo_cxxflags"] != "" {
				currentCgoCXXFlags = os.Getenv("CGO_CXXFLAGS")
				os.Setenv("CGO_CXXFLAGS", tt.buildContext["cgo_cxxflags"])
				defer func() { os.Setenv("CGO_CXXFLAGS", currentCgoCXXFlags) }()
			}

			if tt.buildContext["cgo_ldflags"] != "" {
				currentCgoLDFlags = os.Getenv("CGO_LDFLAGS")
				os.Setenv("CGO_LDFLAGS", tt.buildContext["cgo_ldflags"])
				defer func() { os.Setenv("CGO_LDFLAGS", currentCgoLDFlags) }()
			}

			tt.pkg.setBuildInfo(context.Background(), tt.autoVCS)
			got := tt.pkg.Internal.BuildInfo

			if sanitize(fmt.Sprintf("%v", got)) != tt.want {
				t.Errorf("setBuildInfo(%q) = %v;\nwant %v", tt.name, sanitize(fmt.Sprintf("%v", got)), tt.want)
			}
		})
	}
}
