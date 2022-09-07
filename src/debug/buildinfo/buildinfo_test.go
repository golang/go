// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildinfo_test

import (
	"bytes"
	"debug/buildinfo"
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

var flagAll = flag.Bool("all", false, "test all supported GOOS/GOARCH platforms, instead of only the current platform")

// TestReadFile confirms that ReadFile can read build information from binaries
// on supported target platforms. It builds a trivial binary on the current
// platforms (or all platforms if -all is set) in various configurations and
// checks that build information can or cannot be read.
func TestReadFile(t *testing.T) {
	if testing.Short() {
		t.Skip("test requires compiling and linking, which may be slow")
	}
	testenv.MustHaveGoBuild(t)

	type platform struct{ goos, goarch string }
	platforms := []platform{
		{"aix", "ppc64"},
		{"darwin", "amd64"},
		{"darwin", "arm64"},
		{"linux", "386"},
		{"linux", "amd64"},
		{"windows", "386"},
		{"windows", "amd64"},
	}
	runtimePlatform := platform{runtime.GOOS, runtime.GOARCH}
	haveRuntimePlatform := false
	for _, p := range platforms {
		if p == runtimePlatform {
			haveRuntimePlatform = true
			break
		}
	}
	if !haveRuntimePlatform {
		platforms = append(platforms, runtimePlatform)
	}

	buildModes := []string{"pie", "exe"}
	if testenv.HasCGO() {
		buildModes = append(buildModes, "c-shared")
	}

	// Keep in sync with src/cmd/go/internal/work/init.go:buildModeInit.
	badmode := func(goos, goarch, buildmode string) string {
		return fmt.Sprintf("-buildmode=%s not supported on %s/%s", buildmode, goos, goarch)
	}

	buildWithModules := func(t *testing.T, goos, goarch, buildmode string) string {
		dir := t.TempDir()
		gomodPath := filepath.Join(dir, "go.mod")
		gomodData := []byte("module example.com/m\ngo 1.18\n")
		if err := os.WriteFile(gomodPath, gomodData, 0666); err != nil {
			t.Fatal(err)
		}
		helloPath := filepath.Join(dir, "hello.go")
		helloData := []byte("package main\nfunc main() {}\n")
		if err := os.WriteFile(helloPath, helloData, 0666); err != nil {
			t.Fatal(err)
		}
		outPath := filepath.Join(dir, path.Base(t.Name()))
		cmd := exec.Command(testenv.GoToolPath(t), "build", "-o="+outPath, "-buildmode="+buildmode)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GO111MODULE=on", "GOOS="+goos, "GOARCH="+goarch)
		stderr := &strings.Builder{}
		cmd.Stderr = stderr
		if err := cmd.Run(); err != nil {
			if badmodeMsg := badmode(goos, goarch, buildmode); strings.Contains(stderr.String(), badmodeMsg) {
				t.Skip(badmodeMsg)
			}
			t.Fatalf("failed building test file: %v\n%s", err, stderr.String())
		}
		return outPath
	}

	buildWithGOPATH := func(t *testing.T, goos, goarch, buildmode string) string {
		gopathDir := t.TempDir()
		pkgDir := filepath.Join(gopathDir, "src/example.com/m")
		if err := os.MkdirAll(pkgDir, 0777); err != nil {
			t.Fatal(err)
		}
		helloPath := filepath.Join(pkgDir, "hello.go")
		helloData := []byte("package main\nfunc main() {}\n")
		if err := os.WriteFile(helloPath, helloData, 0666); err != nil {
			t.Fatal(err)
		}
		outPath := filepath.Join(gopathDir, path.Base(t.Name()))
		cmd := exec.Command(testenv.GoToolPath(t), "build", "-o="+outPath, "-buildmode="+buildmode)
		cmd.Dir = pkgDir
		cmd.Env = append(os.Environ(), "GO111MODULE=off", "GOPATH="+gopathDir, "GOOS="+goos, "GOARCH="+goarch)
		stderr := &strings.Builder{}
		cmd.Stderr = stderr
		if err := cmd.Run(); err != nil {
			if badmodeMsg := badmode(goos, goarch, buildmode); strings.Contains(stderr.String(), badmodeMsg) {
				t.Skip(badmodeMsg)
			}
			t.Fatalf("failed building test file: %v\n%s", err, stderr.String())
		}
		return outPath
	}

	damageBuildInfo := func(t *testing.T, name string) {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		i := bytes.Index(data, []byte("\xff Go buildinf:"))
		if i < 0 {
			t.Fatal("Go buildinf not found")
		}
		data[i+2] = 'N'
		if err := os.WriteFile(name, data, 0666); err != nil {
			t.Fatal(err)
		}
	}

	goVersionRe := regexp.MustCompile("(?m)^go\t.*\n")
	buildRe := regexp.MustCompile("(?m)^build\t.*\n")
	cleanOutputForComparison := func(got string) string {
		// Remove or replace anything that might depend on the test's environment
		// so we can check the output afterward with a string comparison.
		// We'll remove all build lines except the compiler, just to make sure
		// build lines are included.
		got = goVersionRe.ReplaceAllString(got, "go\tGOVERSION\n")
		got = buildRe.ReplaceAllStringFunc(got, func(match string) string {
			if strings.HasPrefix(match, "build\t-compiler=") {
				return match
			}
			return ""
		})
		return got
	}

	cases := []struct {
		name    string
		build   func(t *testing.T, goos, goarch, buildmode string) string
		want    string
		wantErr string
	}{
		{
			name: "doesnotexist",
			build: func(t *testing.T, goos, goarch, buildmode string) string {
				return "doesnotexist.txt"
			},
			wantErr: "doesnotexist",
		},
		{
			name: "empty",
			build: func(t *testing.T, _, _, _ string) string {
				dir := t.TempDir()
				name := filepath.Join(dir, "empty")
				if err := os.WriteFile(name, nil, 0666); err != nil {
					t.Fatal(err)
				}
				return name
			},
			wantErr: "unrecognized file format",
		},
		{
			name:  "valid_modules",
			build: buildWithModules,
			want: "go\tGOVERSION\n" +
				"path\texample.com/m\n" +
				"mod\texample.com/m\t(devel)\t\n" +
				"build\t-compiler=gc\n",
		},
		{
			name: "invalid_modules",
			build: func(t *testing.T, goos, goarch, buildmode string) string {
				name := buildWithModules(t, goos, goarch, buildmode)
				damageBuildInfo(t, name)
				return name
			},
			wantErr: "not a Go executable",
		},
		{
			name:  "valid_gopath",
			build: buildWithGOPATH,
			want: "go\tGOVERSION\n" +
				"path\texample.com/m\n" +
				"build\t-compiler=gc\n",
		},
		{
			name: "invalid_gopath",
			build: func(t *testing.T, goos, goarch, buildmode string) string {
				name := buildWithGOPATH(t, goos, goarch, buildmode)
				damageBuildInfo(t, name)
				return name
			},
			wantErr: "not a Go executable",
		},
	}

	for _, p := range platforms {
		p := p
		t.Run(p.goos+"_"+p.goarch, func(t *testing.T) {
			if p != runtimePlatform && !*flagAll {
				t.Skipf("skipping platforms other than %s_%s because -all was not set", runtimePlatform.goos, runtimePlatform.goarch)
			}
			for _, mode := range buildModes {
				mode := mode
				t.Run(mode, func(t *testing.T) {
					for _, tc := range cases {
						tc := tc
						t.Run(tc.name, func(t *testing.T) {
							t.Parallel()
							name := tc.build(t, p.goos, p.goarch, mode)
							if info, err := buildinfo.ReadFile(name); err != nil {
								if tc.wantErr == "" {
									t.Fatalf("unexpected error: %v", err)
								} else if errMsg := err.Error(); !strings.Contains(errMsg, tc.wantErr) {
									t.Fatalf("got error %q; want error containing %q", errMsg, tc.wantErr)
								}
							} else {
								if tc.wantErr != "" {
									t.Fatalf("unexpected success; want error containing %q", tc.wantErr)
								}
								got := info.String()
								if clean := cleanOutputForComparison(string(got)); got != tc.want && clean != tc.want {
									t.Fatalf("got:\n%s\nwant:\n%s", got, tc.want)
								}
							}
						})
					}
				})
			}
		})
	}
}
