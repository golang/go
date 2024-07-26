// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildinfo_test

import (
	"bytes"
	"debug/buildinfo"
	"debug/pe"
	"encoding/binary"
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

	damageStringLen := func(t *testing.T, name string) {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		i := bytes.Index(data, []byte("\xff Go buildinf:"))
		if i < 0 {
			t.Fatal("Go buildinf not found")
		}
		verLen := data[i+32:]
		binary.PutUvarint(verLen, 16<<40) // 16TB ought to be enough for anyone.
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
			name: "invalid_str_len",
			build: func(t *testing.T, goos, goarch, buildmode string) string {
				name := buildWithModules(t, goos, goarch, buildmode)
				damageStringLen(t, name)
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
								if clean := cleanOutputForComparison(got); got != tc.want && clean != tc.want {
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

// Test117 verifies that parsing of the old, pre-1.18 format works.
func Test117(t *testing.T) {
	// go117 was generated for linux-amd64 with:
	//
	// main.go:
	//
	// package main
	// func main() {}
	//
	// GOTOOLCHAIN=go1.17 go mod init example.com/go117
	// GOTOOLCHAIN=go1.17 go build
	//
	// TODO(prattmic): Ideally this would be built on the fly to better
	// cover all executable formats, but then we need a network connection
	// to download an old Go toolchain.
	info, err := buildinfo.ReadFile("testdata/go117")
	if err != nil {
		t.Fatalf("ReadFile got err %v, want nil", err)
	}

	if info.GoVersion != "go1.17" {
		t.Errorf("GoVersion got %s want go1.17", info.GoVersion)
	}
	if info.Path != "example.com/go117" {
		t.Errorf("Path got %s want example.com/go117", info.Path)
	}
	if info.Main.Path != "example.com/go117" {
		t.Errorf("Main.Path got %s want example.com/go117", info.Main.Path)
	}
}

// TestNotGo verifies that parsing of a non-Go binary returns the proper error.
func TestNotGo(t *testing.T) {
	// notgo was generated for linux-amd64 with:
	//
	// main.c:
	//
	// int main(void) { return 0; }
	//
	// cc -o notgo main.c
	//
	// TODO(prattmic): Ideally this would be built on the fly to better
	// cover all executable formats, but then we need to encode the
	// intricacies of calling each platform's C compiler.
	_, err := buildinfo.ReadFile("testdata/notgo")
	if err == nil {
		t.Fatalf("ReadFile got nil err, want non-nil")
	}

	// The precise error text here isn't critical, but we want something
	// like errNotGoExe rather than e.g., a file read error.
	if !strings.Contains(err.Error(), "not a Go executable") {
		t.Errorf("ReadFile got err %v want not a Go executable", err)
	}
}

// FuzzIssue57002 is a regression test for golang.org/issue/57002.
//
// The cause of issue 57002 is when pointerSize is not being checked,
// the read can panic with slice bounds out of range
func FuzzIssue57002(f *testing.F) {
	// input from issue
	f.Add([]byte{0x4d, 0x5a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x50, 0x45, 0x0, 0x0, 0x0, 0x0, 0x5, 0x0, 0x20, 0x20, 0x20, 0x20, 0x0, 0x0, 0x0, 0x0, 0x20, 0x3f, 0x0, 0x20, 0x0, 0x0, 0x20, 0x20, 0x20, 0x20, 0x20, 0xff, 0x20, 0x20, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xb, 0x20, 0x20, 0x20, 0xfc, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x9, 0x0, 0x0, 0x0, 0x20, 0x0, 0x0, 0x0, 0x20, 0x20, 0x20, 0x20, 0x20, 0xef, 0x20, 0xff, 0xbf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf, 0x0, 0x2, 0x0, 0x20, 0x0, 0x0, 0x9, 0x0, 0x4, 0x0, 0x20, 0xf6, 0x0, 0xd3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x20, 0x1, 0x0, 0x0, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0xa, 0x20, 0xa, 0x20, 0x20, 0x20, 0xff, 0x20, 0x20, 0xff, 0x20, 0x47, 0x6f, 0x20, 0x62, 0x75, 0x69, 0x6c, 0x64, 0x69, 0x6e, 0x66, 0x3a, 0xde, 0xb5, 0xdf, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x6, 0x7f, 0x7f, 0x7f, 0x20, 0xf4, 0xb2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0xb, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x20, 0x20, 0x0, 0x0, 0x0, 0x0, 0x5, 0x0, 0x20, 0x20, 0x20, 0x20, 0x0, 0x0, 0x0, 0x0, 0x20, 0x3f, 0x27, 0x20, 0x0, 0xd, 0x0, 0xa, 0x20, 0x20, 0x20, 0x20, 0x20, 0xff, 0x20, 0x20, 0xff, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x0, 0x20, 0x20, 0x0, 0x0, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x5c, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20})
	f.Fuzz(func(t *testing.T, input []byte) {
		buildinfo.Read(bytes.NewReader(input))
	})
}

// TestIssue54968 is a regression test for golang.org/issue/54968.
//
// The cause of issue 54968 is when the first buildInfoMagic is invalid, it
// enters an infinite loop.
func TestIssue54968(t *testing.T) {
	t.Parallel()

	const (
		paddingSize    = 200
		buildInfoAlign = 16
	)
	buildInfoMagic := []byte("\xff Go buildinf:")

	// Construct a valid PE header.
	var buf bytes.Buffer

	buf.Write([]byte{'M', 'Z'})
	buf.Write(bytes.Repeat([]byte{0}, 0x3c-2))
	// At location 0x3c, the stub has the file offset to the PE signature.
	binary.Write(&buf, binary.LittleEndian, int32(0x3c+4))

	buf.Write([]byte{'P', 'E', 0, 0})

	binary.Write(&buf, binary.LittleEndian, pe.FileHeader{NumberOfSections: 1})

	sh := pe.SectionHeader32{
		Name:             [8]uint8{'t', 0},
		SizeOfRawData:    uint32(paddingSize + len(buildInfoMagic)),
		PointerToRawData: uint32(buf.Len()),
	}
	sh.PointerToRawData = uint32(buf.Len() + binary.Size(sh))

	binary.Write(&buf, binary.LittleEndian, sh)

	start := buf.Len()
	buf.Write(bytes.Repeat([]byte{0}, paddingSize+len(buildInfoMagic)))
	data := buf.Bytes()

	if _, err := pe.NewFile(bytes.NewReader(data)); err != nil {
		t.Fatalf("need a valid PE header for the misaligned buildInfoMagic test: %s", err)
	}

	// Place buildInfoMagic after the header.
	for i := 1; i < paddingSize-len(buildInfoMagic); i++ {
		// Test only misaligned buildInfoMagic.
		if i%buildInfoAlign == 0 {
			continue
		}

		t.Run(fmt.Sprintf("start_at_%d", i), func(t *testing.T) {
			d := data[:start]
			// Construct intentionally-misaligned buildInfoMagic.
			d = append(d, bytes.Repeat([]byte{0}, i)...)
			d = append(d, buildInfoMagic...)
			d = append(d, bytes.Repeat([]byte{0}, paddingSize-i)...)

			_, err := buildinfo.Read(bytes.NewReader(d))

			wantErr := "not a Go executable"
			if err == nil {
				t.Errorf("got error nil; want error containing %q", wantErr)
			} else if errMsg := err.Error(); !strings.Contains(errMsg, wantErr) {
				t.Errorf("got error %q; want error containing %q", errMsg, wantErr)
			}
		})
	}
}
