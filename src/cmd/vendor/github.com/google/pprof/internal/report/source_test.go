package report

import (
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"github.com/google/pprof/internal/binutils"
	"github.com/google/pprof/profile"
)

func TestWebList(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skip("weblist only tested on x86-64 linux")
	}

	cpu := readProfile(filepath.Join("testdata", "sample.cpu"), t)
	rpt := New(cpu, &Options{
		OutputFormat: WebList,
		Symbol:       regexp.MustCompile("busyLoop"),
		SampleValue:  func(v []int64) int64 { return v[1] },
		SampleUnit:   cpu.SampleType[1].Unit,
	})
	var buf bytes.Buffer
	if err := Generate(&buf, rpt, &binutils.Binutils{}); err != nil {
		t.Fatalf("could not generate weblist: %v", err)
	}
	output := buf.String()

	for _, expect := range []string{"func busyLoop", "callq", "math.Abs"} {
		if !strings.Contains(output, expect) {
			t.Errorf("weblist output does not contain '%s':\n%s", expect, output)
		}
	}
}

func TestOpenSourceFile(t *testing.T) {
	tempdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	const lsep = string(filepath.ListSeparator)
	for _, tc := range []struct {
		desc       string
		searchPath string
		trimPath   string
		fs         []string
		path       string
		wantPath   string // If empty, error is wanted.
	}{
		{
			desc:     "exact absolute path is found",
			fs:       []string{"foo/bar.cc"},
			path:     "$dir/foo/bar.cc",
			wantPath: "$dir/foo/bar.cc",
		},
		{
			desc:       "exact relative path is found",
			searchPath: "$dir",
			fs:         []string{"foo/bar.cc"},
			path:       "foo/bar.cc",
			wantPath:   "$dir/foo/bar.cc",
		},
		{
			desc:       "multiple search path",
			searchPath: "some/path" + lsep + "$dir",
			fs:         []string{"foo/bar.cc"},
			path:       "foo/bar.cc",
			wantPath:   "$dir/foo/bar.cc",
		},
		{
			desc:       "relative path is found in parent dir",
			searchPath: "$dir/foo/bar",
			fs:         []string{"bar.cc", "foo/bar/baz.cc"},
			path:       "bar.cc",
			wantPath:   "$dir/bar.cc",
		},
		{
			desc:       "trims configured prefix",
			searchPath: "$dir",
			trimPath:   "some-path" + lsep + "/some/remote/path",
			fs:         []string{"my-project/foo/bar.cc"},
			path:       "/some/remote/path/my-project/foo/bar.cc",
			wantPath:   "$dir/my-project/foo/bar.cc",
		},
		{
			desc:       "trims heuristically",
			searchPath: "$dir/my-project",
			fs:         []string{"my-project/foo/bar.cc"},
			path:       "/some/remote/path/my-project/foo/bar.cc",
			wantPath:   "$dir/my-project/foo/bar.cc",
		},
		{
			desc: "error when not found",
			path: "foo.cc",
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			defer func() {
				if err := os.RemoveAll(tempdir); err != nil {
					t.Fatalf("failed to remove dir %q: %v", tempdir, err)
				}
			}()
			for _, f := range tc.fs {
				path := filepath.Join(tempdir, filepath.FromSlash(f))
				dir := filepath.Dir(path)
				if err := os.MkdirAll(dir, 0755); err != nil {
					t.Fatalf("failed to create dir %q: %v", dir, err)
				}
				if err := ioutil.WriteFile(path, nil, 0644); err != nil {
					t.Fatalf("failed to create file %q: %v", path, err)
				}
			}
			tc.searchPath = filepath.FromSlash(strings.Replace(tc.searchPath, "$dir", tempdir, -1))
			tc.path = filepath.FromSlash(strings.Replace(tc.path, "$dir", tempdir, 1))
			tc.wantPath = filepath.FromSlash(strings.Replace(tc.wantPath, "$dir", tempdir, 1))
			if file, err := openSourceFile(tc.path, tc.searchPath, tc.trimPath); err != nil && tc.wantPath != "" {
				t.Errorf("openSourceFile(%q, %q, %q) = err %v, want path %q", tc.path, tc.searchPath, tc.trimPath, err, tc.wantPath)
			} else if err == nil {
				defer file.Close()
				gotPath := file.Name()
				if tc.wantPath == "" {
					t.Errorf("openSourceFile(%q, %q, %q) = %q, want error", tc.path, tc.searchPath, tc.trimPath, gotPath)
				} else if gotPath != tc.wantPath {
					t.Errorf("openSourceFile(%q, %q, %q) = %q, want path %q", tc.path, tc.searchPath, tc.trimPath, gotPath, tc.wantPath)
				}
			}
		})
	}
}

func TestIndentation(t *testing.T) {
	for _, c := range []struct {
		str        string
		wantIndent int
	}{
		{"", 0},
		{"foobar", 0},
		{"  foo", 2},
		{"\tfoo", 8},
		{"\t foo", 9},
		{"  \tfoo", 8},
		{"       \tfoo", 8},
		{"        \tfoo", 16},
	} {
		if n := indentation(c.str); n != c.wantIndent {
			t.Errorf("indentation(%v): got %d, want %d", c.str, n, c.wantIndent)
		}
	}
}

func readProfile(fname string, t *testing.T) *profile.Profile {
	file, err := os.Open(fname)
	if err != nil {
		t.Fatalf("%s: could not open profile: %v", fname, err)
	}
	defer file.Close()
	p, err := profile.Parse(file)
	if err != nil {
		t.Fatalf("%s: could not parse profile: %v", fname, err)
	}

	// Fix file names so they do not include absolute path names.
	fix := func(s string) string {
		const testdir = "/internal/report/"
		pos := strings.Index(s, testdir)
		if pos == -1 {
			return s
		}
		return s[pos+len(testdir):]
	}
	for _, m := range p.Mapping {
		m.File = fix(m.File)
	}
	for _, f := range p.Function {
		f.Filename = fix(f.Filename)
	}

	return p
}
