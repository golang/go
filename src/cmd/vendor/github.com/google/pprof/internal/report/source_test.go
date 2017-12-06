package report

import (
	"bytes"
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
	buf := bytes.NewBuffer(nil)
	if err := Generate(buf, rpt, &binutils.Binutils{}); err != nil {
		t.Fatalf("could not generate weblist: %v", err)
	}
	output := buf.String()

	for _, expect := range []string{"func busyLoop", "callq", "math.Abs"} {
		if !strings.Contains(output, expect) {
			t.Errorf("weblist output does not contain '%s':\n%s", expect, output)
		}
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
