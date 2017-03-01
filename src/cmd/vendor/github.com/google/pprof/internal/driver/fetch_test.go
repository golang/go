// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package driver

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"testing"
	"time"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/proftest"
	"github.com/google/pprof/profile"
)

func TestSymbolizationPath(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test assumes Unix paths")
	}

	// Save environment variables to restore after test
	saveHome := os.Getenv(homeEnv())
	savePath := os.Getenv("PPROF_BINARY_PATH")

	tempdir, err := ioutil.TempDir("", "home")
	if err != nil {
		t.Fatal("creating temp dir: ", err)
	}
	defer os.RemoveAll(tempdir)
	os.MkdirAll(filepath.Join(tempdir, "pprof", "binaries", "abcde10001"), 0700)
	os.Create(filepath.Join(tempdir, "pprof", "binaries", "abcde10001", "binary"))

	obj := testObj{tempdir}
	os.Setenv(homeEnv(), tempdir)
	for _, tc := range []struct {
		env, file, buildID, want string
		msgCount                 int
	}{
		{"", "/usr/bin/binary", "", "/usr/bin/binary", 0},
		{"", "/usr/bin/binary", "fedcb10000", "/usr/bin/binary", 0},
		{"/usr", "/bin/binary", "", "/usr/bin/binary", 0},
		{"", "/prod/path/binary", "abcde10001", filepath.Join(tempdir, "pprof/binaries/abcde10001/binary"), 0},
		{"/alternate/architecture", "/usr/bin/binary", "", "/alternate/architecture/binary", 0},
		{"/alternate/architecture", "/usr/bin/binary", "abcde10001", "/alternate/architecture/binary", 0},
		{"/nowhere:/alternate/architecture", "/usr/bin/binary", "fedcb10000", "/usr/bin/binary", 1},
		{"/nowhere:/alternate/architecture", "/usr/bin/binary", "abcde10002", "/usr/bin/binary", 1},
	} {
		os.Setenv("PPROF_BINARY_PATH", tc.env)
		p := &profile.Profile{
			Mapping: []*profile.Mapping{
				{
					File:    tc.file,
					BuildID: tc.buildID,
				},
			},
		}
		s := &source{}
		locateBinaries(p, s, obj, &proftest.TestUI{T: t, Ignore: tc.msgCount})
		if file := p.Mapping[0].File; file != tc.want {
			t.Errorf("%s:%s:%s, want %s, got %s", tc.env, tc.file, tc.buildID, tc.want, file)
		}
	}
	os.Setenv(homeEnv(), saveHome)
	os.Setenv("PPROF_BINARY_PATH", savePath)
}

func TestCollectMappingSources(t *testing.T) {
	const startAddress uint64 = 0x40000
	const url = "http://example.com"
	for _, tc := range []struct {
		file, buildID string
		want          plugin.MappingSources
	}{
		{"/usr/bin/binary", "buildId", mappingSources("buildId", url, startAddress)},
		{"/usr/bin/binary", "", mappingSources("/usr/bin/binary", url, startAddress)},
		{"", "", mappingSources(url, url, startAddress)},
	} {
		p := &profile.Profile{
			Mapping: []*profile.Mapping{
				{
					File:    tc.file,
					BuildID: tc.buildID,
					Start:   startAddress,
				},
			},
		}
		got := collectMappingSources(p, url)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s:%s, want %v, got %v", tc.file, tc.buildID, tc.want, got)
		}
	}
}

func TestUnsourceMappings(t *testing.T) {
	for _, tc := range []struct {
		file, buildID, want string
	}{
		{"/usr/bin/binary", "buildId", "/usr/bin/binary"},
		{"http://example.com", "", ""},
	} {
		p := &profile.Profile{
			Mapping: []*profile.Mapping{
				{
					File:    tc.file,
					BuildID: tc.buildID,
				},
			},
		}
		unsourceMappings(p)
		if got := p.Mapping[0].File; got != tc.want {
			t.Errorf("%s:%s, want %s, got %s", tc.file, tc.buildID, tc.want, got)
		}
	}
}

type testObj struct {
	home string
}

func (o testObj) Open(file string, start, limit, offset uint64) (plugin.ObjFile, error) {
	switch file {
	case "/alternate/architecture/binary":
		return testFile{file, "abcde10001"}, nil
	case "/usr/bin/binary":
		return testFile{file, "fedcb10000"}, nil
	case filepath.Join(o.home, "pprof/binaries/abcde10001/binary"):
		return testFile{file, "abcde10001"}, nil
	}
	return nil, fmt.Errorf("not found: %s", file)
}
func (testObj) Demangler(_ string) func(names []string) (map[string]string, error) {
	return func(names []string) (map[string]string, error) { return nil, nil }
}
func (testObj) Disasm(file string, start, end uint64) ([]plugin.Inst, error) { return nil, nil }

type testFile struct{ name, buildID string }

func (f testFile) Name() string                                               { return f.name }
func (testFile) Base() uint64                                                 { return 0 }
func (f testFile) BuildID() string                                            { return f.buildID }
func (testFile) SourceLine(addr uint64) ([]plugin.Frame, error)               { return nil, nil }
func (testFile) Symbols(r *regexp.Regexp, addr uint64) ([]*plugin.Sym, error) { return nil, nil }
func (testFile) Close() error                                                 { return nil }

func TestFetch(t *testing.T) {
	const path = "testdata/"

	// Intercept http.Get calls from HTTPFetcher.
	httpGet = stubHTTPGet

	type testcase struct {
		source, execName string
	}

	for _, tc := range []testcase{
		{path + "go.crc32.cpu", ""},
		{path + "go.nomappings.crash", "/bin/gotest.exe"},
		{"http://localhost/profile?file=cppbench.cpu", ""},
	} {
		p, _, _, err := grabProfile(&source{ExecName: tc.execName}, tc.source, 0, nil, testObj{}, &proftest.TestUI{T: t})
		if err != nil {
			t.Fatalf("%s: %s", tc.source, err)
		}
		if len(p.Sample) == 0 {
			t.Errorf("%s: want non-zero samples", tc.source)
		}
		if e := tc.execName; e != "" {
			switch {
			case len(p.Mapping) == 0 || p.Mapping[0] == nil:
				t.Errorf("%s: want mapping[0].execName == %s, got no mappings", tc.source, e)
			case p.Mapping[0].File != e:
				t.Errorf("%s: want mapping[0].execName == %s, got %s", tc.source, e, p.Mapping[0].File)
			}
		}
	}
}

// mappingSources creates MappingSources map with a single item.
func mappingSources(key, source string, start uint64) plugin.MappingSources {
	return plugin.MappingSources{
		key: []struct {
			Source string
			Start  uint64
		}{
			{Source: source, Start: start},
		},
	}
}

// stubHTTPGet intercepts a call to http.Get and rewrites it to use
// "file://" to get the profile directly from a file.
func stubHTTPGet(source string, _ time.Duration) (*http.Response, error) {
	url, err := url.Parse(source)
	if err != nil {
		return nil, err
	}

	values := url.Query()
	file := values.Get("file")

	if file == "" {
		return nil, fmt.Errorf("want .../file?profile, got %s", source)
	}

	t := &http.Transport{}
	t.RegisterProtocol("file", http.NewFileTransport(http.Dir("testdata/")))

	c := &http.Client{Transport: t}
	return c.Get("file:///" + file)
}
