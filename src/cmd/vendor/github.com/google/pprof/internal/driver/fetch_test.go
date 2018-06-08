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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"math/big"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/internal/binutils"
	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/proftest"
	"github.com/google/pprof/internal/symbolizer"
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
	savedHTTPGet := httpGet
	defer func() { httpGet = savedHTTPGet }()
	httpGet = stubHTTPGet

	type testcase struct {
		source, execName string
	}

	for _, tc := range []testcase{
		{path + "go.crc32.cpu", ""},
		{path + "go.nomappings.crash", "/bin/gotest.exe"},
		{"http://localhost/profile?file=cppbench.cpu", ""},
	} {
		p, _, _, err := grabProfile(&source{ExecName: tc.execName}, tc.source, nil, testObj{}, &proftest.TestUI{T: t})
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

func TestFetchWithBase(t *testing.T) {
	baseVars := pprofVariables
	defer func() { pprofVariables = baseVars }()

	type WantSample struct {
		values []int64
		labels map[string][]string
	}

	const path = "testdata/"
	type testcase struct {
		desc         string
		sources      []string
		bases        []string
		diffBases    []string
		normalize    bool
		wantSamples  []WantSample
		wantErrorMsg string
	}

	testcases := []testcase{
		{
			"not normalized base is same as source",
			[]string{path + "cppbench.contention"},
			[]string{path + "cppbench.contention"},
			nil,
			false,
			nil,
			"",
		},
		{
			"not normalized base is same as source",
			[]string{path + "cppbench.contention"},
			[]string{path + "cppbench.contention"},
			nil,
			false,
			nil,
			"",
		},
		{
			"not normalized single source, multiple base (all profiles same)",
			[]string{path + "cppbench.contention"},
			[]string{path + "cppbench.contention", path + "cppbench.contention"},
			nil,
			false,
			[]WantSample{
				{
					values: []int64{-2700, -608881724},
					labels: map[string][]string{},
				},
				{
					values: []int64{-100, -23992},
					labels: map[string][]string{},
				},
				{
					values: []int64{-200, -179943},
					labels: map[string][]string{},
				},
				{
					values: []int64{-100, -17778444},
					labels: map[string][]string{},
				},
				{
					values: []int64{-100, -75976},
					labels: map[string][]string{},
				},
				{
					values: []int64{-300, -63568134},
					labels: map[string][]string{},
				},
			},
			"",
		},
		{
			"not normalized, different base and source",
			[]string{path + "cppbench.contention"},
			[]string{path + "cppbench.small.contention"},
			nil,
			false,
			[]WantSample{
				{
					values: []int64{1700, 608878600},
					labels: map[string][]string{},
				},
				{
					values: []int64{100, 23992},
					labels: map[string][]string{},
				},
				{
					values: []int64{200, 179943},
					labels: map[string][]string{},
				},
				{
					values: []int64{100, 17778444},
					labels: map[string][]string{},
				},
				{
					values: []int64{100, 75976},
					labels: map[string][]string{},
				},
				{
					values: []int64{300, 63568134},
					labels: map[string][]string{},
				},
			},
			"",
		},
		{
			"normalized base is same as source",
			[]string{path + "cppbench.contention"},
			[]string{path + "cppbench.contention"},
			nil,
			true,
			nil,
			"",
		},
		{
			"normalized single source, multiple base (all profiles same)",
			[]string{path + "cppbench.contention"},
			[]string{path + "cppbench.contention", path + "cppbench.contention"},
			nil,
			true,
			nil,
			"",
		},
		{
			"normalized different base and source",
			[]string{path + "cppbench.contention"},
			[]string{path + "cppbench.small.contention"},
			nil,
			true,
			[]WantSample{
				{
					values: []int64{-229, -370},
					labels: map[string][]string{},
				},
				{
					values: []int64{28, 0},
					labels: map[string][]string{},
				},
				{
					values: []int64{57, 0},
					labels: map[string][]string{},
				},
				{
					values: []int64{28, 80},
					labels: map[string][]string{},
				},
				{
					values: []int64{28, 0},
					labels: map[string][]string{},
				},
				{
					values: []int64{85, 287},
					labels: map[string][]string{},
				},
			},
			"",
		},
		{
			"not normalized diff base is same as source",
			[]string{path + "cppbench.contention"},
			nil,
			[]string{path + "cppbench.contention"},
			false,
			[]WantSample{
				{
					values: []int64{2700, 608881724},
					labels: map[string][]string{},
				},
				{
					values: []int64{100, 23992},
					labels: map[string][]string{},
				},
				{
					values: []int64{200, 179943},
					labels: map[string][]string{},
				},
				{
					values: []int64{100, 17778444},
					labels: map[string][]string{},
				},
				{
					values: []int64{100, 75976},
					labels: map[string][]string{},
				},
				{
					values: []int64{300, 63568134},
					labels: map[string][]string{},
				},
				{
					values: []int64{-2700, -608881724},
					labels: map[string][]string{"pprof::base": {"true"}},
				},
				{
					values: []int64{-100, -23992},
					labels: map[string][]string{"pprof::base": {"true"}},
				},
				{
					values: []int64{-200, -179943},
					labels: map[string][]string{"pprof::base": {"true"}},
				},
				{
					values: []int64{-100, -17778444},
					labels: map[string][]string{"pprof::base": {"true"}},
				},
				{
					values: []int64{-100, -75976},
					labels: map[string][]string{"pprof::base": {"true"}},
				},
				{
					values: []int64{-300, -63568134},
					labels: map[string][]string{"pprof::base": {"true"}},
				},
			},
			"",
		},
		{
			"diff_base and base both specified",
			[]string{path + "cppbench.contention"},
			[]string{path + "cppbench.contention"},
			[]string{path + "cppbench.contention"},
			false,
			nil,
			"-base and -diff_base flags cannot both be specified",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			pprofVariables = baseVars.makeCopy()
			f := testFlags{
				stringLists: map[string][]string{
					"base":      tc.bases,
					"diff_base": tc.diffBases,
				},
				bools: map[string]bool{
					"normalize": tc.normalize,
				},
			}
			f.args = tc.sources

			o := setDefaults(&plugin.Options{
				UI:      &proftest.TestUI{T: t, AllowRx: "Local symbolization failed|Some binary filenames not available"},
				Flagset: f,
			})
			src, _, err := parseFlags(o)

			if tc.wantErrorMsg != "" {
				if err == nil {
					t.Fatalf("got nil, want error %q", tc.wantErrorMsg)
				}

				if gotErrMsg := err.Error(); gotErrMsg != tc.wantErrorMsg {
					t.Fatalf("got error %q, want error %q", gotErrMsg, tc.wantErrorMsg)
				}
				return
			}

			if err != nil {
				t.Fatalf("got error %q, want no error", err)
			}

			p, err := fetchProfiles(src, o)

			if err != nil {
				t.Fatalf("got error %q, want no error", err)
			}

			if got, want := len(p.Sample), len(tc.wantSamples); got != want {
				t.Fatalf("got %d samples want %d", got, want)
			}

			for i, sample := range p.Sample {
				if !reflect.DeepEqual(tc.wantSamples[i].values, sample.Value) {
					t.Errorf("for sample %d got values %v, want %v", i, sample.Value, tc.wantSamples[i])
				}
				if !reflect.DeepEqual(tc.wantSamples[i].labels, sample.Label) {
					t.Errorf("for sample %d got labels %v, want %v", i, sample.Label, tc.wantSamples[i].labels)
				}
			}
		})
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

func closedError() string {
	if runtime.GOOS == "plan9" {
		return "listen hungup"
	}
	return "use of closed"
}

func TestHttpsInsecure(t *testing.T) {
	if runtime.GOOS == "nacl" || runtime.GOOS == "js" {
		t.Skip("test assumes tcp available")
	}
	saveHome := os.Getenv(homeEnv())
	tempdir, err := ioutil.TempDir("", "home")
	if err != nil {
		t.Fatal("creating temp dir: ", err)
	}
	defer os.RemoveAll(tempdir)

	// pprof writes to $HOME/pprof by default which is not necessarily
	// writeable (e.g. on a Debian buildd) so set $HOME to something we
	// know we can write to for the duration of the test.
	os.Setenv(homeEnv(), tempdir)
	defer os.Setenv(homeEnv(), saveHome)

	baseVars := pprofVariables
	pprofVariables = baseVars.makeCopy()
	defer func() { pprofVariables = baseVars }()

	tlsConfig := &tls.Config{Certificates: []tls.Certificate{selfSignedCert(t)}}

	l, err := tls.Listen("tcp", "localhost:0", tlsConfig)
	if err != nil {
		t.Fatalf("net.Listen: got error %v, want no error", err)
	}

	donec := make(chan error, 1)
	go func(donec chan<- error) {
		donec <- http.Serve(l, nil)
	}(donec)
	defer func() {
		if got, want := <-donec, closedError(); !strings.Contains(got.Error(), want) {
			t.Fatalf("Serve got error %v, want %q", got, want)
		}
	}()
	defer l.Close()

	outputTempFile, err := ioutil.TempFile("", "profile_output")
	if err != nil {
		t.Fatalf("Failed to create tempfile: %v", err)
	}
	defer os.Remove(outputTempFile.Name())
	defer outputTempFile.Close()

	address := "https+insecure://" + l.Addr().String() + "/debug/pprof/goroutine"
	s := &source{
		Sources:   []string{address},
		Seconds:   10,
		Timeout:   10,
		Symbolize: "remote",
	}
	o := &plugin.Options{
		Obj: &binutils.Binutils{},
		UI:  &proftest.TestUI{T: t, AllowRx: "Saved profile in"},
	}
	o.Sym = &symbolizer.Symbolizer{Obj: o.Obj, UI: o.UI}
	p, err := fetchProfiles(s, o)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.SampleType) == 0 {
		t.Fatalf("fetchProfiles(%s) got empty profile: len(p.SampleType)==0", address)
	}
	if len(p.Function) == 0 {
		t.Fatalf("fetchProfiles(%s) got non-symbolized profile: len(p.Function)==0", address)
	}
	if err := checkProfileHasFunction(p, "TestHttpsInsecure"); err != nil {
		t.Fatalf("fetchProfiles(%s) %v", address, err)
	}
}

func checkProfileHasFunction(p *profile.Profile, fname string) error {
	for _, f := range p.Function {
		if strings.Contains(f.Name, fname) {
			return nil
		}
	}
	return fmt.Errorf("got %s, want function %q", p.String(), fname)
}

func selfSignedCert(t *testing.T) tls.Certificate {
	privKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("failed to generate private key: %v", err)
	}
	b, err := x509.MarshalECPrivateKey(privKey)
	if err != nil {
		t.Fatalf("failed to marshal private key: %v", err)
	}
	bk := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: b})

	tmpl := x509.Certificate{
		SerialNumber: big.NewInt(1),
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(10 * time.Minute),
	}

	b, err = x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, privKey.Public(), privKey)
	if err != nil {
		t.Fatalf("failed to create cert: %v", err)
	}
	bc := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: b})

	cert, err := tls.X509KeyPair(bc, bk)
	if err != nil {
		t.Fatalf("failed to create TLS key pair: %v", err)
	}
	return cert
}
