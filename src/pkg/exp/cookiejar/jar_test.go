// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cookiejar

import (
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"testing"
	"time"
)

// testPSL implements PublicSuffixList with just two rules: "co.uk"
// and the default rule "*".
type testPSL struct{}

func (testPSL) String() string {
	return "testPSL"
}
func (testPSL) PublicSuffix(d string) string {
	if d == "co.uk" || strings.HasSuffix(d, ".co.uk") {
		return "co.uk"
	}
	return d[strings.LastIndex(d, ".")+1:]
}

// newTestJar creates an empty Jar with testPSL as the public suffix list.
func newTestJar() *Jar {
	jar, err := New(&Options{PublicSuffixList: testPSL{}})
	if err != nil {
		panic(err)
	}
	return jar
}

var canonicalHostTests = map[string]string{
	"www.example.com":         "www.example.com",
	"WWW.EXAMPLE.COM":         "www.example.com",
	"wWw.eXAmple.CoM":         "www.example.com",
	"www.example.com:80":      "www.example.com",
	"192.168.0.10":            "192.168.0.10",
	"192.168.0.5:8080":        "192.168.0.5",
	"2001:4860:0:2001::68":    "2001:4860:0:2001::68",
	"[2001:4860:0:::68]:8080": "2001:4860:0:::68",
	// "www.bücher.de":        "www.xn--bcher-kva.de",  // TODO de-comment once proper idna is available
	"www.example.com.": "www.example.com",
}

func TestCanonicalHost(t *testing.T) {
	for h, want := range canonicalHostTests {
		got, _ := canonicalHost(h)
		if got != want {
			t.Errorf("%q: got %q, want %q", h, got, want)
		}
		// TODO handle errors
	}
}

var hasPortTests = map[string]bool{
	"www.example.com":      false,
	"www.example.com:80":   true,
	"127.0.0.1":            false,
	"127.0.0.1:8080":       true,
	"2001:4860:0:2001::68": false,
	"[2001::0:::68]:80":    true,
}

func TestHasPort(t *testing.T) {
	for host, want := range hasPortTests {
		if got := hasPort(host); got != want {
			t.Errorf("%q: got %t, want %t", host, got, want)
		}
	}
}

var jarKeyTests = map[string]string{
	"foo.www.example.com": "example.com",
	"www.example.com":     "example.com",
	"example.com":         "example.com",
	"com":                 "com",
	"foo.www.bbc.co.uk":   "bbc.co.uk",
	"www.bbc.co.uk":       "bbc.co.uk",
	"bbc.co.uk":           "bbc.co.uk",
	"co.uk":               "co.uk",
	"uk":                  "uk",
	"192.168.0.5":         "192.168.0.5",
}

func TestJarKey(t *testing.T) {
	for host, want := range jarKeyTests {
		if got := jarKey(host, testPSL{}); got != want {
			t.Errorf("%q: got %q, want %q", host, got, want)
		}
	}

	for _, host := range []string{"www.example.com", "example.com", "com"} {
		if got := jarKey(host, nil); got != "com" {
			t.Errorf(`%q: got %q, want "com"`, host, got)
		}
	}
}

var isIPTests = map[string]bool{
	"127.0.0.1":            true,
	"1.2.3.4":              true,
	"2001:4860:0:2001::68": true,
	"example.com":          false,
	"1.1.1.300":            false,
	"www.foo.bar.net":      false,
	"123.foo.bar.net":      false,
}

func TestIsIP(t *testing.T) {
	for host, want := range isIPTests {
		if got := isIP(host); got != want {
			t.Errorf("%q: got %t, want %t", host, got, want)
		}
	}
}

var defaultPathTests = map[string]string{
	"/":           "/",
	"/abc":        "/",
	"/abc/":       "/abc",
	"/abc/xyz":    "/abc",
	"/abc/xyz/":   "/abc/xyz",
	"/a/b/c.html": "/a/b",
	"":            "/",
	"strange":     "/",
	"//":          "/",
	"/a//b":       "/a/",
	"/a/./b":      "/a/.",
	"/a/../b":     "/a/..",
}

func TestDefaultPath(t *testing.T) {
	for path, want := range defaultPathTests {
		if got := defaultPath(path); got != want {
			t.Errorf("%q: got %q, want %q", path, got, want)
		}
	}
}

var domainAndTypeTests = [...]struct {
	host         string // host Set-Cookie header was received from
	domain       string // domain attribute in Set-Cookie header
	wantDomain   string // expected domain of cookie
	wantHostOnly bool   // expected host-cookie flag
	wantErr      error  // expected error
}{
	{"www.example.com", "", "www.example.com", true, nil},
	{"127.0.0.1", "", "127.0.0.1", true, nil},
	{"2001:4860:0:2001::68", "", "2001:4860:0:2001::68", true, nil},
	{"www.example.com", "example.com", "example.com", false, nil},
	{"www.example.com", ".example.com", "example.com", false, nil},
	{"www.example.com", "www.example.com", "www.example.com", false, nil},
	{"www.example.com", ".www.example.com", "www.example.com", false, nil},
	{"foo.sso.example.com", "sso.example.com", "sso.example.com", false, nil},
	{"bar.co.uk", "bar.co.uk", "bar.co.uk", false, nil},
	{"foo.bar.co.uk", ".bar.co.uk", "bar.co.uk", false, nil},
	{"127.0.0.1", "127.0.0.1", "", false, errNoHostname},
	{"2001:4860:0:2001::68", "2001:4860:0:2001::68", "2001:4860:0:2001::68", false, errNoHostname},
	{"www.example.com", ".", "", false, errMalformedDomain},
	{"www.example.com", "..", "", false, errMalformedDomain},
	{"www.example.com", "other.com", "", false, errIllegalDomain},
	{"www.example.com", "com", "", false, errIllegalDomain},
	{"www.example.com", ".com", "", false, errIllegalDomain},
	{"foo.bar.co.uk", ".co.uk", "", false, errIllegalDomain},
	{"127.www.0.0.1", "127.0.0.1", "", false, errIllegalDomain},
	{"com", "", "com", true, nil},
	{"com", "com", "com", true, nil},
	{"com", ".com", "com", true, nil},
	{"co.uk", "", "co.uk", true, nil},
	{"co.uk", "co.uk", "co.uk", true, nil},
	{"co.uk", ".co.uk", "co.uk", true, nil},
}

func TestDomainAndType(t *testing.T) {
	jar := newTestJar()
	for _, tc := range domainAndTypeTests {
		domain, hostOnly, err := jar.domainAndType(tc.host, tc.domain)
		if err != tc.wantErr {
			t.Errorf("%q/%q: got %q error, want %q",
				tc.host, tc.domain, err, tc.wantErr)
			continue
		}
		if err != nil {
			continue
		}
		if domain != tc.wantDomain || hostOnly != tc.wantHostOnly {
			t.Errorf("%q/%q: got %q/%t want %q/%t",
				tc.host, tc.domain, domain, hostOnly,
				tc.wantDomain, tc.wantHostOnly)
		}
	}
}

// content yields the (non-expired) cookies of jar in the form
// "name1=value1 name2=value2 ...".
func (jar *Jar) content() string {
	var cookies []string
	now := time.Now().UTC()
	for _, submap := range jar.entries {
		for _, cookie := range submap {
			if !cookie.Expires.After(now) {
				continue
			}
			cookies = append(cookies, cookie.Name+"="+cookie.Value)
		}
	}
	sort.Strings(cookies)
	return strings.Join(cookies, " ")
}

// expiresIn creates an expires attribute delta seconds from now.
func expiresIn(delta int) string {
	t := time.Now().Round(time.Second).Add(time.Duration(delta) * time.Second)
	return "expires=" + t.Format(time.RFC1123)
}

// mustParseURL parses s to an URL and panics on error.
func mustParseURL(s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil || u.Scheme == "" || u.Host == "" {
		panic(fmt.Sprintf("Unable to parse URL %s.", s))
	}
	return u
}

// jarTest encapsulates the following actions on a jar:
//   1. Perform SetCookies with fromURL and the cookies from setCookies.
//   2. Check that the entries in the jar matches content.
//   3. For each query in tests: Check that Cookies with toURL yields the
//      cookies in want.
type jarTest struct {
	description string   // The description of what this test is supposed to test
	fromURL     string   // The full URL of the request from which Set-Cookie headers where received
	setCookies  []string // All the cookies received from fromURL
	content     string   // The whole (non-expired) content of the jar
	queries     []query  // Queries to test the Jar.Cookies method
}

// query contains one test of the cookies returned from Jar.Cookies.
type query struct {
	toURL string // the URL in the Cookies call
	want  string // the expected list of cookies (order matters)
}

// run runs the jarTest.
func (test jarTest) run(t *testing.T, jar *Jar) {
	u := mustParseURL(test.fromURL)

	// Populate jar with cookies.
	setCookies := make([]*http.Cookie, len(test.setCookies))
	for i, cs := range test.setCookies {
		cookies := (&http.Response{Header: http.Header{"Set-Cookie": {cs}}}).Cookies()
		if len(cookies) != 1 {
			panic(fmt.Sprintf("Wrong cookie line %q: %#v", cs, cookies))
		}
		setCookies[i] = cookies[0]
	}
	jar.SetCookies(u, setCookies)

	// Make sure jar content matches our expectations.
	if got := jar.content(); got != test.content {
		t.Errorf("Test %q Content\ngot  %q\nwant %q",
			test.description, got, test.content)
	}

	// Test different calls to Cookies.
	for _, query := range test.queries {
		var s []string
		for _, c := range jar.Cookies(mustParseURL(query.toURL)) {
			s = append(s, c.Name+"="+c.Value)
		}
		got := strings.Join(s, " ")
		if got != query.want {
			// TODO: t.Errorf() once Cookies is implemented
		}
	}
}

// basicsTests contains fundamental tests. Each jarTest has to be performed on
// a fresh, empty Jar.
var basicsTests = [...]jarTest{
	{
		"Retrieval of a plain host cookie.",
		"http://www.host.test/",
		[]string{"A=a"},
		"A=a",
		[]query{
			{"http://www.host.test", "A=a"},
			{"http://www.host.test/", "A=a"},
			{"http://www.host.test/some/path", "A=a"},
			{"https://www.host.test", "A=a"},
			{"https://www.host.test/", "A=a"},
			{"https://www.host.test/some/path", "A=a"},
			{"ftp://www.host.test", ""},
			{"ftp://www.host.test/", ""},
			{"ftp://www.host.test/some/path", ""},
			{"http://www.other.org", ""},
			{"http://sibling.host.test", ""},
			{"http://deep.www.host.test", ""},
		},
	},
	{
		"Secure cookies are not returned to http.",
		"http://www.host.test/",
		[]string{"A=a; secure"},
		"A=a",
		[]query{
			{"http://www.host.test", ""},
			{"http://www.host.test/", ""},
			{"http://www.host.test/some/path", ""},
			{"https://www.host.test", "A=a"},
			{"https://www.host.test/", "A=a"},
			{"https://www.host.test/some/path", "A=a"},
		},
	},
	{
		"Explicit path.",
		"http://www.host.test/",
		[]string{"A=a; path=/some/path"},
		"A=a",
		[]query{
			{"http://www.host.test", ""},
			{"http://www.host.test/", ""},
			{"http://www.host.test/some", ""},
			{"http://www.host.test/some/", ""},
			{"http://www.host.test/some/path", "A=a"},
			{"http://www.host.test/some/paths", ""},
			{"http://www.host.test/some/path/foo", "A=a"},
			{"http://www.host.test/some/path/foo/", "A=a"},
		},
	},
	{
		"Implicit path #1: path is a directory.",
		"http://www.host.test/some/path/",
		[]string{"A=a"},
		"A=a",
		[]query{
			{"http://www.host.test", ""},
			{"http://www.host.test/", ""},
			{"http://www.host.test/some", ""},
			{"http://www.host.test/some/", ""},
			{"http://www.host.test/some/path", "A=a"},
			{"http://www.host.test/some/paths", ""},
			{"http://www.host.test/some/path/foo", "A=a"},
			{"http://www.host.test/some/path/foo/", "A=a"},
		},
	},
	{
		"Implicit path #2: path is not a directory.",
		"http://www.host.test/some/path/index.html",
		[]string{"A=a"},
		"A=a",
		[]query{
			{"http://www.host.test", ""},
			{"http://www.host.test/", ""},
			{"http://www.host.test/some", ""},
			{"http://www.host.test/some/", ""},
			{"http://www.host.test/some/path", "A=a"},
			{"http://www.host.test/some/paths", ""},
			{"http://www.host.test/some/path/foo", "A=a"},
			{"http://www.host.test/some/path/foo/", "A=a"},
		},
	},
	{
		"Implicit path #3: no path in URL at all.",
		"http://www.host.test",
		[]string{"A=a"},
		"A=a",
		[]query{
			{"http://www.host.test", "A=a"},
			{"http://www.host.test/", "A=a"},
			{"http://www.host.test/some/path", "A=a"},
		},
	},
	{
		"Cookies are sorted by path length.",
		"http://www.host.test/",
		[]string{
			"A=a; path=/foo/bar",
			"B=b; path=/foo/bar/baz/qux",
			"C=c; path=/foo/bar/baz",
			"D=d; path=/foo"},
		"A=a B=b C=c D=d",
		[]query{
			{"http://www.host.test/foo/bar/baz/qux", "B=b C=c A=a D=d"},
			{"http://www.host.test/foo/bar/baz/", "C=c A=a D=d"},
			{"http://www.host.test/foo/bar", "A=a D=d"},
		},
	},
	{
		"Creation time determines sorting on same length paths.",
		"http://www.host.test/",
		[]string{
			"A=a; path=/foo/bar",
			"X=x; path=/foo/bar",
			"Y=y; path=/foo/bar/baz/qux",
			"B=b; path=/foo/bar/baz/qux",
			"C=c; path=/foo/bar/baz",
			"W=w; path=/foo/bar/baz",
			"Z=z; path=/foo",
			"D=d; path=/foo"},
		"A=a B=b C=c D=d W=w X=x Y=y Z=z",
		[]query{
			{"http://www.host.test/foo/bar/baz/qux", "Y=y B=b C=c W=w A=a X=x Z=z D=d"},
			{"http://www.host.test/foo/bar/baz/", "C=c W=w A=a X=x Z=z D=d"},
			{"http://www.host.test/foo/bar", "A=a X=x Z=z D=d"},
		},
	},
	{
		"Sorting of same-name cookies.",
		"http://www.host.test/",
		[]string{
			"A=1; path=/",
			"A=2; path=/path",
			"A=3; path=/quux",
			"A=4; path=/path/foo",
			"A=5; domain=.host.test; path=/path",
			"A=6; domain=.host.test; path=/quux",
			"A=7; domain=.host.test; path=/path/foo",
		},
		"A=1 A=2 A=3 A=4 A=5 A=6 A=7",
		[]query{
			{"http://www.host.test/path", "A=2 A=5 A=1"},
			{"http://www.host.test/path/foo", "A=4 A=7 A=2 A=5 A=1"},
		},
	},
}

func TestBasics(t *testing.T) {
	for _, test := range basicsTests {
		jar := newTestJar()
		test.run(t, jar)
	}
}
