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

// tNow is the synthetic current time used as now during testing.
var tNow = time.Date(2013, 1, 1, 12, 0, 0, 0, time.UTC)

// testPSL implements PublicSuffixList with just two rules: "co.uk"
// and the default rule "*".
// The implementation has two intentional bugs:
//
//	PublicSuffix("www.buggy.psl") == "xy"
//	PublicSuffix("www2.buggy.psl") == "com"
type testPSL struct{}

func (testPSL) String() string {
	return "testPSL"
}
func (testPSL) PublicSuffix(d string) string {
	if d == "co.uk" || strings.HasSuffix(d, ".co.uk") {
		return "co.uk"
	}
	if d == "www.buggy.psl" {
		return "xy"
	}
	if d == "www2.buggy.psl" {
		return "com"
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

var hasDotSuffixTests = [...]struct {
	s, suffix string
}{
	{"", ""},
	{"", "."},
	{"", "x"},
	{".", ""},
	{".", "."},
	{".", ".."},
	{".", "x"},
	{".", "x."},
	{".", ".x"},
	{".", ".x."},
	{"x", ""},
	{"x", "."},
	{"x", ".."},
	{"x", "x"},
	{"x", "x."},
	{"x", ".x"},
	{"x", ".x."},
	{".x", ""},
	{".x", "."},
	{".x", ".."},
	{".x", "x"},
	{".x", "x."},
	{".x", ".x"},
	{".x", ".x."},
	{"x.", ""},
	{"x.", "."},
	{"x.", ".."},
	{"x.", "x"},
	{"x.", "x."},
	{"x.", ".x"},
	{"x.", ".x."},
	{"com", ""},
	{"com", "m"},
	{"com", "om"},
	{"com", "com"},
	{"com", ".com"},
	{"com", "x.com"},
	{"com", "xcom"},
	{"com", "xorg"},
	{"com", "org"},
	{"com", "rg"},
	{"foo.com", ""},
	{"foo.com", "m"},
	{"foo.com", "om"},
	{"foo.com", "com"},
	{"foo.com", ".com"},
	{"foo.com", "o.com"},
	{"foo.com", "oo.com"},
	{"foo.com", "foo.com"},
	{"foo.com", ".foo.com"},
	{"foo.com", "x.foo.com"},
	{"foo.com", "xfoo.com"},
	{"foo.com", "xfoo.org"},
	{"foo.com", "foo.org"},
	{"foo.com", "oo.org"},
	{"foo.com", "o.org"},
	{"foo.com", ".org"},
	{"foo.com", "org"},
	{"foo.com", "rg"},
}

func TestHasDotSuffix(t *testing.T) {
	for _, tc := range hasDotSuffixTests {
		got := hasDotSuffix(tc.s, tc.suffix)
		want := strings.HasSuffix(tc.s, "."+tc.suffix)
		if got != want {
			t.Errorf("s=%q, suffix=%q: got %v, want %v", tc.s, tc.suffix, got, want)
		}
	}
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
	"www.bücher.de":           "www.xn--bcher-kva.de",
	"www.example.com.":        "www.example.com",
	// TODO: Fix canonicalHost so that all of the following malformed
	// domain names trigger an error. (This list is not exhaustive, e.g.
	// malformed internationalized domain names are missing.)
	".":                       "",
	"..":                      ".",
	"...":                     "..",
	".net":                    ".net",
	".net.":                   ".net",
	"a..":                     "a.",
	"b.a..":                   "b.a.",
	"weird.stuff...":          "weird.stuff..",
	"[bad.unmatched.bracket:": "error",
}

func TestCanonicalHost(t *testing.T) {
	for h, want := range canonicalHostTests {
		got, err := canonicalHost(h)
		if want == "error" {
			if err == nil {
				t.Errorf("%q: got %q and nil error, want non-nil", h, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("%q: %v", h, err)
			continue
		}
		if got != want {
			t.Errorf("%q: got %q, want %q", h, got, want)
			continue
		}
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
	"www.buggy.psl":       "www.buggy.psl",
	"www2.buggy.psl":      "buggy.psl",
	// The following are actual outputs of canonicalHost for
	// malformed inputs to canonicalHost (see above).
	"":              "",
	".":             ".",
	"..":            ".",
	".net":          ".net",
	"a.":            "a.",
	"b.a.":          "a.",
	"weird.stuff..": ".",
}

func TestJarKey(t *testing.T) {
	for host, want := range jarKeyTests {
		if got := jarKey(host, testPSL{}); got != want {
			t.Errorf("%q: got %q, want %q", host, got, want)
		}
	}
}

var jarKeyNilPSLTests = map[string]string{
	"foo.www.example.com": "example.com",
	"www.example.com":     "example.com",
	"example.com":         "example.com",
	"com":                 "com",
	"foo.www.bbc.co.uk":   "co.uk",
	"www.bbc.co.uk":       "co.uk",
	"bbc.co.uk":           "co.uk",
	"co.uk":               "co.uk",
	"uk":                  "uk",
	"192.168.0.5":         "192.168.0.5",
	// The following are actual outputs of canonicalHost for
	// malformed inputs to canonicalHost.
	"":              "",
	".":             ".",
	"..":            "..",
	".net":          ".net",
	"a.":            "a.",
	"b.a.":          "a.",
	"weird.stuff..": "stuff..",
}

func TestJarKeyNilPSL(t *testing.T) {
	for host, want := range jarKeyNilPSLTests {
		if got := jarKey(host, nil); got != want {
			t.Errorf("%q: got %q, want %q", host, got, want)
		}
	}
}

var isIPTests = map[string]bool{
	"127.0.0.1":            true,
	"1.2.3.4":              true,
	"2001:4860:0:2001::68": true,
	"::1%zone":             true,
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
	{"127.0.0.1", "127.0.0.1", "127.0.0.1", true, nil},
	{"2001:4860:0:2001::68", "2001:4860:0:2001::68", "2001:4860:0:2001::68", true, nil},
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
			t.Errorf("%q/%q: got %q error, want %v",
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

// expiresIn creates an expires attribute delta seconds from tNow.
func expiresIn(delta int) string {
	t := tNow.Add(time.Duration(delta) * time.Second)
	return "expires=" + t.Format(time.RFC1123)
}

// mustParseURL parses s to a URL and panics on error.
func mustParseURL(s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil || u.Scheme == "" || u.Host == "" {
		panic(fmt.Sprintf("Unable to parse URL %s.", s))
	}
	return u
}

// jarTest encapsulates the following actions on a jar:
//  1. Perform SetCookies with fromURL and the cookies from setCookies.
//     (Done at time tNow + 0 ms.)
//  2. Check that the entries in the jar matches content.
//     (Done at time tNow + 1001 ms.)
//  3. For each query in tests: Check that Cookies with toURL yields the
//     cookies in want.
//     (Query n done at tNow + (n+2)*1001 ms.)
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
	now := tNow

	// Populate jar with cookies.
	setCookies := make([]*http.Cookie, len(test.setCookies))
	for i, cs := range test.setCookies {
		cookies := (&http.Response{Header: http.Header{"Set-Cookie": {cs}}}).Cookies()
		if len(cookies) != 1 {
			panic(fmt.Sprintf("Wrong cookie line %q: %#v", cs, cookies))
		}
		setCookies[i] = cookies[0]
	}
	jar.setCookies(mustParseURL(test.fromURL), setCookies, now)
	now = now.Add(1001 * time.Millisecond)

	// Serialize non-expired entries in the form "name1=val1 name2=val2".
	var cs []string
	for _, submap := range jar.entries {
		for _, cookie := range submap {
			if !cookie.Expires.After(now) {
				continue
			}

			v := cookie.Value
			if strings.ContainsAny(v, " ,") || cookie.Quoted {
				v = `"` + v + `"`
			}
			cs = append(cs, cookie.Name+"="+v)
		}
	}
	sort.Strings(cs)
	got := strings.Join(cs, " ")

	// Make sure jar content matches our expectations.
	if got != test.content {
		t.Errorf("Test %q Content\ngot  %q\nwant %q",
			test.description, got, test.content)
	}

	// Test different calls to Cookies.
	for i, query := range test.queries {
		now = now.Add(1001 * time.Millisecond)
		var s []string
		for _, c := range jar.cookies(mustParseURL(query.toURL), now) {
			s = append(s, c.String())
		}
		if got := strings.Join(s, " "); got != query.want {
			t.Errorf("Test %q #%d\ngot  %q\nwant %q", test.description, i, got, query.want)
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
	{
		"Disallow domain cookie on public suffix.",
		"http://www.bbc.co.uk",
		[]string{
			"a=1",
			"b=2; domain=co.uk",
		},
		"a=1",
		[]query{{"http://www.bbc.co.uk", "a=1"}},
	},
	{
		"Host cookie on IP.",
		"http://192.168.0.10",
		[]string{"a=1"},
		"a=1",
		[]query{{"http://192.168.0.10", "a=1"}},
	},
	{
		"Domain cookies on IP.",
		"http://192.168.0.10",
		[]string{
			"a=1; domain=192.168.0.10",  // allowed
			"b=2; domain=172.31.9.9",    // rejected, can't set cookie for other IP
			"c=3; domain=.192.168.0.10", // rejected like in most browsers
		},
		"a=1",
		[]query{
			{"http://192.168.0.10", "a=1"},
			{"http://172.31.9.9", ""},
			{"http://www.fancy.192.168.0.10", ""},
		},
	},
	{
		"Port is ignored #1.",
		"http://www.host.test/",
		[]string{"a=1"},
		"a=1",
		[]query{
			{"http://www.host.test", "a=1"},
			{"http://www.host.test:8080/", "a=1"},
		},
	},
	{
		"Port is ignored #2.",
		"http://www.host.test:8080/",
		[]string{"a=1"},
		"a=1",
		[]query{
			{"http://www.host.test", "a=1"},
			{"http://www.host.test:8080/", "a=1"},
			{"http://www.host.test:1234/", "a=1"},
		},
	},
	{
		"IPv6 zone is not treated as a host.",
		"https://example.com/",
		[]string{"a=1"},
		"a=1",
		[]query{
			{"https://[::1%25.example.com]:80/", ""},
		},
	},
	{
		"Retrieval of cookies with quoted values", // issue #46443
		"http://www.host.test/",
		[]string{
			`cookie-1="quoted"`,
			`cookie-2="quoted with spaces"`,
			`cookie-3="quoted,with,commas"`,
			`cookie-4= ,`,
		},
		`cookie-1="quoted" cookie-2="quoted with spaces" cookie-3="quoted,with,commas" cookie-4=" ,"`,
		[]query{
			{
				"http://www.host.test",
				`cookie-1="quoted" cookie-2="quoted with spaces" cookie-3="quoted,with,commas" cookie-4=" ,"`,
			},
		},
	},
}

func TestBasics(t *testing.T) {
	for _, test := range basicsTests {
		jar := newTestJar()
		test.run(t, jar)
	}
}

// updateAndDeleteTests contains jarTests which must be performed on the same
// Jar.
var updateAndDeleteTests = [...]jarTest{
	{
		"Set initial cookies.",
		"http://www.host.test",
		[]string{
			"a=1",
			"b=2; secure",
			"c=3; httponly",
			"d=4; secure; httponly"},
		"a=1 b=2 c=3 d=4",
		[]query{
			{"http://www.host.test", "a=1 c=3"},
			{"https://www.host.test", "a=1 b=2 c=3 d=4"},
		},
	},
	{
		"Update value via http.",
		"http://www.host.test",
		[]string{
			"a=w",
			"b=x; secure",
			"c=y; httponly",
			"d=z; secure; httponly"},
		"a=w b=x c=y d=z",
		[]query{
			{"http://www.host.test", "a=w c=y"},
			{"https://www.host.test", "a=w b=x c=y d=z"},
		},
	},
	{
		"Clear Secure flag from an http.",
		"http://www.host.test/",
		[]string{
			"b=xx",
			"d=zz; httponly"},
		"a=w b=xx c=y d=zz",
		[]query{{"http://www.host.test", "a=w b=xx c=y d=zz"}},
	},
	{
		"Delete all.",
		"http://www.host.test/",
		[]string{
			"a=1; max-Age=-1",                    // delete via MaxAge
			"b=2; " + expiresIn(-10),             // delete via Expires
			"c=2; max-age=-1; " + expiresIn(-10), // delete via both
			"d=4; max-age=-1; " + expiresIn(10)}, // MaxAge takes precedence
		"",
		[]query{{"http://www.host.test", ""}},
	},
	{
		"Refill #1.",
		"http://www.host.test",
		[]string{
			"A=1",
			"A=2; path=/foo",
			"A=3; domain=.host.test",
			"A=4; path=/foo; domain=.host.test"},
		"A=1 A=2 A=3 A=4",
		[]query{{"http://www.host.test/foo", "A=2 A=4 A=1 A=3"}},
	},
	{
		"Refill #2.",
		"http://www.google.com",
		[]string{
			"A=6",
			"A=7; path=/foo",
			"A=8; domain=.google.com",
			"A=9; path=/foo; domain=.google.com"},
		"A=1 A=2 A=3 A=4 A=6 A=7 A=8 A=9",
		[]query{
			{"http://www.host.test/foo", "A=2 A=4 A=1 A=3"},
			{"http://www.google.com/foo", "A=7 A=9 A=6 A=8"},
		},
	},
	{
		"Delete A7.",
		"http://www.google.com",
		[]string{"A=; path=/foo; max-age=-1"},
		"A=1 A=2 A=3 A=4 A=6 A=8 A=9",
		[]query{
			{"http://www.host.test/foo", "A=2 A=4 A=1 A=3"},
			{"http://www.google.com/foo", "A=9 A=6 A=8"},
		},
	},
	{
		"Delete A4.",
		"http://www.host.test",
		[]string{"A=; path=/foo; domain=host.test; max-age=-1"},
		"A=1 A=2 A=3 A=6 A=8 A=9",
		[]query{
			{"http://www.host.test/foo", "A=2 A=1 A=3"},
			{"http://www.google.com/foo", "A=9 A=6 A=8"},
		},
	},
	{
		"Delete A6.",
		"http://www.google.com",
		[]string{"A=; max-age=-1"},
		"A=1 A=2 A=3 A=8 A=9",
		[]query{
			{"http://www.host.test/foo", "A=2 A=1 A=3"},
			{"http://www.google.com/foo", "A=9 A=8"},
		},
	},
	{
		"Delete A3.",
		"http://www.host.test",
		[]string{"A=; domain=host.test; max-age=-1"},
		"A=1 A=2 A=8 A=9",
		[]query{
			{"http://www.host.test/foo", "A=2 A=1"},
			{"http://www.google.com/foo", "A=9 A=8"},
		},
	},
	{
		"No cross-domain delete.",
		"http://www.host.test",
		[]string{
			"A=; domain=google.com; max-age=-1",
			"A=; path=/foo; domain=google.com; max-age=-1"},
		"A=1 A=2 A=8 A=9",
		[]query{
			{"http://www.host.test/foo", "A=2 A=1"},
			{"http://www.google.com/foo", "A=9 A=8"},
		},
	},
	{
		"Delete A8 and A9.",
		"http://www.google.com",
		[]string{
			"A=; domain=google.com; max-age=-1",
			"A=; path=/foo; domain=google.com; max-age=-1"},
		"A=1 A=2",
		[]query{
			{"http://www.host.test/foo", "A=2 A=1"},
			{"http://www.google.com/foo", ""},
		},
	},
}

func TestUpdateAndDelete(t *testing.T) {
	jar := newTestJar()
	for _, test := range updateAndDeleteTests {
		test.run(t, jar)
	}
}

func TestExpiration(t *testing.T) {
	jar := newTestJar()
	jarTest{
		"Expiration.",
		"http://www.host.test",
		[]string{
			"a=1",
			"b=2; max-age=3",
			"c=3; " + expiresIn(3),
			"d=4; max-age=5",
			"e=5; " + expiresIn(5),
			"f=6; max-age=100",
		},
		"a=1 b=2 c=3 d=4 e=5 f=6", // executed at t0 + 1001 ms
		[]query{
			{"http://www.host.test", "a=1 b=2 c=3 d=4 e=5 f=6"}, // t0 + 2002 ms
			{"http://www.host.test", "a=1 d=4 e=5 f=6"},         // t0 + 3003 ms
			{"http://www.host.test", "a=1 d=4 e=5 f=6"},         // t0 + 4004 ms
			{"http://www.host.test", "a=1 f=6"},                 // t0 + 5005 ms
			{"http://www.host.test", "a=1 f=6"},                 // t0 + 6006 ms
		},
	}.run(t, jar)
}

//
// Tests derived from Chromium's cookie_store_unittest.h.
//

// See http://src.chromium.org/viewvc/chrome/trunk/src/net/cookies/cookie_store_unittest.h?revision=159685&content-type=text/plain
// Some of the original tests are in a bad condition (e.g.
// DomainWithTrailingDotTest) or are not RFC 6265 conforming (e.g.
// TestNonDottedAndTLD #1 and #6) and have not been ported.

// chromiumBasicsTests contains fundamental tests. Each jarTest has to be
// performed on a fresh, empty Jar.
var chromiumBasicsTests = [...]jarTest{
	{
		"DomainWithTrailingDotTest.",
		"http://www.google.com/",
		[]string{
			"a=1; domain=.www.google.com.",
			"b=2; domain=.www.google.com.."},
		"",
		[]query{
			{"http://www.google.com", ""},
		},
	},
	{
		"ValidSubdomainTest #1.",
		"http://a.b.c.d.com",
		[]string{
			"a=1; domain=.a.b.c.d.com",
			"b=2; domain=.b.c.d.com",
			"c=3; domain=.c.d.com",
			"d=4; domain=.d.com"},
		"a=1 b=2 c=3 d=4",
		[]query{
			{"http://a.b.c.d.com", "a=1 b=2 c=3 d=4"},
			{"http://b.c.d.com", "b=2 c=3 d=4"},
			{"http://c.d.com", "c=3 d=4"},
			{"http://d.com", "d=4"},
		},
	},
	{
		"ValidSubdomainTest #2.",
		"http://a.b.c.d.com",
		[]string{
			"a=1; domain=.a.b.c.d.com",
			"b=2; domain=.b.c.d.com",
			"c=3; domain=.c.d.com",
			"d=4; domain=.d.com",
			"X=bcd; domain=.b.c.d.com",
			"X=cd; domain=.c.d.com"},
		"X=bcd X=cd a=1 b=2 c=3 d=4",
		[]query{
			{"http://b.c.d.com", "b=2 c=3 d=4 X=bcd X=cd"},
			{"http://c.d.com", "c=3 d=4 X=cd"},
		},
	},
	{
		"InvalidDomainTest #1.",
		"http://foo.bar.com",
		[]string{
			"a=1; domain=.yo.foo.bar.com",
			"b=2; domain=.foo.com",
			"c=3; domain=.bar.foo.com",
			"d=4; domain=.foo.bar.com.net",
			"e=5; domain=ar.com",
			"f=6; domain=.",
			"g=7; domain=/",
			"h=8; domain=http://foo.bar.com",
			"i=9; domain=..foo.bar.com",
			"j=10; domain=..bar.com",
			"k=11; domain=.foo.bar.com?blah",
			"l=12; domain=.foo.bar.com/blah",
			"m=12; domain=.foo.bar.com:80",
			"n=14; domain=.foo.bar.com:",
			"o=15; domain=.foo.bar.com#sup",
		},
		"", // Jar is empty.
		[]query{{"http://foo.bar.com", ""}},
	},
	{
		"InvalidDomainTest #2.",
		"http://foo.com.com",
		[]string{"a=1; domain=.foo.com.com.com"},
		"",
		[]query{{"http://foo.bar.com", ""}},
	},
	{
		"DomainWithoutLeadingDotTest #1.",
		"http://manage.hosted.filefront.com",
		[]string{"a=1; domain=filefront.com"},
		"a=1",
		[]query{{"http://www.filefront.com", "a=1"}},
	},
	{
		"DomainWithoutLeadingDotTest #2.",
		"http://www.google.com",
		[]string{"a=1; domain=www.google.com"},
		"a=1",
		[]query{
			{"http://www.google.com", "a=1"},
			{"http://sub.www.google.com", "a=1"},
			{"http://something-else.com", ""},
		},
	},
	{
		"CaseInsensitiveDomainTest.",
		"http://www.google.com",
		[]string{
			"a=1; domain=.GOOGLE.COM",
			"b=2; domain=.www.gOOgLE.coM"},
		"a=1 b=2",
		[]query{{"http://www.google.com", "a=1 b=2"}},
	},
	{
		"TestIpAddress #1.",
		"http://1.2.3.4/foo",
		[]string{"a=1; path=/"},
		"a=1",
		[]query{{"http://1.2.3.4/foo", "a=1"}},
	},
	{
		"TestIpAddress #2.",
		"http://1.2.3.4/foo",
		[]string{
			"a=1; domain=.1.2.3.4",
			"b=2; domain=.3.4"},
		"",
		[]query{{"http://1.2.3.4/foo", ""}},
	},
	{
		"TestIpAddress #3.",
		"http://1.2.3.4/foo",
		[]string{"a=1; domain=1.2.3.3"},
		"",
		[]query{{"http://1.2.3.4/foo", ""}},
	},
	{
		"TestIpAddress #4.",
		"http://1.2.3.4/foo",
		[]string{"a=1; domain=1.2.3.4"},
		"a=1",
		[]query{{"http://1.2.3.4/foo", "a=1"}},
	},
	{
		"TestNonDottedAndTLD #2.",
		"http://com./index.html",
		[]string{"a=1"},
		"a=1",
		[]query{
			{"http://com./index.html", "a=1"},
			{"http://no-cookies.com./index.html", ""},
		},
	},
	{
		"TestNonDottedAndTLD #3.",
		"http://a.b",
		[]string{
			"a=1; domain=.b",
			"b=2; domain=b"},
		"",
		[]query{{"http://bar.foo", ""}},
	},
	{
		"TestNonDottedAndTLD #4.",
		"http://google.com",
		[]string{
			"a=1; domain=.com",
			"b=2; domain=com"},
		"",
		[]query{{"http://google.com", ""}},
	},
	{
		"TestNonDottedAndTLD #5.",
		"http://google.co.uk",
		[]string{
			"a=1; domain=.co.uk",
			"b=2; domain=.uk"},
		"",
		[]query{
			{"http://google.co.uk", ""},
			{"http://else.co.com", ""},
			{"http://else.uk", ""},
		},
	},
	{
		"TestHostEndsWithDot.",
		"http://www.google.com",
		[]string{
			"a=1",
			"b=2; domain=.www.google.com."},
		"a=1",
		[]query{{"http://www.google.com", "a=1"}},
	},
	{
		"PathTest",
		"http://www.google.izzle",
		[]string{"a=1; path=/wee"},
		"a=1",
		[]query{
			{"http://www.google.izzle/wee", "a=1"},
			{"http://www.google.izzle/wee/", "a=1"},
			{"http://www.google.izzle/wee/war", "a=1"},
			{"http://www.google.izzle/wee/war/more/more", "a=1"},
			{"http://www.google.izzle/weehee", ""},
			{"http://www.google.izzle/", ""},
		},
	},
}

func TestChromiumBasics(t *testing.T) {
	for _, test := range chromiumBasicsTests {
		jar := newTestJar()
		test.run(t, jar)
	}
}

// chromiumDomainTests contains jarTests which must be executed all on the
// same Jar.
var chromiumDomainTests = [...]jarTest{
	{
		"Fill #1.",
		"http://www.google.izzle",
		[]string{"A=B"},
		"A=B",
		[]query{{"http://www.google.izzle", "A=B"}},
	},
	{
		"Fill #2.",
		"http://www.google.izzle",
		[]string{"C=D; domain=.google.izzle"},
		"A=B C=D",
		[]query{{"http://www.google.izzle", "A=B C=D"}},
	},
	{
		"Verify A is a host cookie and not accessible from subdomain.",
		"http://unused.nil",
		[]string{},
		"A=B C=D",
		[]query{{"http://foo.www.google.izzle", "C=D"}},
	},
	{
		"Verify domain cookies are found on proper domain.",
		"http://www.google.izzle",
		[]string{"E=F; domain=.www.google.izzle"},
		"A=B C=D E=F",
		[]query{{"http://www.google.izzle", "A=B C=D E=F"}},
	},
	{
		"Leading dots in domain attributes are optional.",
		"http://www.google.izzle",
		[]string{"G=H; domain=www.google.izzle"},
		"A=B C=D E=F G=H",
		[]query{{"http://www.google.izzle", "A=B C=D E=F G=H"}},
	},
	{
		"Verify domain enforcement works #1.",
		"http://www.google.izzle",
		[]string{"K=L; domain=.bar.www.google.izzle"},
		"A=B C=D E=F G=H",
		[]query{{"http://bar.www.google.izzle", "C=D E=F G=H"}},
	},
	{
		"Verify domain enforcement works #2.",
		"http://unused.nil",
		[]string{},
		"A=B C=D E=F G=H",
		[]query{{"http://www.google.izzle", "A=B C=D E=F G=H"}},
	},
}

func TestChromiumDomain(t *testing.T) {
	jar := newTestJar()
	for _, test := range chromiumDomainTests {
		test.run(t, jar)
	}

}

// chromiumDeletionTests must be performed all on the same Jar.
var chromiumDeletionTests = [...]jarTest{
	{
		"Create session cookie a1.",
		"http://www.google.com",
		[]string{"a=1"},
		"a=1",
		[]query{{"http://www.google.com", "a=1"}},
	},
	{
		"Delete sc a1 via MaxAge.",
		"http://www.google.com",
		[]string{"a=1; max-age=-1"},
		"",
		[]query{{"http://www.google.com", ""}},
	},
	{
		"Create session cookie b2.",
		"http://www.google.com",
		[]string{"b=2"},
		"b=2",
		[]query{{"http://www.google.com", "b=2"}},
	},
	{
		"Delete sc b2 via Expires.",
		"http://www.google.com",
		[]string{"b=2; " + expiresIn(-10)},
		"",
		[]query{{"http://www.google.com", ""}},
	},
	{
		"Create persistent cookie c3.",
		"http://www.google.com",
		[]string{"c=3; max-age=3600"},
		"c=3",
		[]query{{"http://www.google.com", "c=3"}},
	},
	{
		"Delete pc c3 via MaxAge.",
		"http://www.google.com",
		[]string{"c=3; max-age=-1"},
		"",
		[]query{{"http://www.google.com", ""}},
	},
	{
		"Create persistent cookie d4.",
		"http://www.google.com",
		[]string{"d=4; max-age=3600"},
		"d=4",
		[]query{{"http://www.google.com", "d=4"}},
	},
	{
		"Delete pc d4 via Expires.",
		"http://www.google.com",
		[]string{"d=4; " + expiresIn(-10)},
		"",
		[]query{{"http://www.google.com", ""}},
	},
}

func TestChromiumDeletion(t *testing.T) {
	jar := newTestJar()
	for _, test := range chromiumDeletionTests {
		test.run(t, jar)
	}
}

// domainHandlingTests tests and documents the rules for domain handling.
// Each test must be performed on an empty new Jar.
var domainHandlingTests = [...]jarTest{
	{
		"Host cookie",
		"http://www.host.test",
		[]string{"a=1"},
		"a=1",
		[]query{
			{"http://www.host.test", "a=1"},
			{"http://host.test", ""},
			{"http://bar.host.test", ""},
			{"http://foo.www.host.test", ""},
			{"http://other.test", ""},
			{"http://test", ""},
		},
	},
	{
		"Domain cookie #1",
		"http://www.host.test",
		[]string{"a=1; domain=host.test"},
		"a=1",
		[]query{
			{"http://www.host.test", "a=1"},
			{"http://host.test", "a=1"},
			{"http://bar.host.test", "a=1"},
			{"http://foo.www.host.test", "a=1"},
			{"http://other.test", ""},
			{"http://test", ""},
		},
	},
	{
		"Domain cookie #2",
		"http://www.host.test",
		[]string{"a=1; domain=.host.test"},
		"a=1",
		[]query{
			{"http://www.host.test", "a=1"},
			{"http://host.test", "a=1"},
			{"http://bar.host.test", "a=1"},
			{"http://foo.www.host.test", "a=1"},
			{"http://other.test", ""},
			{"http://test", ""},
		},
	},
	{
		"Host cookie on IDNA domain #1",
		"http://www.bücher.test",
		[]string{"a=1"},
		"a=1",
		[]query{
			{"http://www.bücher.test", "a=1"},
			{"http://www.xn--bcher-kva.test", "a=1"},
			{"http://bücher.test", ""},
			{"http://xn--bcher-kva.test", ""},
			{"http://bar.bücher.test", ""},
			{"http://bar.xn--bcher-kva.test", ""},
			{"http://foo.www.bücher.test", ""},
			{"http://foo.www.xn--bcher-kva.test", ""},
			{"http://other.test", ""},
			{"http://test", ""},
		},
	},
	{
		"Host cookie on IDNA domain #2",
		"http://www.xn--bcher-kva.test",
		[]string{"a=1"},
		"a=1",
		[]query{
			{"http://www.bücher.test", "a=1"},
			{"http://www.xn--bcher-kva.test", "a=1"},
			{"http://bücher.test", ""},
			{"http://xn--bcher-kva.test", ""},
			{"http://bar.bücher.test", ""},
			{"http://bar.xn--bcher-kva.test", ""},
			{"http://foo.www.bücher.test", ""},
			{"http://foo.www.xn--bcher-kva.test", ""},
			{"http://other.test", ""},
			{"http://test", ""},
		},
	},
	{
		"Domain cookie on IDNA domain #1",
		"http://www.bücher.test",
		[]string{"a=1; domain=xn--bcher-kva.test"},
		"a=1",
		[]query{
			{"http://www.bücher.test", "a=1"},
			{"http://www.xn--bcher-kva.test", "a=1"},
			{"http://bücher.test", "a=1"},
			{"http://xn--bcher-kva.test", "a=1"},
			{"http://bar.bücher.test", "a=1"},
			{"http://bar.xn--bcher-kva.test", "a=1"},
			{"http://foo.www.bücher.test", "a=1"},
			{"http://foo.www.xn--bcher-kva.test", "a=1"},
			{"http://other.test", ""},
			{"http://test", ""},
		},
	},
	{
		"Domain cookie on IDNA domain #2",
		"http://www.xn--bcher-kva.test",
		[]string{"a=1; domain=xn--bcher-kva.test"},
		"a=1",
		[]query{
			{"http://www.bücher.test", "a=1"},
			{"http://www.xn--bcher-kva.test", "a=1"},
			{"http://bücher.test", "a=1"},
			{"http://xn--bcher-kva.test", "a=1"},
			{"http://bar.bücher.test", "a=1"},
			{"http://bar.xn--bcher-kva.test", "a=1"},
			{"http://foo.www.bücher.test", "a=1"},
			{"http://foo.www.xn--bcher-kva.test", "a=1"},
			{"http://other.test", ""},
			{"http://test", ""},
		},
	},
	{
		"Host cookie on TLD.",
		"http://com",
		[]string{"a=1"},
		"a=1",
		[]query{
			{"http://com", "a=1"},
			{"http://any.com", ""},
			{"http://any.test", ""},
		},
	},
	{
		"Domain cookie on TLD becomes a host cookie.",
		"http://com",
		[]string{"a=1; domain=com"},
		"a=1",
		[]query{
			{"http://com", "a=1"},
			{"http://any.com", ""},
			{"http://any.test", ""},
		},
	},
	{
		"Host cookie on public suffix.",
		"http://co.uk",
		[]string{"a=1"},
		"a=1",
		[]query{
			{"http://co.uk", "a=1"},
			{"http://uk", ""},
			{"http://some.co.uk", ""},
			{"http://foo.some.co.uk", ""},
			{"http://any.uk", ""},
		},
	},
	{
		"Domain cookie on public suffix is ignored.",
		"http://some.co.uk",
		[]string{"a=1; domain=co.uk"},
		"",
		[]query{
			{"http://co.uk", ""},
			{"http://uk", ""},
			{"http://some.co.uk", ""},
			{"http://foo.some.co.uk", ""},
			{"http://any.uk", ""},
		},
	},
}

func TestDomainHandling(t *testing.T) {
	for _, test := range domainHandlingTests {
		jar := newTestJar()
		test.run(t, jar)
	}
}

func TestIssue19384(t *testing.T) {
	cookies := []*http.Cookie{{Name: "name", Value: "value"}}
	for _, host := range []string{"", ".", "..", "..."} {
		jar, _ := New(nil)
		u := &url.URL{Scheme: "http", Host: host, Path: "/"}
		if got := jar.Cookies(u); len(got) != 0 {
			t.Errorf("host %q, got %v", host, got)
		}
		jar.SetCookies(u, cookies)
		if got := jar.Cookies(u); len(got) != 1 || got[0].Value != "value" {
			t.Errorf("host %q, got %v", host, got)
		}
	}
}
