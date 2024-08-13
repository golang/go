package auth

import (
	"testing"
)

func TestParseGitAuth(t *testing.T) {
	testCases := []struct {
		gitauth      string // contents of 'git credential fill'
		wantPrefix   string
		wantUsername string
		wantPassword string
	}{
		{ // Standard case.
			gitauth: `
protocol=https
host=example.com
username=bob
password=secr3t
`,
			wantPrefix:   "https://example.com",
			wantUsername: "bob",
			wantPassword: "secr3t",
		},
		{ // Should not use an invalid url.
			gitauth: `
protocol=https
host=example.com
username=bob
password=secr3t
url=invalid
`,
			wantPrefix:   "https://example.com",
			wantUsername: "bob",
			wantPassword: "secr3t",
		},
		{ // Should use the new url.
			gitauth: `
protocol=https
host=example.com
username=bob
password=secr3t
url=https://go.dev
`,
			wantPrefix:   "https://go.dev",
			wantUsername: "bob",
			wantPassword: "secr3t",
		},
		{ // Empty data.
			gitauth: `
`,
			wantPrefix:   "",
			wantUsername: "",
			wantPassword: "",
		},
		{ // Does not follow the '=' format.
			gitauth: `
protocol:https
host:example.com
username:bob
password:secr3t
`,
			wantPrefix:   "",
			wantUsername: "",
			wantPassword: "",
		},
	}
	for _, tc := range testCases {
		parsedPrefix, username, password := parseGitAuth([]byte(tc.gitauth))
		if parsedPrefix != tc.wantPrefix {
			t.Errorf("parseGitAuth(%s):\nhave %q\nwant %q", tc.gitauth, parsedPrefix, tc.wantPrefix)
		}
		if username != tc.wantUsername {
			t.Errorf("parseGitAuth(%s):\nhave %q\nwant %q", tc.gitauth, username, tc.wantUsername)
		}
		if password != tc.wantPassword {
			t.Errorf("parseGitAuth(%s):\nhave %q\nwant %q", tc.gitauth, password, tc.wantPassword)
		}
	}
}
