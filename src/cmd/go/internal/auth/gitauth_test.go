// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package auth

import (
	"strings"
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

func BenchmarkParseGitAuth(b *testing.B) {
	// Define different test scenarios to benchmark
	testCases := []struct {
		name string
		data []byte
	}{{
		// Standard scenario with all basic fields present
		name: "standard",
		data: []byte(`
protocol=https
host=example.com
username=bob
password=secr3t
`),
	}, {
		// Scenario with URL field included
		name: "with_url",
		data: []byte(`
protocol=https
host=example.com
username=bob
password=secr3t
url=https://example.com/repo
`),
	}, {
		// Minimal scenario with only required fields
		name: "minimal",
		data: []byte(`
protocol=https
host=example.com
`),
	}, {
		// Complex scenario with longer values and extra fields
		name: "complex",
		data: func() []byte {
			var builder strings.Builder
			builder.WriteString("protocol=https\n")
			builder.WriteString("host=example.com\n")
			builder.WriteString("username=longusernamenamename\n")
			builder.WriteString("password=longpasswordwithmanycharacters123456789\n")
			builder.WriteString("url=https://example.com/very/long/path/to/repository\n")
			builder.WriteString("extra1=value1\n")
			builder.WriteString("extra2=value2\n")
			return []byte(builder.String())
		}(),
	}, {
		// Scenario with empty input
		name: "empty",
		data: []byte(``),
	}, {
		// Scenario with malformed input (using colon instead of equals)
		name: "malformed",
		data: []byte(`
protocol:https
host:example.com
username:bob
password:secr3t
`),
	}}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				prefix, username, password := parseGitAuth(tc.data)

				_ = prefix
				_ = username
				_ = password
			}
		})
	}
}
