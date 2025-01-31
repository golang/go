// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package auth provides access to user-provided authentication credentials.
package auth

import (
	"cmd/internal/quoted"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"os/exec"
	"strings"
)

// runAuthCommand executes a user provided GOAUTH command, parses its output, and
// returns a mapping of prefix → http.Header.
// It uses the client to verify the credential and passes the status to the
// command's stdin.
// res is used for the GOAUTH command's stdin.
func runAuthCommand(command string, url string, res *http.Response) (map[string]http.Header, error) {
	if command == "" {
		panic("GOAUTH invoked an empty authenticator command:" + command) // This should be caught earlier.
	}
	cmd, err := buildCommand(command)
	if err != nil {
		return nil, err
	}
	if url != "" {
		cmd.Args = append(cmd.Args, url)
	}
	cmd.Stderr = new(strings.Builder)
	if res != nil && writeResponseToStdin(cmd, res) != nil {
		return nil, fmt.Errorf("could not run command %s: %v\n%s", command, err, cmd.Stderr)
	}
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("could not run command %s: %v\n%s", command, err, cmd.Stderr)
	}
	credentials, err := parseUserAuth(string(out))
	if err != nil {
		return nil, fmt.Errorf("cannot parse output of GOAUTH command %s: %v", command, err)
	}
	return credentials, nil
}

// parseUserAuth parses the output from a GOAUTH command and
// returns a mapping of prefix → http.Header without the leading "https://"
// or an error if the data does not follow the expected format.
// Returns an nil error and an empty map if the data is empty.
// See the expected format in 'go help goauth'.
func parseUserAuth(data string) (map[string]http.Header, error) {
	credentials := make(map[string]http.Header)
	for data != "" {
		var line string
		var ok bool
		var urls []string
		// Parse URLS first.
		for {
			line, data, ok = strings.Cut(data, "\n")
			if !ok {
				return nil, fmt.Errorf("invalid format: missing empty line after URLs")
			}
			if line == "" {
				break
			}
			u, err := url.ParseRequestURI(line)
			if err != nil {
				return nil, fmt.Errorf("could not parse URL %s: %v", line, err)
			}
			urls = append(urls, u.String())
		}
		// Parse Headers second.
		header := make(http.Header)
		for {
			line, data, ok = strings.Cut(data, "\n")
			if !ok {
				return nil, fmt.Errorf("invalid format: missing empty line after headers")
			}
			if line == "" {
				break
			}
			name, value, ok := strings.Cut(line, ": ")
			value = strings.TrimSpace(value)
			if !ok || !validHeaderFieldName(name) || !validHeaderFieldValue(value) {
				return nil, fmt.Errorf("invalid format: invalid header line")
			}
			header.Add(name, value)
		}
		maps.Copy(credentials, mapHeadersToPrefixes(urls, header))
	}
	return credentials, nil
}

// mapHeadersToPrefixes returns a mapping of prefix → http.Header without
// the leading "https://".
func mapHeadersToPrefixes(prefixes []string, header http.Header) map[string]http.Header {
	prefixToHeaders := make(map[string]http.Header, len(prefixes))
	for _, p := range prefixes {
		p = strings.TrimPrefix(p, "https://")
		prefixToHeaders[p] = header.Clone() // Clone the header to avoid sharing
	}
	return prefixToHeaders
}

func buildCommand(command string) (*exec.Cmd, error) {
	words, err := quoted.Split(command)
	if err != nil {
		return nil, fmt.Errorf("cannot parse GOAUTH command %s: %v", command, err)
	}
	cmd := exec.Command(words[0], words[1:]...)
	return cmd, nil
}

// writeResponseToStdin writes the HTTP response to the command's stdin.
func writeResponseToStdin(cmd *exec.Cmd, res *http.Response) error {
	var output strings.Builder
	output.WriteString(res.Proto + " " + res.Status + "\n")
	for k, v := range res.Header {
		output.WriteString(k + ": " + strings.Join(v, ", ") + "\n")
	}
	output.WriteString("\n")
	cmd.Stdin = strings.NewReader(output.String())
	return nil
}
