// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package auth provides access to user-provided authentication credentials.
package auth

import (
	"bufio"
	"bytes"
	"cmd/internal/quoted"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/textproto"
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
	credentials, err := parseUserAuth(bytes.NewReader(out))
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
func parseUserAuth(data io.Reader) (map[string]http.Header, error) {
	credentials := make(map[string]http.Header)
	reader := textproto.NewReader(bufio.NewReader(data))
	for {
		// Return the processed credentials if the reader is at EOF.
		if _, err := reader.R.Peek(1); err == io.EOF {
			return credentials, nil
		}
		urls, err := readURLs(reader)
		if err != nil {
			return nil, err
		}
		if len(urls) == 0 {
			return nil, fmt.Errorf("invalid format: expected url prefix")
		}
		mimeHeader, err := reader.ReadMIMEHeader()
		if err != nil {
			return nil, err
		}
		header := http.Header(mimeHeader)
		// Process the block (urls and headers).
		credentialMap := mapHeadersToPrefixes(urls, header)
		maps.Copy(credentials, credentialMap)
	}
}

// readURLs reads URL prefixes from the given reader until an empty line
// is encountered or an error occurs. It returns the list of URLs or an error
// if the format is invalid.
func readURLs(reader *textproto.Reader) (urls []string, err error) {
	for {
		line, err := reader.ReadLine()
		if err != nil {
			return nil, err
		}
		trimmedLine := strings.TrimSpace(line)
		if trimmedLine != line {
			return nil, fmt.Errorf("invalid format: leading or trailing white space")
		}
		if strings.HasPrefix(line, "https://") {
			urls = append(urls, line)
		} else if line == "" {
			return urls, nil
		} else {
			return nil, fmt.Errorf("invalid format: expected url prefix or empty line")
		}
	}
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
	if err := res.Header.Write(&output); err != nil {
		return err
	}
	output.WriteString("\n")
	cmd.Stdin = strings.NewReader(output.String())
	return nil
}
