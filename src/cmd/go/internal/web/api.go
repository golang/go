// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package web defines minimal helper routines for accessing HTTP/HTTPS
// resources without requiring external dependencies on the net package.
//
// If the cmd_go_bootstrap build tag is present, web avoids the use of the net
// package and returns errors for all network operations.
package web

import (
	"bytes"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"strings"
	"unicode"
	"unicode/utf8"
)

// SecurityMode specifies whether a function should make network
// calls using insecure transports (eg, plain text HTTP).
// The zero value is "secure".
type SecurityMode int

const (
	SecureOnly      SecurityMode = iota // Reject plain HTTP; validate HTTPS.
	DefaultSecurity                     // Allow plain HTTP if explicit; validate HTTPS.
	Insecure                            // Allow plain HTTP if not explicitly HTTPS; skip HTTPS validation.
)

// An HTTPError describes an HTTP error response (non-200 result).
type HTTPError struct {
	URL        string // redacted
	Status     string
	StatusCode int
	Err        error  // underlying error, if known
	Detail     string // limited to maxErrorDetailLines and maxErrorDetailBytes
}

const (
	maxErrorDetailLines = 8
	maxErrorDetailBytes = maxErrorDetailLines * 81
)

func (e *HTTPError) Error() string {
	if e.Detail != "" {
		detailSep := " "
		if strings.ContainsRune(e.Detail, '\n') {
			detailSep = "\n\t"
		}
		return fmt.Sprintf("reading %s: %v\n\tserver response:%s%s", e.URL, e.Status, detailSep, e.Detail)
	}

	if err := e.Err; err != nil {
		if pErr, ok := e.Err.(*fs.PathError); ok && strings.HasSuffix(e.URL, pErr.Path) {
			// Remove the redundant copy of the path.
			err = pErr.Err
		}
		return fmt.Sprintf("reading %s: %v", e.URL, err)
	}

	return fmt.Sprintf("reading %s: %v", e.URL, e.Status)
}

func (e *HTTPError) Is(target error) bool {
	return target == fs.ErrNotExist && (e.StatusCode == 404 || e.StatusCode == 410)
}

func (e *HTTPError) Unwrap() error {
	return e.Err
}

// GetBytes returns the body of the requested resource, or an error if the
// response status was not http.StatusOK.
//
// GetBytes is a convenience wrapper around Get and Response.Err.
func GetBytes(u *url.URL) ([]byte, error) {
	resp, err := Get(DefaultSecurity, u)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if err := resp.Err(); err != nil {
		return nil, err
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %v", u.Redacted(), err)
	}
	return b, nil
}

type Response struct {
	URL        string // redacted
	Status     string
	StatusCode int
	Header     map[string][]string
	Body       io.ReadCloser // Either the original body or &errorDetail.

	fileErr     error
	errorDetail errorDetailBuffer
}

// Err returns an *HTTPError corresponding to the response r.
// If the response r has StatusCode 200 or 0 (unset), Err returns nil.
// Otherwise, Err may read from r.Body in order to extract relevant error detail.
func (r *Response) Err() error {
	if r.StatusCode == 200 || r.StatusCode == 0 {
		return nil
	}

	return &HTTPError{
		URL:        r.URL,
		Status:     r.Status,
		StatusCode: r.StatusCode,
		Err:        r.fileErr,
		Detail:     r.formatErrorDetail(),
	}
}

// formatErrorDetail converts r.errorDetail (a prefix of the output of r.Body)
// into a short, tab-indented summary.
func (r *Response) formatErrorDetail() string {
	if r.Body != &r.errorDetail {
		return "" // Error detail collection not enabled.
	}

	// Ensure that r.errorDetail has been populated.
	_, _ = io.Copy(io.Discard, r.Body)

	s := r.errorDetail.buf.String()
	if !utf8.ValidString(s) {
		return "" // Don't try to recover non-UTF-8 error messages.
	}
	for _, r := range s {
		if !unicode.IsGraphic(r) && !unicode.IsSpace(r) {
			return "" // Don't let the server do any funny business with the user's terminal.
		}
	}

	var detail strings.Builder
	for i, line := range strings.Split(s, "\n") {
		if strings.TrimSpace(line) == "" {
			break // Stop at the first blank line.
		}
		if i > 0 {
			detail.WriteString("\n\t")
		}
		if i >= maxErrorDetailLines {
			detail.WriteString("[Truncated: too many lines.]")
			break
		}
		if detail.Len()+len(line) > maxErrorDetailBytes {
			detail.WriteString("[Truncated: too long.]")
			break
		}
		detail.WriteString(line)
	}

	return detail.String()
}

// Get returns the body of the HTTP or HTTPS resource specified at the given URL.
//
// If the URL does not include an explicit scheme, Get first tries "https".
// If the server does not respond under that scheme and the security mode is
// Insecure, Get then tries "http".
// The URL included in the response indicates which scheme was actually used,
// and it is a redacted URL suitable for use in error messages.
//
// For the "https" scheme only, credentials are attached using the
// cmd/go/internal/auth package. If the URL itself includes a username and
// password, it will not be attempted under the "http" scheme unless the
// security mode is Insecure.
//
// Get returns a non-nil error only if the request did not receive a response
// under any applicable scheme. (A non-2xx response does not cause an error.)
func Get(security SecurityMode, u *url.URL) (*Response, error) {
	return get(security, u)
}

// OpenBrowser attempts to open the requested URL in a web browser.
func OpenBrowser(url string) (opened bool) {
	return openBrowser(url)
}

// Join returns the result of adding the slash-separated
// path elements to the end of u's path.
func Join(u *url.URL, path string) *url.URL {
	j := *u
	if path == "" {
		return &j
	}
	j.Path = strings.TrimSuffix(u.Path, "/") + "/" + strings.TrimPrefix(path, "/")
	j.RawPath = strings.TrimSuffix(u.RawPath, "/") + "/" + strings.TrimPrefix(path, "/")
	return &j
}

// An errorDetailBuffer is an io.ReadCloser that copies up to
// maxErrorDetailLines into a buffer for later inspection.
type errorDetailBuffer struct {
	r        io.ReadCloser
	buf      strings.Builder
	bufLines int
}

func (b *errorDetailBuffer) Close() error {
	return b.r.Close()
}

func (b *errorDetailBuffer) Read(p []byte) (n int, err error) {
	n, err = b.r.Read(p)

	// Copy the first maxErrorDetailLines+1 lines into b.buf,
	// discarding any further lines.
	//
	// Note that the read may begin or end in the middle of a UTF-8 character,
	// so don't try to do anything fancy with characters that encode to larger
	// than one byte.
	if b.bufLines <= maxErrorDetailLines {
		for _, line := range bytes.SplitAfterN(p[:n], []byte("\n"), maxErrorDetailLines-b.bufLines) {
			b.buf.Write(line)
			if len(line) > 0 && line[len(line)-1] == '\n' {
				b.bufLines++
				if b.bufLines > maxErrorDetailLines {
					break
				}
			}
		}
	}

	return n, err
}
