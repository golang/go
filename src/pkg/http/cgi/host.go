// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the host side of CGI (being the webserver
// parent process).

// Package cgi implements CGI (Common Gateway Interface) as specified
// in RFC 3875.
//
// Note that using CGI means starting a new process to handle each
// request, which is typically less efficient than using a
// long-running server.  This package is intended primarily for
// compatibility with existing systems.
package cgi

import (
	"bufio"
	"bytes"
	"exec"
	"fmt"
	"http"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

var trailingPort = regexp.MustCompile(`:([0-9]+)$`)

var osDefaultInheritEnv = map[string][]string{
	"darwin":  []string{"DYLD_LIBRARY_PATH"},
	"freebsd": []string{"LD_LIBRARY_PATH"},
	"hpux":    []string{"LD_LIBRARY_PATH", "SHLIB_PATH"},
	"irix":    []string{"LD_LIBRARY_PATH", "LD_LIBRARYN32_PATH", "LD_LIBRARY64_PATH"},
	"linux":   []string{"LD_LIBRARY_PATH"},
	"solaris": []string{"LD_LIBRARY_PATH", "LD_LIBRARY_PATH_32", "LD_LIBRARY_PATH_64"},
	"windows": []string{"SystemRoot", "COMSPEC", "PATHEXT", "WINDIR"},
}

// Handler runs an executable in a subprocess with a CGI environment.
type Handler struct {
	Path string // path to the CGI executable
	Root string // root URI prefix of handler or empty for "/"

	Env        []string    // extra environment variables to set, if any, as "key=value"
	InheritEnv []string    // environment variables to inherit from host, as "key"
	Logger     *log.Logger // optional log for errors or nil to use log.Print
	Args       []string    // optional arguments to pass to child process

	// PathLocationHandler specifies the root http Handler that
	// should handle internal redirects when the CGI process
	// returns a Location header value starting with a "/", as
	// specified in RFC 3875 § 6.3.2. This will likely be
	// http.DefaultServeMux.
	//
	// If nil, a CGI response with a local URI path is instead sent
	// back to the client and not redirected internally.
	PathLocationHandler http.Handler
}

func (h *Handler) ServeHTTP(rw http.ResponseWriter, req *http.Request) {
	root := h.Root
	if root == "" {
		root = "/"
	}

	if len(req.TransferEncoding) > 0 && req.TransferEncoding[0] == "chunked" {
		rw.WriteHeader(http.StatusBadRequest)
		rw.Write([]byte("Chunked request bodies are not supported by CGI."))
		return
	}

	pathInfo := req.URL.Path
	if root != "/" && strings.HasPrefix(pathInfo, root) {
		pathInfo = pathInfo[len(root):]
	}

	port := "80"
	if matches := trailingPort.FindStringSubmatch(req.Host); len(matches) != 0 {
		port = matches[1]
	}

	env := []string{
		"SERVER_SOFTWARE=go",
		"SERVER_NAME=" + req.Host,
		"SERVER_PROTOCOL=HTTP/1.1",
		"HTTP_HOST=" + req.Host,
		"GATEWAY_INTERFACE=CGI/1.1",
		"REQUEST_METHOD=" + req.Method,
		"QUERY_STRING=" + req.URL.RawQuery,
		"REQUEST_URI=" + req.URL.RawPath,
		"PATH_INFO=" + pathInfo,
		"SCRIPT_NAME=" + root,
		"SCRIPT_FILENAME=" + h.Path,
		"REMOTE_ADDR=" + req.RemoteAddr,
		"REMOTE_HOST=" + req.RemoteAddr,
		"SERVER_PORT=" + port,
	}

	if req.TLS != nil {
		env = append(env, "HTTPS=on")
	}

	if len(req.Cookie) > 0 {
		b := new(bytes.Buffer)
		for idx, c := range req.Cookie {
			if idx > 0 {
				b.Write([]byte("; "))
			}
			fmt.Fprintf(b, "%s=%s", c.Name, c.Value)
		}
		env = append(env, "HTTP_COOKIE="+b.String())
	}

	for k, v := range req.Header {
		k = strings.Map(upperCaseAndUnderscore, k)
		env = append(env, "HTTP_"+k+"="+strings.Join(v, ", "))
	}

	if req.ContentLength > 0 {
		env = append(env, fmt.Sprintf("CONTENT_LENGTH=%d", req.ContentLength))
	}
	if ctype := req.Header.Get("Content-Type"); ctype != "" {
		env = append(env, "CONTENT_TYPE="+ctype)
	}

	if h.Env != nil {
		env = append(env, h.Env...)
	}

	path := os.Getenv("PATH")
	if path == "" {
		path = "/bin:/usr/bin:/usr/ucb:/usr/bsd:/usr/local/bin"
	}
	env = append(env, "PATH="+path)

	for _, e := range h.InheritEnv {
		if v := os.Getenv(e); v != "" {
			env = append(env, e+"="+v)
		}
	}

	for _, e := range osDefaultInheritEnv[runtime.GOOS] {
		if v := os.Getenv(e); v != "" {
			env = append(env, e+"="+v)
		}
	}

	cwd, pathBase := filepath.Split(h.Path)
	if cwd == "" {
		cwd = "."
	}

	args := []string{h.Path}
	args = append(args, h.Args...)

	cmd, err := exec.Run(
		pathBase,
		args,
		env,
		cwd,
		exec.Pipe,        // stdin
		exec.Pipe,        // stdout
		exec.PassThrough, // stderr (for now)
	)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		h.printf("CGI error: %v", err)
		return
	}
	defer func() {
		cmd.Stdin.Close()
		cmd.Stdout.Close()
		cmd.Wait(0) // no zombies
	}()

	if req.ContentLength != 0 {
		go io.Copy(cmd.Stdin, req.Body)
	}

	linebody, _ := bufio.NewReaderSize(cmd.Stdout, 1024)
	headers := make(http.Header)
	statusCode := 0
	for {
		line, isPrefix, err := linebody.ReadLine()
		if isPrefix {
			rw.WriteHeader(http.StatusInternalServerError)
			h.printf("cgi: long header line from subprocess.")
			return
		}
		if err == os.EOF {
			break
		}
		if err != nil {
			rw.WriteHeader(http.StatusInternalServerError)
			h.printf("cgi: error reading headers: %v", err)
			return
		}
		if len(line) == 0 {
			break
		}
		parts := strings.Split(string(line), ":", 2)
		if len(parts) < 2 {
			h.printf("cgi: bogus header line: %s", string(line))
			continue
		}
		header, val := parts[0], parts[1]
		header = strings.TrimSpace(header)
		val = strings.TrimSpace(val)
		switch {
		case header == "Status":
			if len(val) < 3 {
				h.printf("cgi: bogus status (short): %q", val)
				return
			}
			code, err := strconv.Atoi(val[0:3])
			if err != nil {
				h.printf("cgi: bogus status: %q", val)
				h.printf("cgi: line was %q", line)
				return
			}
			statusCode = code
		default:
			headers.Add(header, val)
		}
	}

	if loc := headers.Get("Location"); loc != "" {
		if strings.HasPrefix(loc, "/") && h.PathLocationHandler != nil {
			h.handleInternalRedirect(rw, req, loc)
			return
		}
		if statusCode == 0 {
			statusCode = http.StatusFound
		}
	}

	if statusCode == 0 {
		statusCode = http.StatusOK
	}

	// Copy headers to rw's headers, after we've decided not to
	// go into handleInternalRedirect, which won't want its rw
	// headers to have been touched.
	for k, vv := range headers {
		for _, v := range vv {
			rw.Header().Add(k, v)
		}
	}

	rw.WriteHeader(statusCode)

	_, err = io.Copy(rw, linebody)
	if err != nil {
		h.printf("cgi: copy error: %v", err)
	}
}

func (h *Handler) printf(format string, v ...interface{}) {
	if h.Logger != nil {
		h.Logger.Printf(format, v...)
	} else {
		log.Printf(format, v...)
	}
}

func (h *Handler) handleInternalRedirect(rw http.ResponseWriter, req *http.Request, path string) {
	url, err := req.URL.ParseURL(path)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		h.printf("cgi: error resolving local URI path %q: %v", path, err)
		return
	}
	// TODO: RFC 3875 isn't clear if only GET is supported, but it
	// suggests so: "Note that any message-body attached to the
	// request (such as for a POST request) may not be available
	// to the resource that is the target of the redirect."  We
	// should do some tests against Apache to see how it handles
	// POST, HEAD, etc. Does the internal redirect get the same
	// method or just GET? What about incoming headers?
	// (e.g. Cookies) Which headers, if any, are copied into the
	// second request?
	newReq := &http.Request{
		Method:     "GET",
		URL:        url,
		RawURL:     path,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     make(http.Header),
		Host:       url.Host,
		RemoteAddr: req.RemoteAddr,
		TLS:        req.TLS,
	}
	h.PathLocationHandler.ServeHTTP(rw, newReq)
}

func upperCaseAndUnderscore(rune int) int {
	switch {
	case rune >= 'a' && rune <= 'z':
		return rune - ('a' - 'A')
	case rune == '-':
		return '_'
	case rune == '=':
		// Maybe not part of the CGI 'spec' but would mess up
		// the environment in any case, as Go represents the
		// environment as a slice of "key=value" strings.
		return '_'
	}
	// TODO: other transformations in spec or practice?
	return rune
}
