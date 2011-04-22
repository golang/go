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
	"linux":   []string{"LD_LIBRARY_PATH"},
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
	headers := rw.Header()
	statusCode := http.StatusOK
	for {
		line, isPrefix, err := linebody.ReadLine()
		if isPrefix {
			rw.WriteHeader(http.StatusInternalServerError)
			h.printf("CGI: long header line from subprocess.")
			return
		}
		if err == os.EOF {
			break
		}
		if err != nil {
			rw.WriteHeader(http.StatusInternalServerError)
			h.printf("CGI: error reading headers: %v", err)
			return
		}
		if len(line) == 0 {
			break
		}
		parts := strings.Split(string(line), ":", 2)
		if len(parts) < 2 {
			h.printf("CGI: bogus header line: %s", string(line))
			continue
		}
		header, val := parts[0], parts[1]
		header = strings.TrimSpace(header)
		val = strings.TrimSpace(val)
		switch {
		case header == "Status":
			if len(val) < 3 {
				h.printf("CGI: bogus status (short): %q", val)
				return
			}
			code, err := strconv.Atoi(val[0:3])
			if err != nil {
				h.printf("CGI: bogus status: %q", val)
				h.printf("CGI: line was %q", line)
				return
			}
			statusCode = code
		default:
			headers.Add(header, val)
		}
	}
	rw.WriteHeader(statusCode)

	_, err = io.Copy(rw, linebody)
	if err != nil {
		h.printf("CGI: copy error: %v", err)
	}
}

func (h *Handler) printf(format string, v ...interface{}) {
	if h.Logger != nil {
		h.Logger.Printf(format, v...)
	} else {
		log.Printf(format, v...)
	}
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
