// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build buildlet

// The buildlet is an HTTP server that untars content to disk and runs
// commands it has untarred, streaming their output back over HTTP.
// It is part of Go's continuous build system.
//
// This program intentionally allows remote code execution, and
// provides no security of its own. It is assumed that any user uses
// it with an appropriately-configured firewall between their VM
// instances.
package main // import "golang.org/x/tools/dashboard/buildlet"

import (
	"archive/tar"
	"compress/gzip"
	"crypto/tls"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"google.golang.org/cloud/compute/metadata"
)

var (
	scratchDir = flag.String("scratchdir", "", "Temporary directory to use. The contents of this directory may be deleted at any time. If empty, TempDir is used to create one.")
	listenAddr = flag.String("listen", defaultListenAddr(), "address to listen on. Warning: this service is inherently insecure and offers no protection of its own. Do not expose this port to the world.")
)

func defaultListenAddr() string {
	if runtime.GOOS == "darwin" {
		// Darwin will never run on GCE, so let's always
		// listen on a high port (so we don't need to be
		// root).
		return ":5936"
	}
	if !metadata.OnGCE() {
		return "localhost:5936"
	}
	// In production, default to port 80 or 443, depending on
	// whether TLS is configured.
	if metadataValue("tls-cert") != "" {
		return ":443"
	}
	return ":80"
}

func main() {
	flag.Parse()
	if !metadata.OnGCE() && !strings.HasPrefix(*listenAddr, "localhost:") {
		log.Printf("** WARNING ***  This server is unsafe and offers no security. Be careful.")
	}
	if runtime.GOOS == "plan9" {
		// Plan 9 is too slow on GCE, so stop running run.rc after the basics.
		// See https://golang.org/cl/2522 and https://golang.org/issue/9491
		// TODO(bradfitz): once the buildlet has environment variable support,
		// the coordinator can send this in, and this variable can be part of
		// the build configuration struct instead of hard-coded here.
		// But no need for environment variables quite yet.
		os.Setenv("GOTESTONLY", "std")
	}

	if *scratchDir == "" {
		dir, err := ioutil.TempDir("", "buildlet-scatch")
		if err != nil {
			log.Fatalf("error creating scratchdir with ioutil.TempDir: %v", err)
		}
		*scratchDir = dir
	}
	if _, err := os.Lstat(*scratchDir); err != nil {
		log.Fatalf("invalid --scratchdir %q: %v", *scratchDir, err)
	}
	http.HandleFunc("/", handleRoot)

	password := metadataValue("password")
	http.Handle("/writetgz", requirePassword{http.HandlerFunc(handleWriteTGZ), password})
	http.Handle("/exec", requirePassword{http.HandlerFunc(handleExec), password})
	// TODO: removeall

	tlsCert, tlsKey := metadataValue("tls-cert"), metadataValue("tls-key")
	if (tlsCert == "") != (tlsKey == "") {
		log.Fatalf("tls-cert and tls-key must both be supplied, or neither.")
	}

	log.Printf("Listening on %s ...", *listenAddr)
	ln, err := net.Listen("tcp", *listenAddr)
	if err != nil {
		log.Fatalf("Failed to listen on %s: %v", *listenAddr, err)
	}
	ln = tcpKeepAliveListener{ln.(*net.TCPListener)}

	var srv http.Server
	if tlsCert != "" {
		cert, err := tls.X509KeyPair([]byte(tlsCert), []byte(tlsKey))
		if err != nil {
			log.Fatalf("TLS cert error: %v", err)
		}
		tlsConf := &tls.Config{
			Certificates: []tls.Certificate{cert},
		}
		ln = tls.NewListener(ln, tlsConf)
	}

	log.Fatalf("Serve: %v", srv.Serve(ln))
}

// metadataValue returns the GCE metadata instance value for the given key.
// If the metadata is not defined, the returned string is empty.
//
// If not running on GCE, it falls back to using environment variables
// for local development.
func metadataValue(key string) string {
	// The common case:
	if metadata.OnGCE() {
		v, err := metadata.InstanceAttributeValue(key)
		if _, notDefined := err.(metadata.NotDefinedError); notDefined {
			return ""
		}
		if err != nil {
			log.Fatalf("metadata.InstanceAttributeValue(%q): %v", key, err)
		}
		return v
	}

	// Else let developers use environment variables to fake
	// metadata keys, for local testing.
	envKey := "GCEMETA_" + strings.Replace(key, "-", "_", -1)
	v := os.Getenv(envKey)
	// Respect curl-style '@' prefix to mean the rest is a filename.
	if strings.HasPrefix(v, "@") {
		slurp, err := ioutil.ReadFile(v[1:])
		if err != nil {
			log.Fatalf("Error reading file for GCEMETA_%v: %v", key, err)
		}
		return string(slurp)
	}
	if v == "" {
		log.Printf("Warning: not running on GCE, and no %v environment variable defined", envKey)
	}
	return v
}

// tcpKeepAliveListener is a net.Listener that sets TCP keep-alive
// timeouts on accepted connections.
type tcpKeepAliveListener struct {
	*net.TCPListener
}

func (ln tcpKeepAliveListener) Accept() (c net.Conn, err error) {
	tc, err := ln.AcceptTCP()
	if err != nil {
		return
	}
	tc.SetKeepAlive(true)
	tc.SetKeepAlivePeriod(3 * time.Minute)
	return tc, nil
}

func handleRoot(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "buildlet running on %s-%s\n", runtime.GOOS, runtime.GOARCH)
}

func handleWriteTGZ(w http.ResponseWriter, r *http.Request) {
	if r.Method != "PUT" {
		http.Error(w, "requires PUT method", http.StatusBadRequest)
		return
	}
	err := untar(r.Body, *scratchDir)
	if err != nil {
		status := http.StatusInternalServerError
		if he, ok := err.(httpStatuser); ok {
			status = he.httpStatus()
		}
		http.Error(w, err.Error(), status)
		return
	}
	io.WriteString(w, "OK")
}

// untar reads the gzip-compressed tar file from r and writes it into dir.
func untar(r io.Reader, dir string) error {
	zr, err := gzip.NewReader(r)
	if err != nil {
		return badRequest("requires gzip-compressed body: " + err.Error())
	}
	tr := tar.NewReader(zr)
	for {
		f, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("tar reading error: %v", err)
			return badRequest("tar error: " + err.Error())
		}
		if !validRelPath(f.Name) {
			return badRequest(fmt.Sprintf("tar file contained invalid name %q", f.Name))
		}
		rel := filepath.FromSlash(f.Name)
		abs := filepath.Join(dir, rel)

		fi := f.FileInfo()
		mode := fi.Mode()
		switch {
		case mode.IsRegular():
			// Make the directory. This is redundant because it should
			// already be made by a directory entry in the tar
			// beforehand. Thus, don't check for errors; the next
			// write will fail with the same error.
			os.MkdirAll(filepath.Dir(abs), 0755)
			wf, err := os.OpenFile(abs, os.O_RDWR|os.O_CREATE|os.O_TRUNC, mode.Perm())
			if err != nil {
				return err
			}
			n, err := io.Copy(wf, tr)
			if closeErr := wf.Close(); closeErr != nil && err == nil {
				err = closeErr
			}
			if err != nil {
				return fmt.Errorf("error writing to %s: %v", abs, err)
			}
			if n != f.Size {
				return fmt.Errorf("only wrote %d bytes to %s; expected %d", n, abs, f.Size)
			}
			log.Printf("wrote %s", abs)
		case mode.IsDir():
			if err := os.MkdirAll(abs, 0755); err != nil {
				return err
			}
		default:
			return badRequest(fmt.Sprintf("tar file entry %s contained unsupported file type %v", f.Name, mode))
		}
	}
	return nil
}

// Process-State is an HTTP Trailer set in the /exec handler to "ok"
// on success, or os.ProcessState.String() on failure.
const hdrProcessState = "Process-State"

func handleExec(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "requires POST method", http.StatusBadRequest)
		return
	}
	if r.ProtoMajor*10+r.ProtoMinor < 11 {
		// We need trailers, only available in HTTP/1.1 or HTTP/2.
		http.Error(w, "HTTP/1.1 or higher required", http.StatusBadRequest)
		return
	}

	w.Header().Set("Trailer", hdrProcessState) // declare it so we can set it

	cmdPath := r.FormValue("cmd") // required
	if !validRelPath(cmdPath) {
		http.Error(w, "requires 'cmd' parameter", http.StatusBadRequest)
		return
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	absCmd := filepath.Join(*scratchDir, filepath.FromSlash(cmdPath))
	cmd := exec.Command(absCmd, r.PostForm["cmdArg"]...)
	cmd.Dir = filepath.Dir(absCmd)
	cmdOutput := &flushWriter{w: w}
	cmd.Stdout = cmdOutput
	cmd.Stderr = cmdOutput
	err := cmd.Run()
	state := "ok"
	if err != nil {
		if ps := cmd.ProcessState; ps != nil {
			state = ps.String()
		} else {
			state = err.Error()
		}
	}
	w.Header().Set(hdrProcessState, state)
	log.Printf("Run = %s", state)
}

// flushWriter is an io.Writer wrapper that writes to w and
// Flushes the output immediately, if w is an http.Flusher.
type flushWriter struct {
	mu sync.Mutex
	w  http.ResponseWriter
}

func (hw *flushWriter) Write(p []byte) (n int, err error) {
	hw.mu.Lock()
	defer hw.mu.Unlock()
	n, err = hw.w.Write(p)
	if f, ok := hw.w.(http.Flusher); ok {
		f.Flush()
	}
	return
}

func validRelPath(p string) bool {
	if p == "" || strings.Contains(p, `\`) || strings.HasPrefix(p, "/") || strings.Contains(p, "../") {
		return false
	}
	return true
}

type httpStatuser interface {
	error
	httpStatus() int
}

type httpError struct {
	statusCode int
	msg        string
}

func (he httpError) Error() string   { return he.msg }
func (he httpError) httpStatus() int { return he.statusCode }

func badRequest(msg string) error {
	return httpError{http.StatusBadRequest, msg}
}

// requirePassword is an http.Handler auth wrapper that enforces a
// HTTP Basic password. The username is ignored.
type requirePassword struct {
	h        http.Handler
	password string // empty means no password
}

func (h requirePassword) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	_, gotPass, _ := r.BasicAuth()
	if h.password != "" && h.password != gotPass {
		http.Error(w, "invalid password", http.StatusForbidden)
		return
	}
	h.h.ServeHTTP(w, r)
}
