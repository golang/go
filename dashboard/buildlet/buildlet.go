// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The buildlet is an HTTP server that untars content to disk and runs
// commands it has untarred, streaming their output back over HTTP.
// It is part of Go's continuous build system.
//
// This program intentionally allows remote code execution, and
// provides no security of its own. It is assumed that any user uses
// it with an appropriately-configured firewall between their VM
// instances.
package main // import "golang.org/x/tools/dashboard/buildlet"

/* Notes:

https://go.googlesource.com/go/+archive/3b76b017cabb.tar.gz
curl -X PUT --data-binary "@go-3b76b017cabb.tar.gz" http://127.0.0.1:5937/writetgz

curl -d "cmd=src/make.bash" http://127.0.0.1:5937/exec

*/

import (
	"archive/tar"
	"compress/gzip"
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
)

var (
	scratchDir = flag.String("scratchdir", "", "Temporary directory to use. The contents of this directory may be deleted at any time. If empty, TempDir is used to create one.")
	listenAddr = flag.String("listen", defaultListenAddr(), "address to listen on. Warning: this service is inherently insecure and offers no protection of its own. Do not expose this port to the world.")
)

func defaultListenAddr() string {
	if OnGCE() {
		// In production, default to
		return ":80"
	}
	return "localhost:5936"
}

func main() {
	flag.Parse()
	if !OnGCE() && !strings.HasPrefix(*listenAddr, "localhost:") {
		log.Printf("** WARNING ***  This server is unsafe and offers no security. Be careful.")
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
	http.HandleFunc("/writetgz", handleWriteTGZ)
	http.HandleFunc("/exec", handleExec)
	http.HandleFunc("/", handleRoot)
	// TODO: removeall
	log.Printf("Listening on %s ...", *listenAddr)
	log.Fatalf("ListenAndServe: %v", http.ListenAndServe(*listenAddr, nil))
}

func handleRoot(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "buildlet running on %s-%s", runtime.GOOS, runtime.GOARCH)
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

// metaClient to fetch GCE metadata values.
var metaClient = &http.Client{
	Transport: &http.Transport{
		Dial: (&net.Dialer{
			Timeout:   750 * time.Millisecond,
			KeepAlive: 30 * time.Second,
		}).Dial,
		ResponseHeaderTimeout: 750 * time.Millisecond,
	},
}

var onGCE struct {
	sync.Mutex
	set bool
	v   bool
}

// OnGCE reports whether this process is running on Google Compute Engine.
func OnGCE() bool {
	defer onGCE.Unlock()
	onGCE.Lock()
	if onGCE.set {
		return onGCE.v
	}
	onGCE.set = true

	res, err := metaClient.Get("http://metadata.google.internal")
	if err != nil {
		return false
	}
	onGCE.v = res.Header.Get("Metadata-Flavor") == "Google"
	return onGCE.v
}
