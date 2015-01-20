// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

// The buildlet is an HTTP server that untars content to disk and runs
// commands it has untarred, streaming their output back over HTTP.
// It is part of Go's continuous build system.
//
// This program intentionally allows remote code execution, and
// provides no security of its own. It is assumed that any user uses
// it with an appropriately-configured firewall between their VM
// instances.
package main // import "golang.org/x/tools/dashboard/cmd/buildlet"

import (
	"archive/tar"
	"compress/gzip"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"google.golang.org/cloud/compute/metadata"
)

var (
	haltEntireOS = flag.Bool("halt", true, "halt OS in /halt handler. If false, the buildlet process just ends.")
	scratchDir   = flag.String("scratchdir", "", "Temporary directory to use. The contents of this directory may be deleted at any time. If empty, TempDir is used to create one.")
	listenAddr   = flag.String("listen", defaultListenAddr(), "address to listen on. Warning: this service is inherently insecure and offers no protection of its own. Do not expose this port to the world.")
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

var osHalt func() // set by some machines

func main() {
	flag.Parse()
	onGCE := metadata.OnGCE()
	if !onGCE && !strings.HasPrefix(*listenAddr, "localhost:") {
		log.Printf("** WARNING ***  This server is unsafe and offers no security. Be careful.")
	}
	if onGCE {
		fixMTU()
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
	// TODO(bradfitz): if this becomes more of a general tool,
	// perhaps we want to remove this hard-coded here. Also,
	// if/once the exec handler ever gets generic environment
	// variable support, it would make sense to remove this too
	// and push it to the client.  This hard-codes policy. But
	// that's okay for now.
	os.Setenv("GOROOT_BOOTSTRAP", filepath.Join(*scratchDir, "go1.4"))
	os.Setenv("WORKDIR", *scratchDir) // mostly for demos

	if _, err := os.Lstat(*scratchDir); err != nil {
		log.Fatalf("invalid --scratchdir %q: %v", *scratchDir, err)
	}
	http.HandleFunc("/", handleRoot)
	http.HandleFunc("/debug/goroutines", handleGoroutines)
	http.HandleFunc("/debug/x", handleX)

	password := metadataValue("password")
	requireAuth := func(handler func(w http.ResponseWriter, r *http.Request)) http.Handler {
		return requirePasswordHandler{http.HandlerFunc(handler), password}
	}
	http.Handle("/writetgz", requireAuth(handleWriteTGZ))
	http.Handle("/exec", requireAuth(handleExec))
	http.Handle("/halt", requireAuth(handleHalt))
	http.Handle("/tgz", requireAuth(handleGetTGZ))
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

func fixMTU_freebsd() error { return fixMTU_ifconfig("vtnet0") }
func fixMTU_openbsd() error { return fixMTU_ifconfig("vio0") }
func fixMTU_ifconfig(iface string) error {
	out, err := exec.Command("/sbin/ifconfig", iface, "mtu", "1460").CombinedOutput()
	if err != nil {
		return fmt.Errorf("/sbin/ifconfig %s mtu 1460: %v, %s", iface, err, out)
	}
	return nil
}

func fixMTU_plan9() error {
	f, err := os.OpenFile("/net/ipifc/0/ctl", os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(f, "mtu 1400\n"); err != nil { // not 1460
		f.Close()
		return err
	}
	return f.Close()
}

func fixMTU() {
	fn, ok := map[string]func() error{
		"openbsd": fixMTU_openbsd,
		"freebsd": fixMTU_freebsd,
		"plan9":   fixMTU_plan9,
	}[runtime.GOOS]
	if ok {
		if err := fn(); err != nil {
			log.Printf("Failed to set MTU: %v", err)
		} else {
			log.Printf("Adjusted MTU.")
		}
	}
}

// mtuWriter is a hack for environments where we can't (or can't yet)
// fix the machine's MTU.
// Instead of telling the operating system the MTU, we just cut up our
// writes into small pieces to make sure we don't get too near the
// MTU, and we hope the kernel doesn't coalesce different flushed
// writes back together into the same TCP IP packets.
type mtuWriter struct {
	rw http.ResponseWriter
}

func (mw mtuWriter) Write(p []byte) (n int, err error) {
	const mtu = 1000 // way less than 1460; since HTTP response headers might be in there too
	for len(p) > 0 {
		chunk := p
		if len(chunk) > mtu {
			chunk = p[:mtu]
		}
		n0, err := mw.rw.Write(chunk)
		n += n0
		if n0 != len(chunk) && err == nil {
			err = io.ErrShortWrite
		}
		if err != nil {
			return n, err
		}
		p = p[n0:]
		mw.rw.(http.Flusher).Flush()
		if len(p) > 0 {
			// Whitelisted operating systems:
			if runtime.GOOS == "openbsd" || runtime.GOOS == "linux" {
				// Nothing
			} else {
				// Try to prevent the kernel from Nagel-ing the IP packets
				// together into one that's too large.
				time.Sleep(250 * time.Millisecond)
			}
		}
	}
	return n, nil
}

func handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	fmt.Fprintf(w, "buildlet running on %s-%s\n", runtime.GOOS, runtime.GOARCH)
}

// unauthenticated /debug/goroutines handler
func handleGoroutines(rw http.ResponseWriter, r *http.Request) {
	w := mtuWriter{rw}
	log.Printf("Dumping goroutines.")
	rw.Header().Set("Content-Type", "text/plain; charset=utf-8")
	buf := make([]byte, 2<<20)
	buf = buf[:runtime.Stack(buf, true)]
	w.Write(buf)
	log.Printf("Dumped goroutines.")
}

// unauthenticated /debug/x handler, to test MTU settings.
func handleX(w http.ResponseWriter, r *http.Request) {
	n, _ := strconv.Atoi(r.FormValue("n"))
	if n > 1<<20 {
		n = 1 << 20
	}
	log.Printf("Dumping %d X.", n)
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	buf := make([]byte, n)
	for i := range buf {
		buf[i] = 'X'
	}
	w.Write(buf)
	log.Printf("Dumped X.")
}

// This is a remote code execution daemon, so security is kinda pointless, but:
func validRelativeDir(dir string) bool {
	if strings.Contains(dir, `\`) || path.IsAbs(dir) {
		return false
	}
	dir = path.Clean(dir)
	if strings.HasPrefix(dir, "../") || strings.HasSuffix(dir, "/..") || dir == ".." {
		return false
	}
	return true
}

func handleGetTGZ(rw http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(rw, "requires GET method", http.StatusBadRequest)
		return
	}
	dir := r.FormValue("dir")
	if !validRelativeDir(dir) {
		http.Error(rw, "bogus dir", http.StatusBadRequest)
		return
	}
	zw := gzip.NewWriter(mtuWriter{rw})
	tw := tar.NewWriter(zw)
	base := filepath.Join(*scratchDir, filepath.FromSlash(dir))
	err := filepath.Walk(base, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel := strings.TrimPrefix(strings.TrimPrefix(path, base), "/")
		var linkName string
		if fi.Mode()&os.ModeSymlink != 0 {
			linkName, err = os.Readlink(path)
			if err != nil {
				return err
			}
		}
		th, err := tar.FileInfoHeader(fi, linkName)
		if err != nil {
			return err
		}
		th.Name = rel
		if fi.IsDir() && !strings.HasSuffix(th.Name, "/") {
			th.Name += "/"
		}
		if th.Name == "/" {
			return nil
		}
		if err := tw.WriteHeader(th); err != nil {
			return err
		}
		if fi.Mode().IsRegular() {
			f, err := os.Open(path)
			if err != nil {
				return err
			}
			defer f.Close()
			if _, err := io.Copy(tw, f); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		log.Printf("Walk error: %v", err)
		// Decent way to signal failure to the caller, since it'll break
		// the chunked response, rather than have a valid EOF.
		conn, _, _ := rw.(http.Hijacker).Hijack()
		conn.Close()
	}
	tw.Close()
	zw.Close()
}

func handleWriteTGZ(w http.ResponseWriter, r *http.Request) {
	var tgz io.Reader
	switch r.Method {
	case "PUT":
		tgz = r.Body
	case "POST":
		urlStr := r.FormValue("url")
		if urlStr == "" {
			http.Error(w, "missing url POST param", http.StatusBadRequest)
			return
		}
		res, err := http.Get(urlStr)
		if err != nil {
			http.Error(w, fmt.Sprintf("fetching URL %s: %v", urlStr, err), http.StatusInternalServerError)
			return
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			http.Error(w, fmt.Sprintf("fetching provided url: %s", res.Status), http.StatusInternalServerError)
			return
		}
		tgz = res.Body
	default:
		http.Error(w, "requires PUT or POST method", http.StatusBadRequest)
		return
	}

	urlParam, _ := url.ParseQuery(r.URL.RawQuery)
	baseDir := *scratchDir
	if dir := urlParam.Get("dir"); dir != "" {
		if !validRelativeDir(dir) {
			http.Error(w, "bogus dir", http.StatusBadRequest)
			return
		}
		dir = filepath.FromSlash(dir)
		baseDir = filepath.Join(baseDir, dir)
		if err := os.MkdirAll(baseDir, 0755); err != nil {
			http.Error(w, "mkdir of base: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}

	err := untar(tgz, baseDir)
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
	cn := w.(http.CloseNotifier)
	clientGone := cn.CloseNotify()
	handlerDone := make(chan bool)
	defer close(handlerDone)

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
	absCmd := cmdPath
	sysMode := r.FormValue("mode") == "sys"
	if sysMode {
		if cmdPath == "" {
			http.Error(w, "requires 'cmd' parameter", http.StatusBadRequest)
			return
		}
	} else {
		if !validRelPath(cmdPath) {
			http.Error(w, "requires 'cmd' parameter", http.StatusBadRequest)
			return
		}
		absCmd = filepath.Join(*scratchDir, filepath.FromSlash(cmdPath))
	}

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	cmd := exec.Command(absCmd, r.PostForm["cmdArg"]...)
	if sysMode {
		cmd.Dir = *scratchDir
	} else {
		cmd.Dir = filepath.Dir(absCmd)
	}
	cmdOutput := mtuWriter{w}
	cmd.Stdout = cmdOutput
	cmd.Stderr = cmdOutput
	err := cmd.Start()
	if err == nil {
		go func() {
			select {
			case <-clientGone:
				cmd.Process.Kill()
			case <-handlerDone:
				return
			}
		}()
		err = cmd.Wait()
	}
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

func handleHalt(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "requires POST method", http.StatusBadRequest)
		return
	}
	log.Printf("Halting in 1 second.")
	// do the halt in 1 second, to give the HTTP response time to complete:
	time.AfterFunc(1*time.Second, haltMachine)
}

func haltMachine() {
	if !*haltEntireOS {
		log.Printf("Ending buildlet process due to halt.")
		os.Exit(0)
		return
	}
	log.Printf("Halting machine.")
	time.AfterFunc(5*time.Second, func() { os.Exit(0) })
	if osHalt != nil {
		// TODO: Windows: http://msdn.microsoft.com/en-us/library/windows/desktop/aa376868%28v=vs.85%29.aspx
		osHalt()
		os.Exit(0)
	}
	// Backup mechanism, if exec hangs for any reason:
	var err error
	switch runtime.GOOS {
	case "openbsd":
		// Quick, no fs flush, and power down:
		err = exec.Command("halt", "-q", "-n", "-p").Run()
	case "freebsd":
		// Power off (-p), via halt (-o), now.
		err = exec.Command("shutdown", "-p", "-o", "now").Run()
	case "linux":
		// Don't sync (-n), force without shutdown (-f), and power off (-p).
		err = exec.Command("/bin/halt", "-n", "-f", "-p").Run()
	case "plan9":
		err = exec.Command("fshalt").Run()
	default:
		err = errors.New("No system-specific halt command run; will just end buildlet process.")
	}
	log.Printf("Shutdown: %v", err)
	log.Printf("Ending buildlet process post-halt")
	os.Exit(0)
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
type requirePasswordHandler struct {
	h        http.Handler
	password string // empty means no password
}

func (h requirePasswordHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	_, gotPass, _ := r.BasicAuth()
	if h.password != "" && h.password != gotPass {
		http.Error(w, "invalid password", http.StatusForbidden)
		return
	}
	h.h.ServeHTTP(w, r)
}
