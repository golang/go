// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Command tipgodoc is the beginning of the new tip.golang.org server,
// serving the latest HEAD straight from the Git oven.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

const (
	repoURL      = "https://go.googlesource.com/"
	metaURL      = "https://go.googlesource.com/?b=master&format=JSON"
	startTimeout = 5 * time.Minute
)

func main() {
	const k = "TIP_BUILDER"
	var b Builder
	switch os.Getenv(k) {
	case "godoc":
		b = godocBuilder{}
	case "talks":
		b = talksBuilder{}
	default:
		log.Fatalf("Unknown %v value: %q", k, os.Getenv(k))
	}

	p := &Proxy{builder: b}
	go p.run()
	http.Handle("/", p)
	http.HandleFunc("/_ah/health", p.serveHealthCheck)

	if err := http.ListenAndServe(":8080", nil); err != nil {
		p.stop()
		log.Fatal(err)
	}
}

// Proxy implements the tip.golang.org server: a reverse-proxy
// that builds and runs godoc instances showing the latest docs.
type Proxy struct {
	builder Builder

	mu       sync.Mutex // protects the followin'
	proxy    http.Handler
	cur      string    // signature of gorepo+toolsrepo
	cmd      *exec.Cmd // live godoc instance, or nil for none
	side     string
	hostport string // host and port of the live instance
	err      error
}

type Builder interface {
	Signature(heads map[string]string) string
	Init(dir, hostport string, heads map[string]string) (*exec.Cmd, error)
	HealthCheck(hostport string) error
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/_tipstatus" {
		p.serveStatus(w, r)
		return
	}
	p.mu.Lock()
	proxy := p.proxy
	err := p.err
	p.mu.Unlock()
	if proxy == nil {
		s := "starting up"
		if err != nil {
			s = err.Error()
		}
		http.Error(w, s, http.StatusInternalServerError)
		return
	}
	if r.URL.Path == "/_ah/health" {
		if err := p.builder.HealthCheck(p.hostport); err != nil {
			http.Error(w, "Health check failde: "+err.Error(), http.StatusInternalServerError)
			return
		}
		fmt.Fprintln(w, "OK")
		return
	}
	proxy.ServeHTTP(w, r)
}

func (p *Proxy) serveStatus(w http.ResponseWriter, r *http.Request) {
	p.mu.Lock()
	defer p.mu.Unlock()
	fmt.Fprintf(w, "side=%v\ncurrent=%v\nerror=%v\n", p.side, p.cur, p.err)
}

func (p *Proxy) serveHealthCheck(w http.ResponseWriter, r *http.Request) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.proxy == nil {
		http.Error(w, "not ready", 500)
		return
	}
	io.WriteString(w, "ok")
}

// run runs in its own goroutine.
func (p *Proxy) run() {
	p.side = "a"
	for {
		p.poll()
		time.Sleep(30 * time.Second)
	}
}

func (p *Proxy) stop() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cmd != nil {
		p.cmd.Process.Kill()
	}
}

// poll runs from the run loop goroutine.
func (p *Proxy) poll() {
	heads := gerritMetaMap()
	if heads == nil {
		return
	}

	sig := p.builder.Signature(heads)

	p.mu.Lock()
	changes := sig != p.cur
	curSide := p.side
	p.cur = sig
	p.mu.Unlock()

	if !changes {
		return
	}

	newSide := "b"
	if curSide == "b" {
		newSide = "a"
	}

	dir := filepath.Join(os.TempDir(), "tip", newSide)
	if err := os.MkdirAll(dir, 0755); err != nil {
		p.err = err
		return
	}
	hostport := "localhost:8081"
	if newSide == "b" {
		hostport = "localhost:8082"
	}
	cmd, err := p.builder.Init(dir, hostport, heads)
	if err == nil {
		go func() {
			// TODO(adg,bradfitz): be smarter about dead processes
			if err := cmd.Wait(); err != nil {
				log.Printf("process in %v exited: %v", dir, err)
			}
		}()
		err = waitReady(p.builder, hostport)
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if err != nil {
		log.Println(err)
		p.err = err
		return
	}

	u, err := url.Parse(fmt.Sprintf("http://%v/", hostport))
	if err != nil {
		log.Println(err)
		p.err = err
		return
	}
	p.proxy = httputil.NewSingleHostReverseProxy(u)
	p.side = newSide
	p.hostport = hostport
	if p.cmd != nil {
		p.cmd.Process.Kill()
	}
	p.cmd = cmd
}

func waitReady(b Builder, hostport string) error {
	var err error
	deadline := time.Now().Add(startTimeout)
	for time.Now().Before(deadline) {
		if err = b.HealthCheck(hostport); err == nil {
			return nil
		}
		time.Sleep(time.Second)
	}
	return fmt.Errorf("timed out waiting for process at %v: %v", hostport, err)
}

func runErr(cmd *exec.Cmd) error {
	out, err := cmd.CombinedOutput()
	if err != nil {
		if len(out) == 0 {
			return err
		}
		return fmt.Errorf("%s\n%v", out, err)
	}
	return nil
}

func checkout(repo, hash, path string) error {
	// Clone git repo if it doesn't exist.
	if _, err := os.Stat(filepath.Join(path, ".git")); os.IsNotExist(err) {
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return err
		}
		if err := runErr(exec.Command("git", "clone", repo, path)); err != nil {
			return err
		}
	} else if err != nil {
		return err
	}

	// Pull down changes and update to hash.
	cmd := exec.Command("git", "fetch")
	cmd.Dir = path
	if err := runErr(cmd); err != nil {
		return err
	}
	cmd = exec.Command("git", "reset", "--hard", hash)
	cmd.Dir = path
	if err := runErr(cmd); err != nil {
		return err
	}
	cmd = exec.Command("git", "clean", "-d", "-f", "-x")
	cmd.Dir = path
	return runErr(cmd)
}

// gerritMetaMap returns the map from repo name (e.g. "go") to its
// latest master hash.
// The returned map is nil on any transient error.
func gerritMetaMap() map[string]string {
	res, err := http.Get(metaURL)
	if err != nil {
		return nil
	}
	defer res.Body.Close()
	defer io.Copy(ioutil.Discard, res.Body) // ensure EOF for keep-alive
	if res.StatusCode != 200 {
		return nil
	}
	var meta map[string]struct {
		Branches map[string]string
	}
	br := bufio.NewReader(res.Body)
	// For security reasons or something, this URL starts with ")]}'\n" before
	// the JSON object. So ignore that.
	// Shawn Pearce says it's guaranteed to always be just one line, ending in '\n'.
	for {
		b, err := br.ReadByte()
		if err != nil {
			return nil
		}
		if b == '\n' {
			break
		}
	}
	if err := json.NewDecoder(br).Decode(&meta); err != nil {
		log.Printf("JSON decoding error from %v: %s", metaURL, err)
		return nil
	}
	m := map[string]string{}
	for repo, v := range meta {
		if master, ok := v.Branches["master"]; ok {
			m[repo] = master
		}
	}
	return m
}

func getOK(url string) (body []byte, err error) {
	res, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	body, err = ioutil.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		return nil, err
	}
	if res.StatusCode != http.StatusOK {
		return nil, errors.New(res.Status)
	}
	return body, nil
}
