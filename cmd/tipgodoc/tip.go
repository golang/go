// Copyright 2014 The Go AUTHORS. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Command tipgodoc is the beginning of the new tip.golang.org server,
// serving the latest HEAD straight from the Git oven.
package main

import (
	"bufio"
	"encoding/json"
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
	repoURL = "https://go.googlesource.com/"
	metaURL = "https://go.googlesource.com/?b=master&format=JSON"
)

func main() {
	p := new(Proxy)
	go p.run()
	http.Handle("/", p)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

type Proxy struct {
	mu    sync.Mutex // protects the followin'
	proxy http.Handler
	cur   string // signature of gorepo+toolsrepo
	side  string
	err   error
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
		s := "tip.golang.org is starting up"
		if err != nil {
			s = err.Error()
		}
		http.Error(w, s, http.StatusInternalServerError)
		return
	}
	proxy.ServeHTTP(w, r)
}

func (p *Proxy) serveStatus(w http.ResponseWriter, r *http.Request) {
	p.mu.Lock()
	defer p.mu.Unlock()
	fmt.Fprintf(w, "side=%v\ncurrent=%v\nerror=%v\n", p.side, p.cur, p.err)
}

// run runs in its own goroutine.
func (p *Proxy) run() {
	p.side = "a"
	for {
		p.poll()
		time.Sleep(30 * time.Second)
	}
}

// poll runs from the run loop goroutine.
func (p *Proxy) poll() {
	heads := gerritMetaMap()
	if heads == nil {
		return
	}

	sig := heads["go"] + "-" + heads["tools"]

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

	hostport, err := initSide(newSide, heads["go"], heads["tools"])

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
	p.side = newSide
	p.proxy = httputil.NewSingleHostReverseProxy(u)
}

func initSide(side, goHash, toolsHash string) (hostport string, err error) {
	dir := filepath.Join(os.TempDir(), "tipgodoc", side)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}

	goDir := filepath.Join(dir, "go")
	toolsDir := filepath.Join(dir, "gopath/src/golang.org/x/tools")
	if err := checkout(repoURL+"go", goHash, goDir); err != nil {
		return "", err
	}
	if err := checkout(repoURL+"tools", toolsHash, toolsDir); err != nil {
		return "", err
	}

	make := exec.Command(filepath.Join(goDir, "src/make.bash"))
	make.Dir = filepath.Join(goDir, "src")
	if err := runErr(make); err != nil {
		return "", err
	}
	goBin := filepath.Join(goDir, "bin/go")
	install := exec.Command(goBin, "install", "golang.org/x/tools/cmd/godoc")
	install.Env = []string{"GOROOT=" + goDir, "GOPATH=" + filepath.Join(dir, "gopath")}
	if err := runErr(install); err != nil {
		return "", err
	}

	godocBin := filepath.Join(goDir, "bin/godoc")
	hostport = "localhost:8081"
	if side == "b" {
		hostport = "localhost:8082"
	}
	godoc := exec.Command(godocBin, "-http="+hostport)
	godoc.Env = []string{"GOROOT=" + goDir}
	// TODO(adg): log this somewhere useful
	godoc.Stdout = os.Stdout
	godoc.Stderr = os.Stderr
	if err := godoc.Start(); err != nil {
		return "", err
	}
	go func() {
		// TODO(bradfitz): tell the proxy that this side is dead
		if err := godoc.Wait(); err != nil {
			log.Printf("side %v exited: %v", side, err)
		}
	}()

	for i := 0; i < 15; i++ {
		time.Sleep(time.Second)
		var res *http.Response
		res, err = http.Get(fmt.Sprintf("http://%v/", hostport))
		if err != nil {
			continue
		}
		res.Body.Close()
		if res.StatusCode == http.StatusOK {
			return hostport, nil
		}
	}
	return "", fmt.Errorf("timed out waiting for side %v at %v (%v)", side, hostport, err)
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
