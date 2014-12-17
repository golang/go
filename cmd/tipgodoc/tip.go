// Copyright 2014 The Go AUTHORS. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Command tipgodoc is the beginning of the new tip.golang.org server,
// serving the latest HEAD straight from the Git oven.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
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

const metaURL = "https://go.googlesource.com/?b=master&format=JSON"

var (
	pollInterval = flag.Duration("poll", 10*time.Second, "Remote repo poll interval")
	listenAddr   = flag.String("listen", "localhost:8080", "HTTP listen address")
)

func main() {
	flag.Parse()
	p := new(Proxy)
	go p.run()
	http.Handle("/", p)
	log.Fatal(http.ListenAndServe(*listenAddr, nil))
}

type Proxy struct {
	mu    sync.Mutex // owns the followin'
	proxy *httputil.ReverseProxy
	last  string // signature of gorepo+toolsrepo
	side  string
}

// run runs in its own goroutine.
func (p *Proxy) run() {
	p.side = "a"
	for {
		p.poll()
		time.Sleep(*pollInterval)
	}
}

// poll runs from the run loop goroutine.
func (p *Proxy) poll() {
	heads := gerritMetaMap()
	if heads == nil {
		return
	}

	p.mu.Lock()
	curSide := p.side
	lastSig := p.last
	p.mu.Unlock()

	sig := heads["go"] + "-" + heads["tools"]
	if sig == lastSig {
		return
	}
	newSide := "b"
	if curSide == "b" {
		newSide = "a"
	}

	hostport, err := initSide(newSide, heads["go"], heads["tools"])
	if err != nil {
		log.Println(err)
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	u, err := url.Parse(fmt.Sprintf("http://%v/", hostport))
	if err != nil {
		log.Println(err)
		return
	}
	p.side = newSide
	p.proxy = httputil.NewSingleHostReverseProxy(u)
	p.last = sig
}

func initSide(side, goHash, toolsHash string) (hostport string, err error) {
	dir := filepath.Join(os.TempDir(), "tipgodoc", side)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}

	goDir := filepath.Join(dir, "go")
	toolsDir := filepath.Join(dir, "gopath/src/golang.org/x/tools")
	if err := checkout("https://go.googlesource.com/go", goHash, goDir); err != nil {
		return "", err
	}
	if err := checkout("https://go.googlesource.com/tools", toolsHash, toolsDir); err != nil {
		return "", err

	}

	env := []string{"GOROOT=" + goDir, "GOPATH=" + filepath.Join(dir, "gopath")}

	make := exec.Command("./make.bash")
	make.Stdout = os.Stdout
	make.Stderr = os.Stderr
	make.Dir = filepath.Join(goDir, "src")
	if err := make.Run(); err != nil {
		return "", err
	}
	goBin := filepath.Join(goDir, "bin/go")
	install := exec.Command(goBin, "install", "golang.org/x/tools/cmd/godoc")
	install.Stdout = os.Stdout
	install.Stderr = os.Stderr
	install.Env = env
	if err := install.Run(); err != nil {
		return "", err
	}

	godocBin := filepath.Join(goDir, "bin/godoc")
	hostport = "localhost:8081"
	if side == "b" {
		hostport = "localhost:8082"
	}
	godoc := exec.Command(godocBin, "-http="+hostport)
	godoc.Env = env
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

func checkout(repo, hash, path string) error {
	// Clone git repo if it doesn't exist.
	if _, err := os.Stat(filepath.Join(path, ".git")); os.IsNotExist(err) {
		if err := os.MkdirAll(filepath.Base(path), 0755); err != nil {
			return err
		}
		cmd := exec.Command("git", "clone", repo, path)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			// TODO(bradfitz): capture the standard error output
			return err
		}
	} else if err != nil {
		return err
	}

	cmd := exec.Command("git", "fetch")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Dir = path
	if err := cmd.Run(); err != nil {
		return err
	}
	cmd = exec.Command("git", "reset", "--hard", hash)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Dir = path
	if err := cmd.Run(); err != nil {
		return err
	}
	cmd = exec.Command("git", "clean", "-d", "-f", "-x")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Dir = path
	return cmd.Run()
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/_tipstatus" {
		p.serveStatus(w, r)
		return
	}
	p.mu.Lock()
	proxy := p.proxy
	p.mu.Unlock()
	if proxy == nil {
		http.Error(w, "not ready", http.StatusInternalServerError)
		return
	}
	proxy.ServeHTTP(w, r)
}

func (p *Proxy) serveStatus(w http.ResponseWriter, r *http.Request) {
	p.mu.Lock()
	defer p.mu.Unlock()
	fmt.Fprintf(w, "side=%v\nlast=%v\n", p.side, p.last)
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
