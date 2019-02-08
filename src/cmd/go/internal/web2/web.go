// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package web2

import (
	"bytes"
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
)

var TraceGET = false
var webstack = false

func init() {
	flag.BoolVar(&TraceGET, "webtrace", TraceGET, "trace GET requests")
	flag.BoolVar(&webstack, "webstack", webstack, "print stack for GET requests")
}

type netrcLine struct {
	machine  string
	login    string
	password string
}

var netrcOnce sync.Once
var netrc []netrcLine

func parseNetrc(data string) []netrcLine {
	var nrc []netrcLine
	var l netrcLine
	for _, line := range strings.Split(data, "\n") {
		f := strings.Fields(line)
		for i := 0; i < len(f)-1; i += 2 {
			switch f[i] {
			case "machine":
				l.machine = f[i+1]
			case "login":
				l.login = f[i+1]
			case "password":
				l.password = f[i+1]
			}
		}
		if l.machine != "" && l.login != "" && l.password != "" {
			nrc = append(nrc, l)
			l = netrcLine{}
		}
	}
	return nrc
}

func havePassword(machine string) bool {
	netrcOnce.Do(readNetrc)
	for _, line := range netrc {
		if line.machine == machine {
			return true
		}
	}
	return false
}

func netrcPath() string {
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(os.Getenv("USERPROFILE"), "_netrc")
	case "plan9":
		return filepath.Join(os.Getenv("home"), ".netrc")
	default:
		return filepath.Join(os.Getenv("HOME"), ".netrc")
	}
}

func readNetrc() {
	data, err := ioutil.ReadFile(netrcPath())
	if err != nil {
		return
	}
	netrc = parseNetrc(string(data))
}

type getState struct {
	req      *http.Request
	resp     *http.Response
	body     io.ReadCloser
	non200ok bool
}

type Option interface {
	option(*getState) error
}

func Non200OK() Option {
	return optionFunc(func(g *getState) error {
		g.non200ok = true
		return nil
	})
}

type optionFunc func(*getState) error

func (f optionFunc) option(g *getState) error {
	return f(g)
}

func DecodeJSON(dst interface{}) Option {
	return optionFunc(func(g *getState) error {
		if g.resp != nil {
			return json.NewDecoder(g.body).Decode(dst)
		}
		return nil
	})
}

func ReadAllBody(body *[]byte) Option {
	return optionFunc(func(g *getState) error {
		if g.resp != nil {
			var err error
			*body, err = ioutil.ReadAll(g.body)
			return err
		}
		return nil
	})
}

func Body(body *io.ReadCloser) Option {
	return optionFunc(func(g *getState) error {
		if g.resp != nil {
			*body = g.body
			g.body = nil
		}
		return nil
	})
}

func Header(hdr *http.Header) Option {
	return optionFunc(func(g *getState) error {
		if g.resp != nil {
			*hdr = CopyHeader(g.resp.Header)
		}
		return nil
	})
}

func CopyHeader(hdr http.Header) http.Header {
	if hdr == nil {
		return nil
	}
	h2 := make(http.Header)
	for k, v := range hdr {
		v2 := make([]string, len(v))
		copy(v2, v)
		h2[k] = v2
	}
	return h2
}

var cache struct {
	mu    sync.Mutex
	byURL map[string]*cacheEntry
}

type cacheEntry struct {
	mu   sync.Mutex
	resp *http.Response
	body []byte
}

var httpDo = http.DefaultClient.Do

func SetHTTPDoForTesting(do func(*http.Request) (*http.Response, error)) {
	if do == nil {
		do = http.DefaultClient.Do
	}
	httpDo = do
}

func Get(url string, options ...Option) error {
	if TraceGET || webstack || cfg.BuildV {
		log.Printf("Fetching %s", url)
		if webstack {
			log.Println(string(debug.Stack()))
		}
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	netrcOnce.Do(readNetrc)
	for _, l := range netrc {
		if l.machine == req.URL.Host {
			req.SetBasicAuth(l.login, l.password)
			break
		}
	}

	g := &getState{req: req}
	for _, o := range options {
		if err := o.option(g); err != nil {
			return err
		}
	}

	cache.mu.Lock()
	e := cache.byURL[url]
	if e == nil {
		e = new(cacheEntry)
		if !strings.HasPrefix(url, "file:") {
			if cache.byURL == nil {
				cache.byURL = make(map[string]*cacheEntry)
			}
			cache.byURL[url] = e
		}
	}
	cache.mu.Unlock()

	e.mu.Lock()
	if strings.HasPrefix(url, "file:") {
		body, err := ioutil.ReadFile(req.URL.Path)
		if err != nil {
			e.mu.Unlock()
			return err
		}
		e.body = body
		e.resp = &http.Response{
			StatusCode: 200,
		}
	} else if e.resp == nil {
		resp, err := httpDo(req)
		if err != nil {
			e.mu.Unlock()
			return err
		}
		e.resp = resp
		// TODO: Spool to temp file.
		body, err := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		resp.Body = nil
		if err != nil {
			e.mu.Unlock()
			return err
		}
		e.body = body
	}
	g.resp = e.resp
	g.body = ioutil.NopCloser(bytes.NewReader(e.body))
	e.mu.Unlock()

	defer func() {
		if g.body != nil {
			g.body.Close()
		}
	}()

	if g.resp.StatusCode == 403 && req.URL.Host == "api.github.com" && !havePassword("api.github.com") {
		base.Errorf("%s", githubMessage)
	}
	if !g.non200ok && g.resp.StatusCode != 200 {
		return fmt.Errorf("unexpected status (%s): %v", url, g.resp.Status)
	}

	for _, o := range options {
		if err := o.option(g); err != nil {
			return err
		}
	}
	return err
}

var githubMessage = `go: 403 response from api.github.com

GitHub applies fairly small rate limits to unauthenticated users, and
you appear to be hitting them. To authenticate, please visit
https://github.com/settings/tokens and click "Generate New Token" to
create a Personal Access Token. The token only needs "public_repo"
scope, but you can add "repo" if you want to access private
repositories too.

Add the token to your $HOME/.netrc (%USERPROFILE%\_netrc on Windows):

    machine api.github.com login YOU password TOKEN

Sorry for the interruption.
`
