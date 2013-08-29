// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The linkcheck command finds missing links in the godoc website.
// It crawls a URL recursively and notes URLs and URL fragments
// that it's seen and prints a report of missing links at the end.
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"regexp"
	"strings"
	"sync"
)

var (
	root    = flag.String("root", "http://localhost:6060", "Root to crawl")
	verbose = flag.Bool("verbose", false, "verbose")
)

var wg sync.WaitGroup        // outstanding fetches
var urlq = make(chan string) // URLs to crawl

// urlFrag is a URL and its optional #fragment (without the #)
type urlFrag struct {
	url, frag string
}

var (
	mu          sync.Mutex
	crawled     = make(map[string]bool)      // URL without fragment -> true
	neededFrags = make(map[urlFrag][]string) // URL#frag -> who needs it
)

var aRx = regexp.MustCompile(`<a href=['"]?(/[^\s'">]+)`)

// Owned by crawlLoop goroutine:
var (
	linkSources = make(map[string][]string) // url no fragment -> sources
	fragExists  = make(map[urlFrag]bool)
	problems    []string
)

func localLinks(body string) (links []string) {
	seen := map[string]bool{}
	mv := aRx.FindAllStringSubmatch(body, -1)
	for _, m := range mv {
		ref := m[1]
		if strings.HasPrefix(ref, "/src/") {
			continue
		}
		if !seen[ref] {
			seen[ref] = true
			links = append(links, m[1])
		}
	}
	return
}

var idRx = regexp.MustCompile(`\bid=['"]?([^\s'">]+)`)

func pageIDs(body string) (ids []string) {
	mv := idRx.FindAllStringSubmatch(body, -1)
	for _, m := range mv {
		ids = append(ids, m[1])
	}
	return
}

// url may contain a #fragment, and the fragment is then noted as needing to exist.
func crawl(url string, sourceURL string) {
	if strings.Contains(url, "/devel/release") {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	var frag string
	if i := strings.Index(url, "#"); i >= 0 {
		frag = url[i+1:]
		url = url[:i]
		if frag != "" {
			uf := urlFrag{url, frag}
			neededFrags[uf] = append(neededFrags[uf], sourceURL)
		}
	}
	if crawled[url] {
		return
	}
	crawled[url] = true

	wg.Add(1)
	go func() {
		urlq <- url
	}()
}

func addProblem(url, errmsg string) {
	msg := fmt.Sprintf("Error on %s: %s (from %s)", url, errmsg, linkSources[url])
	log.Print(msg)
	problems = append(problems, msg)
}

func crawlLoop() {
	for url := range urlq {
		res, err := http.Get(url)
		if err != nil {
			addProblem(url, fmt.Sprintf("Error fetching: %v", err))
			wg.Done()
			continue
		}
		if res.StatusCode != 200 {
			addProblem(url, fmt.Sprintf("Status code = %d", res.StatusCode))
			wg.Done()
			continue
		}
		slurp, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			log.Fatalf("Error reading %s body: %v", url, err)
		}
		if *verbose {
			log.Printf("Len of %s: %d", url, len(slurp))
		}
		body := string(slurp)
		for _, ref := range localLinks(body) {
			if *verbose {
				log.Printf("  links to %s", ref)
			}
			dest := *root + ref
			linkSources[dest] = append(linkSources[dest], url)
			crawl(dest, url)
		}
		for _, id := range pageIDs(body) {
			if *verbose {
				log.Printf(" url %s has #%s", url, id)
			}
			fragExists[urlFrag{url, id}] = true
		}

		wg.Done()
	}
}

func main() {
	flag.Parse()

	go crawlLoop()
	crawl(*root, "")
	crawl(*root+"/doc/go1.1.html", "")

	wg.Wait()
	close(urlq)
	for uf, needers := range neededFrags {
		if !fragExists[uf] {
			problems = append(problems, fmt.Sprintf("Missing fragment for %+v from %v", uf, needers))
		}
	}

	for _, s := range problems {
		fmt.Println(s)
	}
}
