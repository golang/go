// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The linkcheck command finds missing links in the godoc website.
// It crawls a URL recursively and notes URLs and URL fragments
// that it's seen and prints a report of missing links at the end.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
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
	if u, frag, ok := strings.Cut(url, "#"); ok {
		url = u
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
	if *verbose {
		log.Print(msg)
	}
	problems = append(problems, msg)
}

func crawlLoop() {
	for url := range urlq {
		if err := doCrawl(url); err != nil {
			addProblem(url, err.Error())
		}
	}
}

func doCrawl(url string) error {
	defer wg.Done()

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	res, err := http.DefaultTransport.RoundTrip(req)
	if err != nil {
		return err
	}
	// Handle redirects.
	if res.StatusCode/100 == 3 {
		newURL, err := res.Location()
		if err != nil {
			return fmt.Errorf("resolving redirect: %v", err)
		}
		if !strings.HasPrefix(newURL.String(), *root) {
			// Skip off-site redirects.
			return nil
		}
		crawl(newURL.String(), url)
		return nil
	}
	if res.StatusCode != http.StatusOK {
		return errors.New(res.Status)
	}
	slurp, err := io.ReadAll(res.Body)
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
	return nil
}

func main() {
	flag.Parse()

	go crawlLoop()
	crawl(*root, "")

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
	if len(problems) > 0 {
		os.Exit(1)
	}
}
