// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strings"
)

type SearchResult struct {
	Query string
	Alert string // error or warning message

	// identifier matches
	Pak HitList       // packages matching Query
	Hit *LookupResult // identifier matches of Query
	Alt *AltWords     // alternative identifiers to look for

	// textual matches
	Found    int         // number of textual occurrences found
	Textual  []FileLines // textual matches of Query
	Complete bool        // true if all textual occurrences of Query are reported
	Idents   map[SpotKind][]Ident
}

func (c *Corpus) Lookup(query string) SearchResult {
	result := &SearchResult{Query: query}

	index, timestamp := c.CurrentIndex()
	if index != nil {
		// identifier search
		var err error
		result, err = index.Lookup(query)
		if err != nil && !c.IndexFullText {
			// ignore the error if full text search is enabled
			// since the query may be a valid regular expression
			result.Alert = "Error in query string: " + err.Error()
			return *result
		}

		// full text search
		if c.IndexFullText && query != "" {
			rx, err := regexp.Compile(query)
			if err != nil {
				result.Alert = "Error in query regular expression: " + err.Error()
				return *result
			}
			// If we get maxResults+1 results we know that there are more than
			// maxResults results and thus the result may be incomplete (to be
			// precise, we should remove one result from the result set, but
			// nobody is going to count the results on the result page).
			result.Found, result.Textual = index.LookupRegexp(rx, c.MaxResults+1)
			result.Complete = result.Found <= c.MaxResults
			if !result.Complete {
				result.Found-- // since we looked for maxResults+1
			}
		}
	}

	// is the result accurate?
	if c.IndexEnabled {
		if ts := c.FSModifiedTime(); timestamp.Before(ts) {
			// The index is older than the latest file system change under godoc's observation.
			result.Alert = "Indexing in progress: result may be inaccurate"
		}
	} else {
		result.Alert = "Search index disabled: no results available"
	}

	return *result
}

func (p *Presentation) HandleSearch(w http.ResponseWriter, r *http.Request) {
	query := strings.TrimSpace(r.FormValue("q"))
	result := p.Corpus.Lookup(query)

	if p.GetPageInfoMode(r)&NoHTML != 0 {
		p.ServeText(w, applyTemplate(p.SearchText, "searchText", result))
		return
	}

	haveResults := result.Hit != nil || len(result.Textual) > 0
	if !haveResults {
		for _, ir := range result.Idents {
			if ir != nil {
				haveResults = true
				break
			}
		}
	}
	var title string
	if haveResults {
		title = fmt.Sprintf(`Results for query %q`, query)
	} else {
		title = fmt.Sprintf(`No results found for query %q`, query)
	}

	p.ServePage(w, Page{
		Title:    title,
		Tabtitle: query,
		Query:    query,
		Body:     applyTemplate(p.SearchHTML, "searchHTML", result),
	})
}

func (p *Presentation) serveSearchDesc(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/opensearchdescription+xml")
	data := map[string]interface{}{
		"BaseURL": fmt.Sprintf("http://%s", r.Host),
	}
	if err := p.SearchDescXML.Execute(w, &data); err != nil && err != http.ErrBodyNotAllowed {
		// Only log if there's an error that's not about writing on HEAD requests.
		// See Issues 5451 and 5454.
		log.Printf("searchDescXML.Execute: %s", err)
	}
}
