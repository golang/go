// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"bytes"
	"fmt"
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
		if r, err := index.Lookup(query); err == nil {
			result = r
		} else if err != nil && !c.IndexFullText {
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

// SearchResultDoc optionally specifies a function returning an HTML body
// displaying search results matching godoc documentation.
func (p *Presentation) SearchResultDoc(result SearchResult) []byte {
	return applyTemplate(p.SearchDocHTML, "searchDocHTML", result)
}

// SearchResultCode optionally specifies a function returning an HTML body
// displaying search results matching source code.
func (p *Presentation) SearchResultCode(result SearchResult) []byte {
	return applyTemplate(p.SearchCodeHTML, "searchCodeHTML", result)
}

// SearchResultTxt optionally specifies a function returning an HTML body
// displaying search results of textual matches.
func (p *Presentation) SearchResultTxt(result SearchResult) []byte {
	return applyTemplate(p.SearchTxtHTML, "searchTxtHTML", result)
}

// HandleSearch obtains results for the requested search and returns a page
// to display them.
func (p *Presentation) HandleSearch(w http.ResponseWriter, r *http.Request) {
	query := strings.TrimSpace(r.FormValue("q"))
	result := p.Corpus.Lookup(query)

	var contents bytes.Buffer
	for _, f := range p.SearchResults {
		contents.Write(f(p, result))
	}

	var title string
	if haveResults := contents.Len() > 0; haveResults {
		title = fmt.Sprintf(`Results for query: %v`, query)
		if !p.Corpus.IndexEnabled {
			result.Alert = ""
		}
	} else {
		title = fmt.Sprintf(`No results found for query %q`, query)
	}

	body := bytes.NewBuffer(applyTemplate(p.SearchHTML, "searchHTML", result))
	body.Write(contents.Bytes())

	p.ServePage(w, Page{
		Title:    title,
		Tabtitle: query,
		Query:    query,
		Body:     body.Bytes(),
		GoogleCN: googleCN(r),
	})
}

func (p *Presentation) serveSearchDesc(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/opensearchdescription+xml")
	data := map[string]interface{}{
		"BaseURL": fmt.Sprintf("http://%s", r.Host),
	}
	applyTemplateToResponseWriter(w, p.SearchDescXML, &data)
}

// tocColCount returns the no. of columns
// to split the toc table to.
func tocColCount(result SearchResult) int {
	tocLen := tocLen(result)
	colCount := 0
	// Simple heuristic based on visual aesthetic in manual testing.
	switch {
	case tocLen <= 10:
		colCount = 1
	case tocLen <= 20:
		colCount = 2
	case tocLen <= 80:
		colCount = 3
	default:
		colCount = 4
	}
	return colCount
}

// tocLen calculates the no. of items in the toc table
// by going through various fields in the SearchResult
// that is rendered in the UI.
func tocLen(result SearchResult) int {
	tocLen := 0
	for _, val := range result.Idents {
		if len(val) != 0 {
			tocLen++
		}
	}
	// If no identifiers, then just one item for the header text "Package <result.Query>".
	// See searchcode.html for further details.
	if len(result.Idents) == 0 {
		tocLen++
	}
	if result.Hit != nil {
		if len(result.Hit.Decls) > 0 {
			tocLen += len(result.Hit.Decls)
			// We need one extra item for the header text "Package-level declarations".
			tocLen++
		}
		if len(result.Hit.Others) > 0 {
			tocLen += len(result.Hit.Others)
			// We need one extra item for the header text "Local declarations and uses".
			tocLen++
		}
	}
	// For "textual occurrences".
	tocLen++
	return tocLen
}
