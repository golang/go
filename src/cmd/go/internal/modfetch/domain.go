// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for custom domains.

package modfetch

import (
	"encoding/xml"
	"fmt"
	"io"
	"net/url"
	"os"
	"strings"

	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modfetch/gitrepo"
)

// metaImport represents the parsed <meta name="go-import"
// content="prefix vcs reporoot" /> tags from HTML files.
type metaImport struct {
	Prefix, VCS, RepoRoot string
}

func lookupCustomDomain(path string) (Repo, error) {
	dom := path
	if i := strings.Index(dom, "/"); i >= 0 {
		dom = dom[:i]
	}
	if !strings.Contains(dom, ".") {
		return nil, fmt.Errorf("unknown module %s: not a domain name", path)
	}
	var body io.ReadCloser
	err := webGetGoGet("https://"+path+"?go-get=1", &body)
	if body != nil {
		defer body.Close()
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "FindRepo: %v\n", err)
		return nil, err
	}
	// Note: accepting a non-200 OK here, so people can serve a
	// meta import in their http 404 page.
	imports, err := parseMetaGoImports(body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "findRepo: %v\n", err)
		return nil, err
	}
	if len(imports) == 0 {
		return nil, fmt.Errorf("unknown module %s: no go-import tags", path)
	}

	// First look for new module definition.
	for _, imp := range imports {
		if path == imp.Prefix || strings.HasPrefix(path, imp.Prefix+"/") {
			if imp.VCS == "mod" {
				u, err := url.Parse(imp.RepoRoot)
				if err != nil {
					return nil, fmt.Errorf("invalid module URL %q", imp.RepoRoot)
				} else if u.Scheme != "https" {
					// TODO: Allow -insecure flag as a build flag?
					return nil, fmt.Errorf("invalid module URL %q: must be HTTPS", imp.RepoRoot)
				}
				return newProxyRepo(imp.RepoRoot, imp.Prefix), nil
			}
		}
	}

	// Fall back to redirections to known version control systems.
	for _, imp := range imports {
		if path == imp.Prefix {
			if !strings.HasPrefix(imp.RepoRoot, "https://") {
				// TODO: Allow -insecure flag as a build flag?
				return nil, fmt.Errorf("invalid server URL %q: must be HTTPS", imp.RepoRoot)
			}
			if imp.VCS == "git" {
				code, err := gitrepo.Repo(imp.RepoRoot, imp.Prefix)
				if err != nil {
					return nil, err
				}
				return newCodeRepo(code, path)
			}
			return nil, fmt.Errorf("unknown VCS, Repo: %s, %s", imp.VCS, imp.RepoRoot)
		}
	}

	// Check for redirect to repo root.
	for _, imp := range imports {
		if strings.HasPrefix(path, imp.Prefix+"/") {
			return nil, &ModuleSubdirError{imp.Prefix}
		}
	}

	return nil, fmt.Errorf("unknown module %s: no matching go-import tags", path)
}

type ModuleSubdirError struct {
	ModulePath string
}

func (e *ModuleSubdirError) Error() string {
	return fmt.Sprintf("module root is %q", e.ModulePath)
}

type customPrefix struct {
	codehost.Repo
	root string
}

func (c *customPrefix) Root() string {
	return c.root
}

// parseMetaGoImports returns meta imports from the HTML in r.
// Parsing ends at the end of the <head> section or the beginning of the <body>.
func parseMetaGoImports(r io.Reader) (imports []metaImport, err error) {
	d := xml.NewDecoder(r)
	d.CharsetReader = charsetReader
	d.Strict = false
	var t xml.Token
	for {
		t, err = d.RawToken()
		if err != nil {
			if err == io.EOF || len(imports) > 0 {
				err = nil
			}
			return
		}
		if e, ok := t.(xml.StartElement); ok && strings.EqualFold(e.Name.Local, "body") {
			return
		}
		if e, ok := t.(xml.EndElement); ok && strings.EqualFold(e.Name.Local, "head") {
			return
		}
		e, ok := t.(xml.StartElement)
		if !ok || !strings.EqualFold(e.Name.Local, "meta") {
			continue
		}
		if attrValue(e.Attr, "name") != "go-import" {
			continue
		}
		if f := strings.Fields(attrValue(e.Attr, "content")); len(f) == 3 {
			imports = append(imports, metaImport{
				Prefix:   f[0],
				VCS:      f[1],
				RepoRoot: f[2],
			})
		}
	}
}

// attrValue returns the attribute value for the case-insensitive key
// `name', or the empty string if nothing is found.
func attrValue(attrs []xml.Attr, name string) string {
	for _, a := range attrs {
		if strings.EqualFold(a.Name.Local, name) {
			return a.Value
		}
	}
	return ""
}

// charsetReader returns a reader for the given charset. Currently
// it only supports UTF-8 and ASCII. Otherwise, it returns a meaningful
// error which is printed by go get, so the user can find why the package
// wasn't downloaded if the encoding is not supported. Note that, in
// order to reduce potential errors, ASCII is treated as UTF-8 (i.e. characters
// greater than 0x7f are not rejected).
func charsetReader(charset string, input io.Reader) (io.Reader, error) {
	switch strings.ToLower(charset) {
	case "ascii":
		return input, nil
	default:
		return nil, fmt.Errorf("can't decode XML document using charset %q", charset)
	}
}
