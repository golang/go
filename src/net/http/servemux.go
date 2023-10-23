// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"net"
	"net/url"
	"path"
	"strings"
	"sync"
)

var (
	// MuxAllMethod defines the methods that Mux allows to register.
	MuxAllMethod = []string{
		MethodGet, MethodPost, MethodPut,
		MethodDelete, MethodHead, MethodPatch,
		MethodOptions, MethodConnect, MethodTrace,
	}
	// MuxAnyMethod defines the Any matching method of Mux.
	MuxAnyMethod  = append([]string{}, MuxAllMethod[0:9]...)
	MuxValueRoute = "route"
	MuxValueAllow = "allow"

	// DefaultServeMux is the default ServeMux used by Serve.
	DefaultServeMux = defaultServeMux

	defaultServeMux = NewServeMux()
	methodAny       = "ANY"
)

// ServeMux is an HTTP request multiplexer.
// It matches the URL of each incoming request against a list of registered
// patterns and calls the handler for the pattern that
// most closely matches the URL.
//
// Patterns name fixed, rooted paths, like "/favicon.ico",
// or rooted subtrees, like "/images/" (note the trailing slash).
// Longer patterns take precedence over shorter ones, so that
// if there are handlers registered for both "/images/"
// and "/images/thumbnails/", the latter handler will be
// called for paths beginning "/images/thumbnails/" and the
// former will receive requests for any other paths in the
// "/images/" subtree.
//
// Pattern name uses "path" or "method path" format.
// The method value is allowed to be the global variable MuxAllMethod
// or the constant "ANY/NOTFOUND/404/METHODNOTALLOWED/405",
// method allows use multiple values, using the "," character join;
// when using the "path" format, the default method is "ANY".
// When the method is "404" or "405", set the corresponding Handler of ServeMux.
// When the method is "ANY", it is allowed to match the method in the global
// variable MuxAnyMethod.
// Use the ":" variable pattern in path to match the string between two "/",
// and use the "*" wildcard pattern to match all remaining paths;
// you can set names for variables, such as: ":name", "*path" ;
// constant key/value pairs are saved in Request.PathValues and
// can be read using the Request.GetPathValue method.
//
// Note that since a pattern ending in a slash names a rooted subtree,
// the pattern "/" matches all paths not matched by other registered
// patterns, not just the URL with Path == "/".
//
// If a subtree has been registered and a request is received naming the
// subtree root without its trailing slash, ServeMux redirects that
// request to the subtree root (adding the trailing slash). This behavior can
// be overridden with a separate registration for the path without
// the trailing slash. For example, registering "/images/" causes ServeMux
// to redirect a request for "/images" to "/images/", unless "/images" has
// been registered separately.
//
// Patterns may optionally begin with a host name, restricting matches to
// URLs on that host only. Host-specific patterns take precedence over
// general patterns, so that a handler might register for the two patterns
// "/codesearch" and "codesearch.google.com/" without also taking over
// requests for "http://www.google.com/".
//
// ServeMux also takes care of sanitizing the URL request path and the Host
// header, stripping the port number and redirecting any request containing . or
// .. elements or repeated slashes to an equivalent, cleaner URL.
type ServeMux struct {
	mu         sync.RWMutex
	root       *muxNode // radix tree
	handler404 Handler
	handler405 Handler
	hosts      bool        // whether any patterns contain hostnames
	mux121     serveMux121 // used only when GODEBUG=httpmuxgo121=1
}

type muxNode struct {
	path  string // constant path
	name  string // variable name
	route string // routing pattern

	// children
	Wchildren *muxNode
	Cchildren []*muxNode
	Pchildren []*muxNode

	// handlers
	anyHandler Handler
	methods    []string
	handlers   []Handler
}

// Handle registers the handler for the given pattern in [DefaultServeMux].
// The documentation for [ServeMux] explains how patterns are matched.
func Handle(pattern string, handler Handler) {
	DefaultServeMux.Handle(pattern, handler)
}

// HandleFunc registers the handler function for the given pattern in [DefaultServeMux].
// The documentation for [ServeMux] explains how patterns are matched.
func HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
	DefaultServeMux.Handle(pattern, HandlerFunc(handler))
}

func NewServeMux() *ServeMux {
	return &ServeMux{
		root:       &muxNode{},
		handler404: HandlerFunc(NotFound),
		handler405: HandlerFunc(MethodNotAllowed),
	}
}

// HandleFunc registers the handler function for the given pattern.
func (mux *ServeMux) HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
	if use121 {
		mux.mux121.handleFunc(pattern, handler)
		return
	}
	if handler == nil {
		panic("http: nil handler")
	}
	mux.Handle(pattern, HandlerFunc(handler))
}

// Handle registers the handler for the given pattern.
// If a handler already exists for pattern, Handle panics.
func (mux *ServeMux) Handle(pattern string, handler Handler) {
	if use121 {
		mux.mux121.handle(pattern, handler)
		return
	}
	mux.mu.Lock()
	defer mux.mu.Unlock()
	if pattern == "" {
		panic("http: invalid pattern")
	}
	if handler == nil {
		panic("http: nil handler")
	}

	methods, path := muxSplitMethods(pattern)
	var host string
	pos := strings.IndexByte(path, '/')
	if pos > 0 {
		mux.hosts = true
		host = path[:pos]
		path = path[pos:]
	} else if path == "" {
		panic("http: invalid pattern")
	}

	for _, method := range methods {
		switch method {
		case methodAny:
			mux.insertRoute(method, host, path, handler)
		case "NOTFOUND", "404":
			mux.handler404 = handler
		case "METHODNOTALLOWED", "405":
			mux.handler405 = handler
		default:
			for _, m := range MuxAllMethod {
				if method == m {
					mux.insertRoute(method, host, path, handler)
					return
				}
			}
			panic("http: invalid method " + method)
		}
	}
}

// ServeHTTP dispatches the request to the handler whose
// pattern most closely matches the request URL.
func (mux *ServeMux) ServeHTTP(w ResponseWriter, r *Request) {
	if r.RequestURI == "*" {
		if r.ProtoAtLeast(1, 1) {
			w.Header().Set("Connection", "close")
		}
		w.WriteHeader(StatusBadRequest)
		return
	}

	var h Handler
	if use121 {
		h, _ = mux.mux121.findHandler(r)
	} else {
		h, _ = mux.Handler(r)
	}
	h.ServeHTTP(w, r)
}

func (mux *ServeMux) Handler(r *Request) (Handler, string) {
	if use121 {
		return mux.mux121.findHandler(r)
	}
	mux.mu.RLock()
	defer mux.mu.RUnlock()

	if len(r.PathValues) > 1 {
		path := "/"
		if r.PathValues[len(r.PathValues)-2] != MuxValueRoute {
			path += r.PathValues[len(r.PathValues)-1]
		}
		h, shouldRedirect := mux.match(r.Method, path, r)
		if h == nil || shouldRedirect {
			h = mux.handler404
		}
		return h, r.GetPathValue(MuxValueRoute)
	}
	return mux.handler(r)
}

func (mux *ServeMux) handler(r *Request) (Handler, string) {
	// CONNECT requests are not canonicalized.
	if r.Method == MethodConnect {
		// If r.URL.Path is /tree and its handler is not registered,
		// the /tree -> /tree/ redirect applies to CONNECT requests
		// but the path canonicalization does not.
		if _, ok := mux.handlerHost(r.URL.Host, r.URL.Path, r); ok {
			u := &url.URL{Path: r.URL.Path + "/", RawQuery: r.URL.RawQuery}
			return RedirectHandler(u.String(), StatusMovedPermanently), u.Path
		}

		r.PathValues = r.PathValues[0:0]
		handler, _ := mux.handlerHost(r.Host, r.URL.Path, r)
		return handler, r.GetPathValue(MuxValueRoute)
	}

	// All other requests have any port stripped and path cleaned
	// before passing to mux.handler.
	host := stripHostPort(r.Host)
	path := cleanPath(r.URL.Path)

	// If the given path is /tree and its handler is not registered,
	// redirect for /tree/.
	handler, shouldRedirect := mux.handlerHost(host, path, r)
	if shouldRedirect {
		u := &url.URL{Path: path + "/", RawQuery: r.URL.RawQuery}
		return RedirectHandler(u.String(), StatusMovedPermanently), u.Path
	}

	if path != r.URL.Path {
		u := &url.URL{Path: path, RawQuery: r.URL.RawQuery}
		return RedirectHandler(u.String(), StatusMovedPermanently), r.GetPathValue(MuxValueRoute)
	}

	return handler, r.GetPathValue(MuxValueRoute)
}

func (mux *ServeMux) handlerHost(host, path string, r *Request) (Handler, bool) {
	if mux.hosts {
		h1, ok1 := mux.match(r.Method, host+path, r)
		if h1 != nil {
			return h1, false
		}

		r.PathValues = nil
		h2, ok2 := mux.match(r.Method, path, r)
		if h2 != nil {
			return h2, false
		}
		// Neither the path nor the Host match and there is a redirection.
		if ok1 || ok2 {
			return nil, true
		}
		return mux.handler404, false
	}
	h, shouldRedirect := mux.match(r.Method, path, r)
	if h == nil {
		h = mux.handler404
	}
	return h, shouldRedirect
}

// cleanPath returns the canonical path for p, eliminating . and .. elements.
func cleanPath(p string) string {
	if p == "" {
		return "/"
	}
	if p[0] != '/' {
		p = "/" + p
	}
	np := path.Clean(p)
	// path.Clean removes trailing slash except for root;
	// put the trailing slash back if necessary.
	if p[len(p)-1] == '/' && np != "/" {
		// Fast path for common case of p being the string we want:
		if len(p) == len(np)+1 && strings.HasPrefix(p, np) {
			np = p
		} else {
			np += "/"
		}
	}
	return np
}

// stripHostPort returns h without any trailing ":<port>".
func stripHostPort(h string) string {
	// If no port on host, return unchanged
	if !strings.Contains(h, ":") {
		return h
	}
	host, _, err := net.SplitHostPort(h)
	if err != nil {
		return h // on error, return unchanged
	}
	return host
}

// The match method matches a path and returns Handler or
// whether redirection is allowed,
// and uses the Request.PathValues property to save the path parameters.
func (mux *ServeMux) match(method, path string, r *Request) (Handler, bool) {
	r.PathValues = append(r.PathValues, MuxValueRoute, "")
	pos := len(r.PathValues) - 1
	node, shouldRedirect := mux.root.lookNode(path, &r.PathValues)
	if node == nil {
		return nil, false
	} else if shouldRedirect {
		r.PathValues[pos] = node.route
		return nil, true
	}
	r.PathValues[pos] = node.route

	// default method
	for i, m := range node.methods {
		if m == method {
			return node.handlers[i], false
		}
	}
	// any method
	if node.anyHandler != nil {
		for _, m := range MuxAnyMethod {
			if method == m {
				return node.anyHandler, false
			}
		}
		r.PathValues = append(r.PathValues, MuxValueAllow, strings.Join(MuxAnyMethod, ", "))
	} else {
		r.PathValues = append(r.PathValues, MuxValueAllow, strings.Join(node.methods, ", "))
	}
	return mux.handler405, false
}

// insertRoute method Add a new route Node.
func (mux *ServeMux) insertRoute(method, host, path string, handler Handler) {
	node := mux.root
	routes := muxSplitRoutes(path)
	path = strings.Join(routes, "")
	routes[0] = host + routes[0]
	for _, route := range routes {
		node = node.insertNode(newMuxNode(route))
	}

	node.route = host + path
	if !node.setHandler(method, handler) {
		pattern := node.route
		if method != methodAny {
			pattern = method + " " + pattern
		}
		panic("http: multiple registrations for " + pattern)
	}

	// Compatible: The path '/' matches '/*', and a new Node is added.
	if strings.HasSuffix(path, "/") {
		node = node.insertNode(newMuxNode("*"))
		node.route = host + path
		node.setHandler(method, handler)
	}
}

// The newMuxNode function creates a Radix node and sets different node names.
//
// The '*' prefix is a wildcard node, the ':' prefix is a parameter node.
func newMuxNode(path string) *muxNode {
	newNode := &muxNode{path: path}
	switch path[0] {
	case '*', ':':
		if len(path) == 1 {
			newNode.name = path
		} else {
			newNode.name = path[1:]
		}
	}
	return newNode
}

// The setHandler method sets the Handler according to whether the method is
// Any, Core, or Other. If the Handler exists, it returns failure.
func (r *muxNode) setHandler(method string, handler Handler) bool {
	// any method
	if method == methodAny {
		if r.anyHandler != nil {
			return false
		}
		r.anyHandler = handler
		return true
	}

	for _, m := range r.methods {
		if m == method {
			return false
		}
	}
	r.methods = append(r.methods, method)
	r.handlers = append(r.handlers, handler)
	return true
}

// insertNode add a child node to the node.
func (r *muxNode) insertNode(nextNode *muxNode) *muxNode {
	if len(nextNode.path) == 0 {
		return r
	}
	switch {
	case nextNode.name == "":
		return r.insertNodeConst(nextNode.path, nextNode)
	case nextNode.path[0] == ':':
		for _, i := range r.Pchildren {
			if i.path == nextNode.path {
				return i
			}
		}
		r.Pchildren = append(r.Pchildren, nextNode)
	case nextNode.path[0] == '*':
		if r.Wchildren == nil {
			r.Wchildren = nextNode
		} else {
			r.Wchildren.path = nextNode.path
			r.Wchildren.name = nextNode.name
		}
		return r.Wchildren
	}
	return nextNode
}

// insertNodeConst method handles adding constant node.
func (r *muxNode) insertNodeConst(path string, nextNode *muxNode) *muxNode {
	for i := range r.Cchildren {
		subStr, find := getSubsetPrefix(path, r.Cchildren[i].path)
		if find {
			// If the constant node path is longer than the public prefix,
			// the node path needs to be split.
			// The public path serves as a parent node and a child node with
			// the remaining paths, so that the parent node path must be the
			// prefix of the new node.
			if subStr != r.Cchildren[i].path {
				r.Cchildren[i].path = strings.TrimPrefix(r.Cchildren[i].path, subStr)
				r.Cchildren[i] = &muxNode{
					path:      subStr,
					Cchildren: []*muxNode{r.Cchildren[i]},
				}
			}
			nextNode.path = strings.TrimPrefix(path, subStr)
			return r.Cchildren[i].insertNode(nextNode)
		}
	}
	r.Cchildren = append(r.Cchildren, nextNode)
	// Sort alphabetically.
	for i := len(r.Cchildren) - 1; i > 0; i-- {
		if r.Cchildren[i].path[0] < r.Cchildren[i-1].path[0] {
			r.Cchildren[i], r.Cchildren[i-1] = r.Cchildren[i-1], r.Cchildren[i]
		}
	}
	return nextNode
}

func (r *muxNode) lookNode(key string, values *[]string) (*muxNode, bool) {
	// constant match, return data
	if len(key) == 0 && r.route != "" {
		return r, false
	}

	if len(key) > 0 {
		// Traverse constant Node match
		for _, child := range r.Cchildren {
			if child.path[0] >= key[0] {
				length := len(child.path)
				if len(key) >= length && key[:length] == child.path {
					if n, b := child.lookNode(key[length:], values); n != nil {
						return n, b
					}
				}

				// Compatible: Try redirecting to /tree/
				if child.path == key+"/" && child.route != "" {
					return child, true
				}
				break
			}
		}

		// parameter matching, Check if there is a parameter match
		if len(r.Pchildren) > 0 {
			pos := strings.IndexByte(key, '/')
			if pos == -1 {
				pos = len(key)
			}
			currentKey, nextSearchKey := key[:pos], key[pos:]
			for _, child := range r.Pchildren {
				if n, b := child.lookNode(nextSearchKey, values); n != nil {
					*values = append(*values, child.name, currentKey)
					return n, b
				}
			}
		}
	}

	// If the Node has a wildcard processing method that directly matches
	if r.Wchildren != nil {
		*values = append(*values, r.Wchildren.name, key)
		return r.Wchildren, false
	}

	// can't match, return nil
	return nil, false
}

// The muxSplitMethods function splits the string from the beginning of the
// string to the first ' ' or before the first '/' into methods.
//
//	"Get,mm, /index"  => [GET MM] /index
//	"Get,MM,m2 com/"  => [GET MM m2] com/
//	"Get,mm,/m2 com/" => [GET MM] /m2 com/
//	"/index"          => [ANY] /index
//	" com/"           => [ANY] com/
func muxSplitMethods(path string) ([]string, string) {
	switch strings.ToUpper(path) {
	case "NOTFOUND", "404", "METHODNOTALLOWED", "405":
		return []string{strings.ToUpper(path)}, "/"
	}
	if strings.IndexByte(path, ' ') == -1 {
		return []string{methodAny}, path
	}

	pos := 0
	methods := make([]string, 0, 2)
	for i, b := range path {
		if b == ',' || b == ' ' {
			method := strings.ToUpper(path[pos:i])
			pos = i + 1
			if method == "" {
				continue
			}

			methods = append(methods, method)
			if b == ' ' {
				break
			}
		} else if b == '/' {
			break
		}
	}

	if len(methods) == 0 {
		methods = append(methods, methodAny)
	}
	return methods, path[pos:]
}

// The muxSplitRoutes function splits strings by Node type.
//
//	/                 => [/]
//	/api/note/        => [/api/note/]
//	api/note/         => [api/note/]
//	/api/space 2/     => [/api/space 2/] len is 1
//	//api/*           => [/api/ *]
//	////api/*name     => [/api/ *name]
//	/api/get/         => [/api/get/]
//	/api/get          => [/api/get]
//	/api/:            => [/api/ :]
//	/api/:get         => [/api/ :get]
//	/api/:get/*       => [/api/ :get / *]
//	/api/:name/info/* => [/api/ :name /info/ *]
func muxSplitRoutes(key string) []string {
	if len(key) < 2 {
		return []string{"/"}
	}
	if key[0] != '/' {
		key = "/" + key
	}

	var strs []string
	length := -1
	isconst := false
	for i := range key {
		switch key[i] {
		case '/':
			// Convert to constant mode to create a new string
			if !isconst {
				length++
				strs = append(strs, "")
				isconst = true
			}
		case ':', '*':
			// Convert to variable mode
			isconst = false
			length++
			strs = append(strs, "")
		}
		strs[length] += key[i : i+1]
	}
	return strs
}

// Get the largest common prefix of the two strings,
// return the largest common prefix and have the largest common prefix.
func getSubsetPrefix(str1, str2 string) (string, bool) {
	findSubset := false
	for i := 0; i < len(str1) && i < len(str2); i++ {
		if str1[i] != str2[i] {
			retStr := str1[:i]
			return retStr, findSubset
		}
		findSubset = true
	}

	if len(str1) > len(str2) {
		return str2, findSubset
	} else if len(str1) == len(str2) {
		return str1, str1 == str2
	}

	return str1, findSubset
}
