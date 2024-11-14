// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements a decision tree for fast matching of requests to
// patterns.
//
// The root of the tree branches on the host of the request.
// The next level branches on the method.
// The remaining levels branch on consecutive segments of the path.
//
// The "more specific wins" precedence rule can result in backtracking.
// For example, given the patterns
//     /a/b/z
//     /a/{x}/c
// we will first try to match the path "/a/b/c" with /a/b/z, and
// when that fails we will try against /a/{x}/c.

package http

import (
	"strings"
)

// A routingNode is a node in the decision tree.
// The same struct is used for leaf and interior nodes.
type routingNode struct {
	// A leaf node holds a single pattern and the Handler it was registered
	// with.
	pattern *pattern
	handler Handler

	// An interior node maps parts of the incoming request to child nodes.
	// special children keys:
	//     "/"	trailing slash (resulting from {$})
	//	   ""   single wildcard
	children   mapping[string, *routingNode]
	multiChild *routingNode // child with multi wildcard
	emptyChild *routingNode // optimization: child with key ""
}

// addPattern adds a pattern and its associated Handler to the tree
// at root.
func (root *routingNode) addPattern(p *pattern, h Handler) {
	// First level of tree is host.
	n := root.addChild(p.host)
	// Second level of tree is method.
	n = n.addChild(p.method)
	// Remaining levels are path.
	n.addSegments(p.segments, p, h)
}

// addSegments adds the given segments to the tree rooted at n.
// If there are no segments, then n is a leaf node that holds
// the given pattern and handler.
func (n *routingNode) addSegments(segs []segment, p *pattern, h Handler) {
	if len(segs) == 0 {
		n.set(p, h)
		return
	}
	seg := segs[0]
	if seg.multi {
		if len(segs) != 1 {
			panic("multi wildcard not last")
		}
		c := &routingNode{}
		n.multiChild = c
		c.set(p, h)
	} else if seg.wild {
		n.addChild("").addSegments(segs[1:], p, h)
	} else {
		n.addChild(seg.s).addSegments(segs[1:], p, h)
	}
}

// set sets the pattern and handler for n, which
// must be a leaf node.
func (n *routingNode) set(p *pattern, h Handler) {
	if n.pattern != nil || n.handler != nil {
		panic("non-nil leaf fields")
	}
	n.pattern = p
	n.handler = h
}

// addChild adds a child node with the given key to n
// if one does not exist, and returns the child.
func (n *routingNode) addChild(key string) *routingNode {
	if key == "" {
		if n.emptyChild == nil {
			n.emptyChild = &routingNode{}
		}
		return n.emptyChild
	}
	if c := n.findChild(key); c != nil {
		return c
	}
	c := &routingNode{}
	n.children.add(key, c)
	return c
}

// findChild returns the child of n with the given key, or nil
// if there is no child with that key.
func (n *routingNode) findChild(key string) *routingNode {
	if key == "" {
		return n.emptyChild
	}
	r, _ := n.children.find(key)
	return r
}

// match returns the leaf node under root that matches the arguments, and a list
// of values for pattern wildcards in the order that the wildcards appear.
// For example, if the request path is "/a/b/c" and the pattern is "/{x}/b/{y}",
// then the second return value will be []string{"a", "c"}.
func (root *routingNode) match(host, method, path string) (*routingNode, []string) {
	if host != "" {
		// There is a host. If there is a pattern that specifies that host and it
		// matches, we are done. If the pattern doesn't match, fall through to
		// try patterns with no host.
		if l, m := root.findChild(host).matchMethodAndPath(method, path); l != nil {
			return l, m
		}
	}
	return root.emptyChild.matchMethodAndPath(method, path)
}

// matchMethodAndPath matches the method and path.
// Its return values are the same as [routingNode.match].
// The receiver should be a child of the root.
func (n *routingNode) matchMethodAndPath(method, path string) (*routingNode, []string) {
	if n == nil {
		return nil, nil
	}
	if l, m := n.findChild(method).matchPath(path, nil); l != nil {
		// Exact match of method name.
		return l, m
	}
	if method == "HEAD" {
		// GET matches HEAD too.
		if l, m := n.findChild("GET").matchPath(path, nil); l != nil {
			return l, m
		}
	}
	// No exact match; try patterns with no method.
	return n.emptyChild.matchPath(path, nil)
}

// matchPath matches a path.
// Its return values are the same as [routingNode.match].
// matchPath calls itself recursively. The matches argument holds the wildcard matches
// found so far.
func (n *routingNode) matchPath(path string, matches []string) (*routingNode, []string) {
	if n == nil {
		return nil, nil
	}
	// If path is empty, then we are done.
	// If n is a leaf node, we found a match; return it.
	// If n is an interior node (which means it has a nil pattern),
	// then we failed to match.
	if path == "" {
		if n.pattern == nil {
			return nil, nil
		}
		return n, matches
	}
	// Get the first segment of path.
	seg, rest := firstSegment(path)
	// First try matching against patterns that have a literal for this position.
	// We know by construction that such patterns are more specific than those
	// with a wildcard at this position (they are either more specific, equivalent,
	// or overlap, and we ruled out the first two when the patterns were registered).
	if n, m := n.findChild(seg).matchPath(rest, matches); n != nil {
		return n, m
	}
	// If matching a literal fails, try again with patterns that have a single
	// wildcard (represented by an empty string in the child mapping).
	// Again, by construction, patterns with a single wildcard must be more specific than
	// those with a multi wildcard.
	// We skip this step if the segment is a trailing slash, because single wildcards
	// don't match trailing slashes.
	if seg != "/" {
		if n, m := n.emptyChild.matchPath(rest, append(matches, seg)); n != nil {
			return n, m
		}
	}
	// Lastly, match the pattern (there can be at most one) that has a multi
	// wildcard in this position to the rest of the path.
	if c := n.multiChild; c != nil {
		// Don't record a match for a nameless wildcard (which arises from a
		// trailing slash in the pattern).
		if c.pattern.lastSegment().s != "" {
			matches = append(matches, pathUnescape(path[1:])) // remove initial slash
		}
		return c, matches
	}
	return nil, nil
}

// firstSegment splits path into its first segment, and the rest.
// The path must begin with "/".
// If path consists of only a slash, firstSegment returns ("/", "").
// The segment is returned unescaped, if possible.
func firstSegment(path string) (seg, rest string) {
	if path == "/" {
		return "/", ""
	}
	path = path[1:] // drop initial slash
	i := strings.IndexByte(path, '/')
	if i < 0 {
		i = len(path)
	}
	return pathUnescape(path[:i]), path[i:]
}

// matchingMethods adds to methodSet all the methods that would result in a
// match if passed to routingNode.match with the given host and path.
func (root *routingNode) matchingMethods(host, path string, methodSet map[string]bool) {
	if host != "" {
		root.findChild(host).matchingMethodsPath(path, methodSet)
	}
	root.emptyChild.matchingMethodsPath(path, methodSet)
	if methodSet["GET"] {
		methodSet["HEAD"] = true
	}
}

func (n *routingNode) matchingMethodsPath(path string, set map[string]bool) {
	if n == nil {
		return
	}
	n.children.eachPair(func { method, c ->
		if p, _ := c.matchPath(path, nil); p != nil {
			set[method] = true
		}
		return true
	})
	// Don't look at the empty child. If there were an empty
	// child, it would match on any method, but we only
	// call this when we fail to match on a method.
}
