// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"fmt"
)

// checkTreeConsistency checks that a node and its descendants are all
// consistent in their parent/child/sibling relationships.
func checkTreeConsistency(n *Node) error {
	return checkTreeConsistency1(n, 0)
}

func checkTreeConsistency1(n *Node, depth int) error {
	if depth == 1e4 {
		return fmt.Errorf("html: tree looks like it contains a cycle")
	}
	if err := checkNodeConsistency(n); err != nil {
		return err
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if err := checkTreeConsistency1(c, depth+1); err != nil {
			return err
		}
	}
	return nil
}

// checkNodeConsistency checks that a node's parent/child/sibling relationships
// are consistent.
func checkNodeConsistency(n *Node) error {
	if n == nil {
		return nil
	}

	nParent := 0
	for p := n.Parent; p != nil; p = p.Parent {
		nParent++
		if nParent == 1e4 {
			return fmt.Errorf("html: parent list looks like an infinite loop")
		}
	}

	nForward := 0
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		nForward++
		if nForward == 1e6 {
			return fmt.Errorf("html: forward list of children looks like an infinite loop")
		}
		if c.Parent != n {
			return fmt.Errorf("html: inconsistent child/parent relationship")
		}
	}

	nBackward := 0
	for c := n.LastChild; c != nil; c = c.PrevSibling {
		nBackward++
		if nBackward == 1e6 {
			return fmt.Errorf("html: backward list of children looks like an infinite loop")
		}
		if c.Parent != n {
			return fmt.Errorf("html: inconsistent child/parent relationship")
		}
	}

	if n.Parent != nil {
		if n.Parent == n {
			return fmt.Errorf("html: inconsistent parent relationship")
		}
		if n.Parent == n.FirstChild {
			return fmt.Errorf("html: inconsistent parent/first relationship")
		}
		if n.Parent == n.LastChild {
			return fmt.Errorf("html: inconsistent parent/last relationship")
		}
		if n.Parent == n.PrevSibling {
			return fmt.Errorf("html: inconsistent parent/prev relationship")
		}
		if n.Parent == n.NextSibling {
			return fmt.Errorf("html: inconsistent parent/next relationship")
		}

		parentHasNAsAChild := false
		for c := n.Parent.FirstChild; c != nil; c = c.NextSibling {
			if c == n {
				parentHasNAsAChild = true
				break
			}
		}
		if !parentHasNAsAChild {
			return fmt.Errorf("html: inconsistent parent/child relationship")
		}
	}

	if n.PrevSibling != nil && n.PrevSibling.NextSibling != n {
		return fmt.Errorf("html: inconsistent prev/next relationship")
	}
	if n.NextSibling != nil && n.NextSibling.PrevSibling != n {
		return fmt.Errorf("html: inconsistent next/prev relationship")
	}

	if (n.FirstChild == nil) != (n.LastChild == nil) {
		return fmt.Errorf("html: inconsistent first/last relationship")
	}
	if n.FirstChild != nil && n.FirstChild == n.LastChild {
		// We have a sole child.
		if n.FirstChild.PrevSibling != nil || n.FirstChild.NextSibling != nil {
			return fmt.Errorf("html: inconsistent sole child's sibling relationship")
		}
	}

	seen := map[*Node]bool{}

	var last *Node
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if seen[c] {
			return fmt.Errorf("html: inconsistent repeated child")
		}
		seen[c] = true
		last = c
	}
	if last != n.LastChild {
		return fmt.Errorf("html: inconsistent last relationship")
	}

	var first *Node
	for c := n.LastChild; c != nil; c = c.PrevSibling {
		if !seen[c] {
			return fmt.Errorf("html: inconsistent missing child")
		}
		delete(seen, c)
		first = c
	}
	if first != n.FirstChild {
		return fmt.Errorf("html: inconsistent first relationship")
	}

	if len(seen) != 0 {
		return fmt.Errorf("html: inconsistent forwards/backwards child list")
	}

	return nil
}
