// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import (
	"io"
	"sync"
)

// Replacer replaces a list of strings with replacements.
// It is safe for concurrent use by multiple goroutines.
type Replacer struct {
	once   sync.Once // guards buildOnce method
	r      replacer
	oldnew []string
}

// replacer is the interface that a replacement algorithm needs to implement.
type replacer interface {
	Replace(s string) string
	WriteString(w io.Writer, s string) (n int, err error)
}

// NewReplacer returns a new [Replacer] from a list of old, new string
// pairs. Replacements are performed in the order they appear in the
// target string, without overlapping matches. The old string
// comparisons are done in argument order.
//
// NewReplacer panics if given an odd number of arguments.
func NewReplacer(oldnew ...string) *Replacer {
	if len(oldnew)%2 == 1 {
		panic("strings.NewReplacer: odd argument count")
	}
	return &Replacer{oldnew: append([]string(nil), oldnew...)}
}

func (r *Replacer) buildOnce() {
	r.r = r.build()
	r.oldnew = nil
}

func (b *Replacer) build() replacer {
	oldnew := b.oldnew
	if len(oldnew) == 2 && len(oldnew[0]) > 1 {
		return makeSingleStringReplacer(oldnew[0], oldnew[1])
	}

	allNewBytes := true
	for i := 0; i < len(oldnew); i += 2 {
		if len(oldnew[i]) != 1 {
			return makeGenericReplacer(oldnew)
		}
		if len(oldnew[i+1]) != 1 {
			allNewBytes = false
		}
	}

	if allNewBytes {
		r := byteReplacer{}
		for i := range r {
			r[i] = byte(i)
		}
		// The first occurrence of old->new map takes precedence
		// over the others with the same old string.
		for i := len(oldnew) - 2; i >= 0; i -= 2 {
			o := oldnew[i][0]
			n := oldnew[i+1][0]
			r[o] = n
		}
		return &r
	}

	r := byteStringReplacer{toReplace: make([]string, 0, len(oldnew)/2)}
	// The first occurrence of old->new map takes precedence
	// over the others with the same old string.
	for i := len(oldnew) - 2; i >= 0; i -= 2 {
		o := oldnew[i][0]
		n := oldnew[i+1]
		// To avoid counting repetitions multiple times.
		if r.replacements[o] == nil {
			// We need to use string([]byte{o}) instead of string(o),
			// to avoid utf8 encoding of o.
			// E. g. byte(150) produces string of length 2.
			r.toReplace = append(r.toReplace, string([]byte{o}))
		}
		r.replacements[o] = []byte(n)

	}
	return &r
}

// Replace returns a copy of s with all replacements performed.
func (r *Replacer) Replace(s string) string {
	r.once.Do(r.buildOnce)
	return r.r.Replace(s)
}

// WriteString writes s to w with all replacements performed.
func (r *Replacer) WriteString(w io.Writer, s string) (n int, err error) {
	r.once.Do(r.buildOnce)
	return r.r.WriteString(w, s)
}

// trieNode is a node in a lookup trie for prioritized key/value pairs. Keys
// and values may be empty. For example, the trie containing keys "ax", "ay",
// "bcbc", "x" and "xy" could have eight nodes:
//
//	n0  -
//	n1  a-
//	n2  .x+
//	n3  .y+
//	n4  b-
//	n5  .cbc+
//	n6  x+
//	n7  .y+
//
// n0 is the root node, and its children are n1, n4 and n6; n1's children are
// n2 and n3; n4's child is n5; n6's child is n7. Nodes n0, n1 and n4 (marked
// with a trailing "-") are partial keys, and nodes n2, n3, n5, n6 and n7
// (marked with a trailing "+") are complete keys.
type trieNode struct {
	// value is the value of the trie node's key/value pair. It is empty if
	// this node is not a complete key.
	value string
	// priority is the priority (higher is more important) of the trie node's
	// key/value pair; keys are not necessarily matched shortest- or longest-
	// first. Priority is positive if this node is a complete key, and zero
	// otherwise. In the example above, positive/zero priorities are marked
	// with a trailing "+" or "-".
	priority int

	// A trie node may have zero, one or more child nodes:
	//  * if the remaining fields are zero, there are no children.
	//  * if prefix and next are non-zero, there is one child in next.
	//  * if table is non-zero, it defines all the children.
	//
	// Prefixes are preferred over tables when there is one child, but the
	// root node always uses a table for lookup efficiency.

	// prefix is the difference in keys between this trie node and the next.
	// In the example above, node n4 has prefix "cbc" and n4's next node is n5.
	// Node n5 has no children and so has zero prefix, next and table fields.
	prefix string
	next   *trieNode

	// table is a lookup table indexed by the next byte in the key, after
	// remapping that byte through genericReplacer.mapping to create a dense
	// index. In the example above, the keys only use 'a', 'b', 'c', 'x' and
	// 'y', which remap to 0, 1, 2, 3 and 4. All other bytes remap to 5, and
	// genericReplacer.tableSize will be 5. Node n0's table will be
	// []*trieNode{ 0:n1, 1:n4, 3:n6 }, where the 0, 1 and 3 are the remapped
	// 'a', 'b' and 'x'.
	table []*trieNode
}

func (t *trieNode) add(key, val string, priority int, r *genericReplacer) {
	if key == "" {
		if t.priority == 0 {
			t.value = val
			t.priority = priority
		}
		return
	}

	if t.prefix != "" {
		// Need to split the prefix among multiple nodes.
		var n int // length of the longest common prefix
		for ; n < len(t.prefix) && n < len(key); n++ {
			if t.prefix[n] != key[n] {
				break
			}
		}
		if n == len(t.prefix) {
			t.next.add(key[n:], val, priority, r)
		} else if n == 0 {
			// First byte differs, start a new lookup table here. Looking up
			// what is currently t.prefix[0] will lead to prefixNode, and
			// looking up key[0] will lead to keyNode.
			var prefixNode *trieNode
			if len(t.prefix) == 1 {
				prefixNode = t.next
			} else {
				prefixNode = &trieNode{
					prefix: t.prefix[1:],
					next:   t.next,
				}
			}
			keyNode := new(trieNode)
			t.table = make([]*trieNode, r.tableSize)
			t.table[r.mapping[t.prefix[0]]] = prefixNode
			t.table[r.mapping[key[0]]] = keyNode
			t.prefix = ""
			t.next = nil
			keyNode.add(key[1:], val, priority, r)
		} else {
			// Insert new node after the common section of the prefix.
			next := &trieNode{
				prefix: t.prefix[n:],
				next:   t.next,
			}
			t.prefix = t.prefix[:n]
			t.next = next
			next.add(key[n:], val, priority, r)
		}
	} else if t.table != nil {
		// Insert into existing table.
		m := r.mapping[key[0]]
		if t.table[m] == nil {
			t.table[m] = new(trieNode)
		}
		t.table[m].add(key[1:], val, priority, r)
	} else {
		t.prefix = key
		t.next = new(trieNode)
		t.next.add("", val, priority, r)
	}
}

func (r *genericReplacer) lookup(s string, ignoreRoot bool) (val string, keylen int, found bool) {
	// Iterate down the trie to the end, and grab the value and keylen with
	// the highest priority.
	bestPriority := 0
	node := &r.root
	n := 0
	for node != nil {
		if node.priority > bestPriority && !(ignoreRoot && node == &r.root) {
			bestPriority = node.priority
			val = node.value
			keylen = n
			found = true
		}

		if s == "" {
			break
		}
		if node.table != nil {
			index := r.mapping[s[0]]
			if int(index) == r.tableSize {
				break
			}
			node = node.table[index]
			s = s[1:]
			n++
		} else if node.prefix != "" && HasPrefix(s, node.prefix) {
			n += len(node.prefix)
			s = s[len(node.prefix):]
			node = node.next
		} else {
			break
		}
	}
	return
}

// genericReplacer is the fully generic algorithm.
// It's used as a fallback when nothing faster can be used.
type genericReplacer struct {
	root trieNode
	// tableSize is the size of a trie node's lookup table. It is the number
	// of unique key bytes.
	tableSize int
	// mapping maps from key bytes to a dense index for trieNode.table.
	mapping [256]byte
}

func makeGenericReplacer(oldnew []string) *genericReplacer {
	r := new(genericReplacer)
	// Find each byte used, then assign them each an index.
	for i := 0; i < len(oldnew); i += 2 {
		key := oldnew[i]
		for j := 0; j < len(key); j++ {
			r.mapping[key[j]] = 1
		}
	}

	for _, b := range r.mapping {
		r.tableSize += int(b)
	}

	var index byte
	for i, b := range r.mapping {
		if b == 0 {
			r.mapping[i] = byte(r.tableSize)
		} else {
			r.mapping[i] = index
			index++
		}
	}
	// Ensure root node uses a lookup table (for performance).
	r.root.table = make([]*trieNode, r.tableSize)

	for i := 0; i < len(oldnew); i += 2 {
		r.root.add(oldnew[i], oldnew[i+1], len(oldnew)-i, r)
	}
	return r
}

type appendSliceWriter []byte

// Write writes to the buffer to satisfy io.Writer.
func (w *appendSliceWriter) Write(p []byte) (int, error) {
	*w = append(*w, p...)
	return len(p), nil
}

// WriteString writes to the buffer without string->[]byte->string allocations.
func (w *appendSliceWriter) WriteString(s string) (int, error) {
	*w = append(*w, s...)
	return len(s), nil
}

type stringWriter struct {
	w io.Writer
}

func (w stringWriter) WriteString(s string) (int, error) {
	return w.w.Write([]byte(s))
}

func getStringWriter(w io.Writer) io.StringWriter {
	sw, ok := w.(io.StringWriter)
	if !ok {
		sw = stringWriter{w}
	}
	return sw
}

func (r *genericReplacer) Replace(s string) string {
	buf := make(appendSliceWriter, 0, len(s))
	r.WriteString(&buf, s)
	return string(buf)
}

func (r *genericReplacer) WriteString(w io.Writer, s string) (n int, err error) {
	sw := getStringWriter(w)
	var last, wn int
	var prevMatchEmpty bool
	for i := 0; i <= len(s); {
		// Fast path: s[i] is not a prefix of any pattern.
		if i != len(s) && r.root.priority == 0 {
			index := int(r.mapping[s[i]])
			if index == r.tableSize || r.root.table[index] == nil {
				i++
				continue
			}
		}

		// Ignore the empty match iff the previous loop found the empty match.
		val, keylen, match := r.lookup(s[i:], prevMatchEmpty)
		prevMatchEmpty = match && keylen == 0
		if match {
			wn, err = sw.WriteString(s[last:i])
			n += wn
			if err != nil {
				return
			}
			wn, err = sw.WriteString(val)
			n += wn
			if err != nil {
				return
			}
			i += keylen
			last = i
			continue
		}
		i++
	}
	if last != len(s) {
		wn, err = sw.WriteString(s[last:])
		n += wn
	}
	return
}

// singleStringReplacer is the implementation that's used when there is only
// one string to replace (and that string has more than one byte).
type singleStringReplacer struct {
	finder *stringFinder
	// value is the new string that replaces that pattern when it's found.
	value string
}

func makeSingleStringReplacer(pattern string, value string) *singleStringReplacer {
	return &singleStringReplacer{finder: makeStringFinder(pattern), value: value}
}

func (r *singleStringReplacer) Replace(s string) string {
	var buf Builder
	i, matched := 0, false
	for {
		match := r.finder.next(s[i:])
		if match == -1 {
			break
		}
		matched = true
		buf.Grow(match + len(r.value))
		buf.WriteString(s[i : i+match])
		buf.WriteString(r.value)
		i += match + len(r.finder.pattern)
	}
	if !matched {
		return s
	}
	buf.WriteString(s[i:])
	return buf.String()
}

func (r *singleStringReplacer) WriteString(w io.Writer, s string) (n int, err error) {
	sw := getStringWriter(w)
	var i, wn int
	for {
		match := r.finder.next(s[i:])
		if match == -1 {
			break
		}
		wn, err = sw.WriteString(s[i : i+match])
		n += wn
		if err != nil {
			return
		}
		wn, err = sw.WriteString(r.value)
		n += wn
		if err != nil {
			return
		}
		i += match + len(r.finder.pattern)
	}
	wn, err = sw.WriteString(s[i:])
	n += wn
	return
}

// byteReplacer is the implementation that's used when all the "old"
// and "new" values are single ASCII bytes.
// The array contains replacement bytes indexed by old byte.
type byteReplacer [256]byte

func (r *byteReplacer) Replace(s string) string {
	var buf []byte // lazily allocated
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r[b] != b {
			if buf == nil {
				buf = []byte(s)
			}
			buf[i] = r[b]
		}
	}
	if buf == nil {
		return s
	}
	return string(buf)
}

func (r *byteReplacer) WriteString(w io.Writer, s string) (n int, err error) {
	sw := getStringWriter(w)
	last := 0
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r[b] == b {
			continue
		}
		if last != i {
			wn, err := sw.WriteString(s[last:i])
			n += wn
			if err != nil {
				return n, err
			}
		}
		last = i + 1
		nw, err := w.Write(r[b : int(b)+1])
		n += nw
		if err != nil {
			return n, err
		}
	}
	if last != len(s) {
		nw, err := sw.WriteString(s[last:])
		n += nw
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// byteStringReplacer is the implementation that's used when all the
// "old" values are single ASCII bytes but the "new" values vary in size.
type byteStringReplacer struct {
	// replacements contains replacement byte slices indexed by old byte.
	// A nil []byte means that the old byte should not be replaced.
	replacements [256][]byte
	// toReplace keeps a list of bytes to replace. Depending on length of toReplace
	// and length of target string it may be faster to use Count, or a plain loop.
	// We store single byte as a string, because Count takes a string.
	toReplace []string
}

// countCutOff controls the ratio of a string length to a number of replacements
// at which (*byteStringReplacer).Replace switches algorithms.
// For strings with higher ration of length to replacements than that value,
// we call Count, for each replacement from toReplace.
// For strings, with a lower ratio we use simple loop, because of Count overhead.
// countCutOff is an empirically determined overhead multiplier.
// TODO(tocarip) revisit once we have register-based abi/mid-stack inlining.
const countCutOff = 8

func (r *byteStringReplacer) Replace(s string) string {
	newSize := len(s)
	anyChanges := false
	// Is it faster to use Count?
	if len(r.toReplace)*countCutOff <= len(s) {
		for _, x := range r.toReplace {
			if c := Count(s, x); c != 0 {
				// The -1 is because we are replacing 1 byte with len(replacements[b]) bytes.
				newSize += c * (len(r.replacements[x[0]]) - 1)
				anyChanges = true
			}

		}
	} else {
		for i := 0; i < len(s); i++ {
			b := s[i]
			if r.replacements[b] != nil {
				// See above for explanation of -1
				newSize += len(r.replacements[b]) - 1
				anyChanges = true
			}
		}
	}
	if !anyChanges {
		return s
	}
	buf := make([]byte, newSize)
	j := 0
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r.replacements[b] != nil {
			j += copy(buf[j:], r.replacements[b])
		} else {
			buf[j] = b
			j++
		}
	}
	return string(buf)
}

func (r *byteStringReplacer) WriteString(w io.Writer, s string) (n int, err error) {
	sw := getStringWriter(w)
	last := 0
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r.replacements[b] == nil {
			continue
		}
		if last != i {
			nw, err := sw.WriteString(s[last:i])
			n += nw
			if err != nil {
				return n, err
			}
		}
		last = i + 1
		nw, err := w.Write(r.replacements[b])
		n += nw
		if err != nil {
			return n, err
		}
	}
	if last != len(s) {
		var nw int
		nw, err = sw.WriteString(s[last:])
		n += nw
	}
	return
}
