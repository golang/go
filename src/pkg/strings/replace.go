// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import "io"

// A Replacer replaces a list of strings with replacements.
type Replacer struct {
	r replacer
}

// replacer is the interface that a replacement algorithm needs to implement.
type replacer interface {
	Replace(s string) string
	WriteString(w io.Writer, s string) (n int, err error)
}

// byteBitmap represents bytes which are sought for replacement.
// byteBitmap is 256 bits wide, with a bit set for each old byte to be
// replaced.
type byteBitmap [256 / 32]uint32

func (m *byteBitmap) set(b byte) {
	m[b>>5] |= uint32(1 << (b & 31))
}

// NewReplacer returns a new Replacer from a list of old, new string pairs.
// Replacements are performed in order, without overlapping matches.
func NewReplacer(oldnew ...string) *Replacer {
	if len(oldnew)%2 == 1 {
		panic("strings.NewReplacer: odd argument count")
	}

	if len(oldnew) == 2 && len(oldnew[0]) > 1 {
		return &Replacer{r: makeSingleStringReplacer(oldnew[0], oldnew[1])}
	}

	allNewBytes := true
	for i := 0; i < len(oldnew); i += 2 {
		if len(oldnew[i]) != 1 {
			return &Replacer{r: makeGenericReplacer(oldnew)}
		}
		if len(oldnew[i+1]) != 1 {
			allNewBytes = false
		}
	}

	if allNewBytes {
		bb := &byteReplacer{}
		for i := 0; i < len(oldnew); i += 2 {
			o, n := oldnew[i][0], oldnew[i+1][0]
			if bb.old[o>>5]&uint32(1<<(o&31)) != 0 {
				// Later old->new maps do not override previous ones with the same old string.
				continue
			}
			bb.old.set(o)
			bb.new[o] = n
		}
		return &Replacer{r: bb}
	}

	bs := &byteStringReplacer{}
	for i := 0; i < len(oldnew); i += 2 {
		o, new := oldnew[i][0], oldnew[i+1]
		if bs.old[o>>5]&uint32(1<<(o&31)) != 0 {
			// Later old->new maps do not override previous ones with the same old string.
			continue
		}
		bs.old.set(o)
		bs.new[o] = []byte(new)
	}
	return &Replacer{r: bs}
}

// Replace returns a copy of s with all replacements performed.
func (r *Replacer) Replace(s string) string {
	return r.r.Replace(s)
}

// WriteString writes s to w with all replacements performed.
func (r *Replacer) WriteString(w io.Writer, s string) (n int, err error) {
	return r.r.WriteString(w, s)
}

// trieNode is a node in a lookup trie for prioritized key/value pairs. Keys
// and values may be empty. For example, the trie containing keys "ax", "ay",
// "bcbc", "x" and "xy" could have eight nodes:
//
//  n0  -
//  n1  a-
//  n2  .x+
//  n3  .y+
//  n4  b-
//  n5  .cbc+
//  n6  x+
//  n7  .y+
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

type stringWriterIface interface {
	WriteString(string) (int, error)
}

type stringWriter struct {
	w io.Writer
}

func (w stringWriter) WriteString(s string) (int, error) {
	return w.w.Write([]byte(s))
}

func getStringWriter(w io.Writer) stringWriterIface {
	sw, ok := w.(stringWriterIface)
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
	var buf []byte
	i, matched := 0, false
	for {
		match := r.finder.next(s[i:])
		if match == -1 {
			break
		}
		matched = true
		buf = append(buf, s[i:i+match]...)
		buf = append(buf, r.value...)
		i += match + len(r.finder.pattern)
	}
	if !matched {
		return s
	}
	buf = append(buf, s[i:]...)
	return string(buf)
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
type byteReplacer struct {
	// old has a bit set for each old byte that should be replaced.
	old byteBitmap

	// replacement byte, indexed by old byte. only valid if
	// corresponding old bit is set.
	new [256]byte
}

func (r *byteReplacer) Replace(s string) string {
	var buf []byte // lazily allocated
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
			if buf == nil {
				buf = []byte(s)
			}
			buf[i] = r.new[b]
		}
	}
	if buf == nil {
		return s
	}
	return string(buf)
}

func (r *byteReplacer) WriteString(w io.Writer, s string) (n int, err error) {
	// TODO(bradfitz): use io.WriteString with slices of s, avoiding allocation.
	bufsize := 32 << 10
	if len(s) < bufsize {
		bufsize = len(s)
	}
	buf := make([]byte, bufsize)

	for len(s) > 0 {
		ncopy := copy(buf, s[:])
		s = s[ncopy:]
		for i, b := range buf[:ncopy] {
			if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
				buf[i] = r.new[b]
			}
		}
		wn, err := w.Write(buf[:ncopy])
		n += wn
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// byteStringReplacer is the implementation that's used when all the
// "old" values are single ASCII bytes but the "new" values vary in
// size.
type byteStringReplacer struct {
	// old has a bit set for each old byte that should be replaced.
	old byteBitmap

	// replacement string, indexed by old byte. only valid if
	// corresponding old bit is set.
	new [256][]byte
}

func (r *byteStringReplacer) Replace(s string) string {
	newSize := 0
	anyChanges := false
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
			anyChanges = true
			newSize += len(r.new[b])
		} else {
			newSize++
		}
	}
	if !anyChanges {
		return s
	}
	buf := make([]byte, newSize)
	bi := buf
	for i := 0; i < len(s); i++ {
		b := s[i]
		if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
			n := copy(bi[:], r.new[b])
			bi = bi[n:]
		} else {
			bi[0] = b
			bi = bi[1:]
		}
	}
	return string(buf)
}

// WriteString maintains one buffer that's at most 32KB.  The bytes in
// s are enumerated and the buffer is filled.  If it reaches its
// capacity or a byte has a replacement, the buffer is flushed to w.
func (r *byteStringReplacer) WriteString(w io.Writer, s string) (n int, err error) {
	// TODO(bradfitz): use io.WriteString with slices of s instead.
	bufsize := 32 << 10
	if len(s) < bufsize {
		bufsize = len(s)
	}
	buf := make([]byte, bufsize)
	bi := buf[:0]

	for i := 0; i < len(s); i++ {
		b := s[i]
		var new []byte
		if r.old[b>>5]&uint32(1<<(b&31)) != 0 {
			new = r.new[b]
		} else {
			bi = append(bi, b)
		}
		if len(bi) == cap(bi) || (len(bi) > 0 && len(new) > 0) {
			nw, err := w.Write(bi)
			n += nw
			if err != nil {
				return n, err
			}
			bi = buf[:0]
		}
		if len(new) > 0 {
			nw, err := w.Write(new)
			n += nw
			if err != nil {
				return n, err
			}
		}
	}
	if len(bi) > 0 {
		nw, err := w.Write(bi)
		n += nw
		if err != nil {
			return n, err
		}
	}
	return n, nil
}
