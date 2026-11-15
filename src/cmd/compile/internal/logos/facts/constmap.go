// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"fmt"
	"strconv"
	"strings"
)

func encodeConstMap(c ConstMapFact) ([]byte, bool) {
	if c.Object == "" || !ValidObject(c.Object) || (c.Path != "" && !validCompositePath(c.Path)) {
		return nil, false
	}
	if len(c.Keys) != len(c.Vals) || len(c.Keys) != len(c.Hashes) ||
		len(c.Keys) != len(c.Canons) || len(c.Order) != len(c.Keys) {
		return nil, false
	}
	var b strings.Builder
	b.WriteString("bearing ")
	if c.Bearing {
		b.WriteString("1\n")
	} else {
		b.WriteString("0\n")
	}
	b.WriteString("order")
	for _, o := range c.Order {
		if o < 0 || o >= len(c.Keys) {
			return nil, false
		}
		fmt.Fprintf(&b, " %d", o)
	}
	b.WriteString("\n")
	for i := range c.Keys {
		fmt.Fprintf(&b, "entry %x %d %d %d\n", c.Hashes[i], len(c.Canons[i]), len(c.Keys[i]), len(c.Vals[i]))
		b.WriteString(c.Canons[i])
		b.WriteString(c.Keys[i])
		b.WriteString(c.Vals[i])
	}
	return []byte(b.String()), true
}

func decodeConstMap(object string, body []byte) (ConstMapFact, error) {
	var c ConstMapFact
	c.Object = object
	rest := string(body)
	bearing, after, ok := cutLine(rest)
	if !ok || (bearing != "bearing 0" && bearing != "bearing 1") {
		return c, fmt.Errorf("bad constmap bearing line %q", bearing)
	}
	rest = after
	orderLine, after, ok := cutLine(rest)
	if !ok || (orderLine != "order" && !strings.HasPrefix(orderLine, "order ")) {
		return c, fmt.Errorf("bad constmap order line %q", orderLine)
	}
	rest = after
	if orderLine != "order" {
		for f := range strings.FieldsSeq(strings.TrimPrefix(orderLine, "order ")) {
			o, err := strconv.Atoi(f)
			if err != nil || o < 0 {
				return c, fmt.Errorf("bad constmap order index %q", f)
			}
			c.Order = append(c.Order, o)
		}
	}
	for len(rest) > 0 {
		var header string
		header, rest, ok = cutLine(rest)
		if !ok {
			return c, fmt.Errorf("truncated constmap entry header")
		}
		fields := strings.Fields(header)
		if len(fields) != 5 || fields[0] != "entry" {
			return c, fmt.Errorf("bad constmap entry header %q", header)
		}
		hash, err := strconv.ParseUint(fields[1], 16, 64)
		if err != nil {
			return c, fmt.Errorf("bad constmap entry hash %q", fields[1])
		}
		lens := make([]int, 3)
		for i, f := range fields[2:] {
			n, err := strconv.Atoi(f)
			if err != nil || n < 0 || n > len(rest) {
				return c, fmt.Errorf("bad constmap entry length %q", f)
			}
			lens[i] = n
		}
		need := lens[0] + lens[1] + lens[2]
		if need > len(rest) {
			return c, fmt.Errorf("truncated constmap entry body")
		}
		c.Hashes = append(c.Hashes, hash)
		c.Canons = append(c.Canons, rest[:lens[0]])
		c.Keys = append(c.Keys, rest[lens[0]:lens[0]+lens[1]])
		c.Vals = append(c.Vals, rest[lens[0]+lens[1]:need])
		rest = rest[need:]
	}
	if len(c.Order) != len(c.Keys) {
		return c, fmt.Errorf("constmap order length %d mismatches entry count %d", len(c.Order), len(c.Keys))
	}
	seen := make(map[int]bool, len(c.Order))
	for _, o := range c.Order {
		if o >= len(c.Keys) || seen[o] {
			return c, fmt.Errorf("bad constmap order index %d", o)
		}
		seen[o] = true
	}
	canons := make(map[string]bool, len(c.Canons))
	for _, k := range c.Canons {
		if canons[k] {
			return c, fmt.Errorf("duplicate constmap key %q", k)
		}
		canons[k] = true
	}
	return c, nil
}

func cutLine(s string) (line, rest string, ok bool) {
	i := strings.IndexByte(s, '\n')
	if i < 0 {
		return "", "", false
	}
	return s[:i], s[i+1:], true
}
