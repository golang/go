// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

// Export for testing.

import (
	"fmt"
)

type Weights struct {
	Primary, Secondary, Tertiary, Quaternary int
}

func W(ce ...int) Weights {
	w := Weights{ce[0], defaultSecondary, defaultTertiary, 0}
	if len(ce) > 1 {
		w.Secondary = ce[1]
	}
	if len(ce) > 2 {
		w.Tertiary = ce[2]
	}
	if len(ce) > 3 {
		w.Quaternary = ce[3]
	} else if w.Tertiary != 0 {
		w.Quaternary = MaxQuaternary
	}
	return w
}
func (w Weights) String() string {
	return fmt.Sprintf("[%X.%X.%X.%X]", w.Primary, w.Secondary, w.Tertiary, w.Quaternary)
}

type Table struct {
	t Weigher
}

func GetTable(c *Collator) *Table {
	return &Table{c.t}
}

func convertToWeights(ws []Elem) []Weights {
	out := make([]Weights, len(ws))
	for i, w := range ws {
		out[i] = Weights{int(w.Primary()), int(w.Secondary()), int(w.Tertiary()), int(w.Quaternary())}
	}
	return out
}

func convertFromWeights(ws []Weights) []Elem {
	out := make([]Elem, len(ws))
	for i, w := range ws {
		out[i] = makeCE([]int{w.Primary, w.Secondary, w.Tertiary, 0})
		if out[i] == ceIgnore && w.Quaternary > 0 {
			out[i] = MakeQuaternary(w.Quaternary)
		}
	}
	return out
}

func (t *Table) AppendNext(s []byte) ([]Weights, int) {
	w, n := t.t.AppendNext(nil, s)
	return convertToWeights(w), n
}

func SetTop(c *Collator, top int) {
	if c.t == nil {
		c.t = &table{}
	}
	c.variableTop = uint32(top)
}

func GetColElems(c *Collator, str []byte) []Weights {
	ce := c.getColElems(str)
	return convertToWeights(ce)
}

func ProcessWeights(h AlternateHandling, top int, w []Weights) []Weights {
	in := convertFromWeights(w)
	processWeights(h, uint32(top), in)
	return convertToWeights(in)
}

func KeyFromElems(c *Collator, buf *Buffer, w []Weights) []byte {
	k := len(buf.key)
	c.keyFromElems(buf, convertFromWeights(w))
	return buf.key[k:]
}
