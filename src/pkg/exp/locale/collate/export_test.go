// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

// Export for testing.

import "fmt"

type Weights struct {
	Primary, Secondary, Tertiary int
}

func W(ce ...int) Weights {
	w := Weights{ce[0], defaultSecondary, defaultTertiary}
	if len(ce) > 1 {
		w.Secondary = ce[1]
	}
	if len(ce) > 2 {
		w.Tertiary = ce[2]
	}
	return w
}
func (w Weights) String() string {
	return fmt.Sprintf("[%d.%d.%d]", w.Primary, w.Secondary, w.Tertiary)
}

type Table struct {
	t *table
	w []weights
}

func GetTable(c *Collator) *Table {
	return &Table{c.t, nil}
}

func convertWeights(ws []weights) []Weights {
	out := make([]Weights, len(ws))
	for i, w := range ws {
		out[i] = Weights{int(w.primary), int(w.secondary), int(w.tertiary)}
	}
	return out
}

func (t *Table) AppendNext(s []byte) ([]Weights, int) {
	w, n := t.t.appendNext(nil, s)
	return convertWeights(w), n
}
