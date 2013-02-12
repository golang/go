// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

// Export for testing.
// TODO: no longer necessary. Remove at some point.

import (
	"exp/locale/collate/colltab"
	"fmt"
)

const (
	defaultSecondary = 0x20
	defaultTertiary  = 0x2
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
	}
	return w
}
func (w Weights) String() string {
	return fmt.Sprintf("[%X.%X.%X.%X]", w.Primary, w.Secondary, w.Tertiary, w.Quaternary)
}

func convertFromWeights(ws []Weights) []colltab.Elem {
	out := make([]colltab.Elem, len(ws))
	for i, w := range ws {
		out[i], _ = colltab.MakeElem(w.Primary, w.Secondary, w.Tertiary, 0)
		if out[i] == colltab.Ignore && w.Quaternary > 0 {
			out[i] = colltab.MakeQuaternary(w.Quaternary)
		}
	}
	return out
}
