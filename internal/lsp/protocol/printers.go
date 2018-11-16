// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains formatting functions for types that
// are commonly printed in debugging information.
// They are separated from their types and gathered here as
// they are hand written and not generated from the spec.
// They should not be relied on for programmatic use (their
// results should never be parsed for instance) but are meant
// for temporary debugging and error messages.

package protocol

import (
	"fmt"
)

func (p Position) Format(f fmt.State, c rune) {
	fmt.Fprintf(f, "%d", int(p.Line))
	if p.Character >= 0 {
		fmt.Fprintf(f, ":%d", int(p.Character))
	}
}

func (r Range) Format(f fmt.State, c rune) {
	switch {
	case r.Start == r.End || r.End.Line < 0:
		fmt.Fprintf(f, "%v", r.Start)
	case r.End.Line == r.Start.Line:
		fmt.Fprintf(f, "%v¦%d", r.Start, int(r.End.Character))
	default:
		fmt.Fprintf(f, "%v¦%v", r.Start, r.End)
	}
}

func (l Location) Format(f fmt.State, c rune) {
	fmt.Fprintf(f, "%s:%v", l.URI, l.Range)
}
