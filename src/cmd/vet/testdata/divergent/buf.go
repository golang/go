// Test of examples with divergent packages.

// Package buf ...
package buf

// Buf is a ...
type Buf []byte

// Append ...
func (*Buf) Append([]byte) {}

func (Buf) Reset() {}

func (Buf) Len() int { return 0 }

// DefaultBuf is a ...
var DefaultBuf Buf
