// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lib

type Reader struct{ B byte }

func (r *Reader) read() byte {
	if r == nil {
		return 9
	}
	return r.B
}

type reader interface {
	*Reader
	read() byte
}

type Alias = *Reader

type aliasReader interface {
	Alias
	read() byte
}

type incompatibleReader struct{}

func (r *incompatibleReader) read() int { return 0 }

// Even though the constraint's type set is empty, its generic function
// body must be written using the signature specified by the constraint.
type emptyReader interface {
	*incompatibleReader
	read() byte
}

func Empty[T emptyReader](r T) byte { return r.read() }

type OtherReader struct{ B byte }

func (r *OtherReader) read() byte { return r.B + 1 }

type unionReader interface {
	*Reader | *OtherReader
	read() byte
}

type Embedded struct{ *Reader }

type promotedReader interface {
	*Embedded
	read() byte
}

func Direct[T reader](r T) byte           { return r.read() }
func Forward[T aliasReader](r T) byte     { return Direct(r) }
func ViaAlias[T aliasReader](r T) byte    { return r.read() }
func TheValue[T reader](r T) func() byte  { return r.read }
func TheExpr[T reader](r T) func(T) byte  { return T.read }
func Union[T unionReader](r T) byte       { return r.read() }
func Promoted[T promotedReader](r T) byte { return r.read() }
