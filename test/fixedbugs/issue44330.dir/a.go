// Copyright 2021 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package a

type Table struct {
	ColumnSeparator bool
	RowSeparator    bool

	// ColumnResizer is called on each Draw. Can be used for custom column sizing.
	ColumnResizer func()
}

func NewTable() *Table {
	return &Table{
		ColumnSeparator: true,
		RowSeparator:    true,
		ColumnResizer:   func() {},
	}
}
