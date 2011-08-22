// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type ByteSize float64

const (
	_           = iota // ignore first value by assigning to blank identifier
	KB ByteSize = 1 << (10 * iota)
	MB
	GB
	TB
	PB
	EB
	ZB
	YB
)

func (b ByteSize) String() string {
	switch {
	case b >= YB:
		return fmt.Sprintf("%.2fYB", float64(b/YB))
	case b >= ZB:
		return fmt.Sprintf("%.2fZB", float64(b/ZB))
	case b >= EB:
		return fmt.Sprintf("%.2fEB", float64(b/EB))
	case b >= PB:
		return fmt.Sprintf("%.2fPB", float64(b/PB))
	case b >= TB:
		return fmt.Sprintf("%.2fTB", float64(b/TB))
	case b >= GB:
		return fmt.Sprintf("%.2fGB", float64(b/GB))
	case b >= MB:
		return fmt.Sprintf("%.2fMB", float64(b/MB))
	case b >= KB:
		return fmt.Sprintf("%.2fKB", float64(b/KB))
	}
	return fmt.Sprintf("%.2fB", float64(b))
}

func main() {
	fmt.Println(YB, ByteSize(1e13))
}
