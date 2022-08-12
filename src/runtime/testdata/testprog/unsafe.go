package main

import "unsafe"

func init() {
	register("panicOnNilAndEleSizeIsZero", panicOnNilAndEleSizeIsZero)
}

func panicOnNilAndEleSizeIsZero() {
	var p *struct{}
	_ = unsafe.Slice(p, 5)
}
