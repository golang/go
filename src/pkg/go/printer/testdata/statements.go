// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package statements

var expr bool;

func _() {
	if {}
	if expr{}
	if _:=expr;{}
	if _:=expr; expr {}
}


func _() {
	switch {}
	switch expr {}
	switch _ := expr; {}
	switch _ := expr; expr {}
}


func _() {
	for{}
	for expr {}
	for;;{}  // TODO ok to lose the semicolons here?
	for _ :=expr;; {}
	for; expr;{}  // TODO ok to lose the semicolons here?
	for; ; expr = false {}
	for _ :=expr; expr; {}
	for _ := expr;; expr=false {}
	for;expr;expr =false {}
	for _ := expr;expr;expr = false {}
	for _ := range []int{} {}
}
