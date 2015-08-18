// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests floating point arithmetic expressions

package main

import "fmt"

// manysub_ssa is designed to tickle bugs that depend on register
// pressure or unfriendly operand ordering in registers (and at
// least once it succeeded in this).
func manysub_ssa(a, b, c, d float64) (aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd float64) {
	switch {
	}
	aa = a + 11.0 - a
	ab = a - b
	ac = a - c
	ad = a - d
	ba = b - a
	bb = b + 22.0 - b
	bc = b - c
	bd = b - d
	ca = c - a
	cb = c - b
	cc = c + 33.0 - c
	cd = c - d
	da = d - a
	db = d - b
	dc = d - c
	dd = d + 44.0 - d
	return
}

func add64_ssa(a, b float64) float64 {
	switch {
	}
	return a + b
}

func mul64_ssa(a, b float64) float64 {
	switch {
	}
	return a * b
}

func sub64_ssa(a, b float64) float64 {
	switch {
	}
	return a - b
}

func div64_ssa(a, b float64) float64 {
	switch {
	}
	return a / b
}

func add32_ssa(a, b float32) float32 {
	switch {
	}
	return a + b
}

func mul32_ssa(a, b float32) float32 {
	switch {
	}
	return a * b
}

func sub32_ssa(a, b float32) float32 {
	switch {
	}
	return a - b
}
func div32_ssa(a, b float32) float32 {
	switch {
	}
	return a / b
}

func conv2Float64_ssa(a int8, b uint8, c int16, d uint16,
	e int32, f uint32, g int64, h uint64, i float32) (aa, bb, cc, dd, ee, ff, gg, hh, ii float64) {
	switch {
	}
	aa = float64(a)
	bb = float64(b)
	cc = float64(c)
	hh = float64(h)
	dd = float64(d)
	ee = float64(e)
	ff = float64(f)
	gg = float64(g)
	ii = float64(i)
	return
}

func conv2Float32_ssa(a int8, b uint8, c int16, d uint16,
	e int32, f uint32, g int64, h uint64, i float64) (aa, bb, cc, dd, ee, ff, gg, hh, ii float32) {
	switch {
	}
	aa = float32(a)
	bb = float32(b)
	cc = float32(c)
	dd = float32(d)
	ee = float32(e)
	ff = float32(f)
	gg = float32(g)
	hh = float32(h)
	ii = float32(i)
	return
}

func integer2floatConversions() int {
	fails := 0
	{
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(0, 0, 0, 0, 0, 0, 0, 0, 0)
		fails += expectAll64("zero64", 0, a, b, c, d, e, f, g, h, i)
	}
	{
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(1, 1, 1, 1, 1, 1, 1, 1, 1)
		fails += expectAll64("one64", 1, a, b, c, d, e, f, g, h, i)
	}
	{
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(0, 0, 0, 0, 0, 0, 0, 0, 0)
		fails += expectAll32("zero32", 0, a, b, c, d, e, f, g, h, i)
	}
	{
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(1, 1, 1, 1, 1, 1, 1, 1, 1)
		fails += expectAll32("one32", 1, a, b, c, d, e, f, g, h, i)
	}
	{
		// Check maximum values
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(127, 255, 32767, 65535, 0x7fffffff, 0xffffffff, 0x7fffFFFFffffFFFF, 0xffffFFFFffffFFFF, 3.402823E38)
		fails += expect64("a", a, 127)
		fails += expect64("b", b, 255)
		fails += expect64("c", c, 32767)
		fails += expect64("d", d, 65535)
		fails += expect64("e", e, float64(int32(0x7fffffff)))
		fails += expect64("f", f, float64(uint32(0xffffffff)))
		fails += expect64("g", g, float64(int64(0x7fffffffffffffff)))
		fails += expect64("h", h, float64(uint64(0xffffffffffffffff)))
		fails += expect64("i", i, float64(float32(3.402823E38)))
	}
	{
		// Check minimum values (and tweaks for unsigned)
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(-128, 254, -32768, 65534, ^0x7fffffff, 0xfffffffe, ^0x7fffFFFFffffFFFF, 0xffffFFFFffffF401, 1.5E-45)
		fails += expect64("a", a, -128)
		fails += expect64("b", b, 254)
		fails += expect64("c", c, -32768)
		fails += expect64("d", d, 65534)
		fails += expect64("e", e, float64(^int32(0x7fffffff)))
		fails += expect64("f", f, float64(uint32(0xfffffffe)))
		fails += expect64("g", g, float64(^int64(0x7fffffffffffffff)))
		fails += expect64("h", h, float64(uint64(0xfffffffffffff401)))
		fails += expect64("i", i, float64(float32(1.5E-45)))
	}
	{
		// Check maximum values
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(127, 255, 32767, 65535, 0x7fffffff, 0xffffffff, 0x7fffFFFFffffFFFF, 0xffffFFFFffffFFFF, 3.402823E38)
		fails += expect32("a", a, 127)
		fails += expect32("b", b, 255)
		fails += expect32("c", c, 32767)
		fails += expect32("d", d, 65535)
		fails += expect32("e", e, float32(int32(0x7fffffff)))
		fails += expect32("f", f, float32(uint32(0xffffffff)))
		fails += expect32("g", g, float32(int64(0x7fffffffffffffff)))
		fails += expect32("h", h, float32(uint64(0xffffffffffffffff)))
		fails += expect32("i", i, float32(float64(3.402823E38)))
	}
	{
		// Check minimum values (and tweaks for unsigned)
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(-128, 254, -32768, 65534, ^0x7fffffff, 0xfffffffe, ^0x7fffFFFFffffFFFF, 0xffffFFFFffffF401, 1.5E-45)
		fails += expect32("a", a, -128)
		fails += expect32("b", b, 254)
		fails += expect32("c", c, -32768)
		fails += expect32("d", d, 65534)
		fails += expect32("e", e, float32(^int32(0x7fffffff)))
		fails += expect32("f", f, float32(uint32(0xfffffffe)))
		fails += expect32("g", g, float32(^int64(0x7fffffffffffffff)))
		fails += expect32("h", h, float32(uint64(0xfffffffffffff401)))
		fails += expect32("i", i, float32(float64(1.5E-45)))
	}
	return fails
}

const (
	aa = 0x1000000000000000
	ab = 0x100000000000000
	ac = 0x10000000000000
	ad = 0x1000000000000
	ba = 0x100000000000
	bb = 0x10000000000
	bc = 0x1000000000
	bd = 0x100000000
	ca = 0x10000000
	cb = 0x1000000
	cc = 0x100000
	cd = 0x10000
	da = 0x1000
	db = 0x100
	dc = 0x10
	dd = 0x1
)

func compares64_ssa(a, b, c, d float64) (lt, le, eq, ne, ge, gt uint64) {

	switch {
	}

	if a < a {
		lt += aa
	}
	if a < b {
		lt += ab
	}
	if a < c {
		lt += ac
	}
	if a < d {
		lt += ad
	}

	if b < a {
		lt += ba
	}
	if b < b {
		lt += bb
	}
	if b < c {
		lt += bc
	}
	if b < d {
		lt += bd
	}

	if c < a {
		lt += ca
	}
	if c < b {
		lt += cb
	}
	if c < c {
		lt += cc
	}
	if c < d {
		lt += cd
	}

	if d < a {
		lt += da
	}
	if d < b {
		lt += db
	}
	if d < c {
		lt += dc
	}
	if d < d {
		lt += dd
	}

	if a <= a {
		le += aa
	}
	if a <= b {
		le += ab
	}
	if a <= c {
		le += ac
	}
	if a <= d {
		le += ad
	}

	if b <= a {
		le += ba
	}
	if b <= b {
		le += bb
	}
	if b <= c {
		le += bc
	}
	if b <= d {
		le += bd
	}

	if c <= a {
		le += ca
	}
	if c <= b {
		le += cb
	}
	if c <= c {
		le += cc
	}
	if c <= d {
		le += cd
	}

	if d <= a {
		le += da
	}
	if d <= b {
		le += db
	}
	if d <= c {
		le += dc
	}
	if d <= d {
		le += dd
	}

	if a == a {
		eq += aa
	}
	if a == b {
		eq += ab
	}
	if a == c {
		eq += ac
	}
	if a == d {
		eq += ad
	}

	if b == a {
		eq += ba
	}
	if b == b {
		eq += bb
	}
	if b == c {
		eq += bc
	}
	if b == d {
		eq += bd
	}

	if c == a {
		eq += ca
	}
	if c == b {
		eq += cb
	}
	if c == c {
		eq += cc
	}
	if c == d {
		eq += cd
	}

	if d == a {
		eq += da
	}
	if d == b {
		eq += db
	}
	if d == c {
		eq += dc
	}
	if d == d {
		eq += dd
	}

	if a != a {
		ne += aa
	}
	if a != b {
		ne += ab
	}
	if a != c {
		ne += ac
	}
	if a != d {
		ne += ad
	}

	if b != a {
		ne += ba
	}
	if b != b {
		ne += bb
	}
	if b != c {
		ne += bc
	}
	if b != d {
		ne += bd
	}

	if c != a {
		ne += ca
	}
	if c != b {
		ne += cb
	}
	if c != c {
		ne += cc
	}
	if c != d {
		ne += cd
	}

	if d != a {
		ne += da
	}
	if d != b {
		ne += db
	}
	if d != c {
		ne += dc
	}
	if d != d {
		ne += dd
	}

	if a >= a {
		ge += aa
	}
	if a >= b {
		ge += ab
	}
	if a >= c {
		ge += ac
	}
	if a >= d {
		ge += ad
	}

	if b >= a {
		ge += ba
	}
	if b >= b {
		ge += bb
	}
	if b >= c {
		ge += bc
	}
	if b >= d {
		ge += bd
	}

	if c >= a {
		ge += ca
	}
	if c >= b {
		ge += cb
	}
	if c >= c {
		ge += cc
	}
	if c >= d {
		ge += cd
	}

	if d >= a {
		ge += da
	}
	if d >= b {
		ge += db
	}
	if d >= c {
		ge += dc
	}
	if d >= d {
		ge += dd
	}

	if a > a {
		gt += aa
	}
	if a > b {
		gt += ab
	}
	if a > c {
		gt += ac
	}
	if a > d {
		gt += ad
	}

	if b > a {
		gt += ba
	}
	if b > b {
		gt += bb
	}
	if b > c {
		gt += bc
	}
	if b > d {
		gt += bd
	}

	if c > a {
		gt += ca
	}
	if c > b {
		gt += cb
	}
	if c > c {
		gt += cc
	}
	if c > d {
		gt += cd
	}

	if d > a {
		gt += da
	}
	if d > b {
		gt += db
	}
	if d > c {
		gt += dc
	}
	if d > d {
		gt += dd
	}

	return
}

func compares32_ssa(a, b, c, d float32) (lt, le, eq, ne, ge, gt uint64) {

	switch {
	}

	if a < a {
		lt += aa
	}
	if a < b {
		lt += ab
	}
	if a < c {
		lt += ac
	}
	if a < d {
		lt += ad
	}

	if b < a {
		lt += ba
	}
	if b < b {
		lt += bb
	}
	if b < c {
		lt += bc
	}
	if b < d {
		lt += bd
	}

	if c < a {
		lt += ca
	}
	if c < b {
		lt += cb
	}
	if c < c {
		lt += cc
	}
	if c < d {
		lt += cd
	}

	if d < a {
		lt += da
	}
	if d < b {
		lt += db
	}
	if d < c {
		lt += dc
	}
	if d < d {
		lt += dd
	}

	if a <= a {
		le += aa
	}
	if a <= b {
		le += ab
	}
	if a <= c {
		le += ac
	}
	if a <= d {
		le += ad
	}

	if b <= a {
		le += ba
	}
	if b <= b {
		le += bb
	}
	if b <= c {
		le += bc
	}
	if b <= d {
		le += bd
	}

	if c <= a {
		le += ca
	}
	if c <= b {
		le += cb
	}
	if c <= c {
		le += cc
	}
	if c <= d {
		le += cd
	}

	if d <= a {
		le += da
	}
	if d <= b {
		le += db
	}
	if d <= c {
		le += dc
	}
	if d <= d {
		le += dd
	}

	if a == a {
		eq += aa
	}
	if a == b {
		eq += ab
	}
	if a == c {
		eq += ac
	}
	if a == d {
		eq += ad
	}

	if b == a {
		eq += ba
	}
	if b == b {
		eq += bb
	}
	if b == c {
		eq += bc
	}
	if b == d {
		eq += bd
	}

	if c == a {
		eq += ca
	}
	if c == b {
		eq += cb
	}
	if c == c {
		eq += cc
	}
	if c == d {
		eq += cd
	}

	if d == a {
		eq += da
	}
	if d == b {
		eq += db
	}
	if d == c {
		eq += dc
	}
	if d == d {
		eq += dd
	}

	if a != a {
		ne += aa
	}
	if a != b {
		ne += ab
	}
	if a != c {
		ne += ac
	}
	if a != d {
		ne += ad
	}

	if b != a {
		ne += ba
	}
	if b != b {
		ne += bb
	}
	if b != c {
		ne += bc
	}
	if b != d {
		ne += bd
	}

	if c != a {
		ne += ca
	}
	if c != b {
		ne += cb
	}
	if c != c {
		ne += cc
	}
	if c != d {
		ne += cd
	}

	if d != a {
		ne += da
	}
	if d != b {
		ne += db
	}
	if d != c {
		ne += dc
	}
	if d != d {
		ne += dd
	}

	if a >= a {
		ge += aa
	}
	if a >= b {
		ge += ab
	}
	if a >= c {
		ge += ac
	}
	if a >= d {
		ge += ad
	}

	if b >= a {
		ge += ba
	}
	if b >= b {
		ge += bb
	}
	if b >= c {
		ge += bc
	}
	if b >= d {
		ge += bd
	}

	if c >= a {
		ge += ca
	}
	if c >= b {
		ge += cb
	}
	if c >= c {
		ge += cc
	}
	if c >= d {
		ge += cd
	}

	if d >= a {
		ge += da
	}
	if d >= b {
		ge += db
	}
	if d >= c {
		ge += dc
	}
	if d >= d {
		ge += dd
	}

	if a > a {
		gt += aa
	}
	if a > b {
		gt += ab
	}
	if a > c {
		gt += ac
	}
	if a > d {
		gt += ad
	}

	if b > a {
		gt += ba
	}
	if b > b {
		gt += bb
	}
	if b > c {
		gt += bc
	}
	if b > d {
		gt += bd
	}

	if c > a {
		gt += ca
	}
	if c > b {
		gt += cb
	}
	if c > c {
		gt += cc
	}
	if c > d {
		gt += cd
	}

	if d > a {
		gt += da
	}
	if d > b {
		gt += db
	}
	if d > c {
		gt += dc
	}
	if d > d {
		gt += dd
	}

	return
}

func le64_ssa(x, y float64) bool {
	switch {
	}
	return x <= y
}
func ge64_ssa(x, y float64) bool {
	switch {
	}
	return x >= y
}
func lt64_ssa(x, y float64) bool {
	switch {
	}
	return x < y
}
func gt64_ssa(x, y float64) bool {
	switch {
	}
	return x > y
}
func eq64_ssa(x, y float64) bool {
	switch {
	}
	return x == y
}
func ne64_ssa(x, y float64) bool {
	switch {
	}
	return x != y
}

func eqbr64_ssa(x, y float64) float64 {
	switch {
	}
	if x == y {
		return 17
	}
	return 42
}
func nebr64_ssa(x, y float64) float64 {
	switch {
	}
	if x != y {
		return 17
	}
	return 42
}
func gebr64_ssa(x, y float64) float64 {
	switch {
	}
	if x >= y {
		return 17
	}
	return 42
}
func lebr64_ssa(x, y float64) float64 {
	switch {
	}
	if x <= y {
		return 17
	}
	return 42
}
func ltbr64_ssa(x, y float64) float64 {
	switch {
	}
	if x < y {
		return 17
	}
	return 42
}
func gtbr64_ssa(x, y float64) float64 {
	switch {
	}
	if x > y {
		return 17
	}
	return 42
}

func le32_ssa(x, y float32) bool {
	switch {
	}
	return x <= y
}
func ge32_ssa(x, y float32) bool {
	switch {
	}
	return x >= y
}
func lt32_ssa(x, y float32) bool {
	switch {
	}
	return x < y
}
func gt32_ssa(x, y float32) bool {
	switch {
	}
	return x > y
}
func eq32_ssa(x, y float32) bool {
	switch {
	}
	return x == y
}
func ne32_ssa(x, y float32) bool {
	switch {
	}
	return x != y
}

func eqbr32_ssa(x, y float32) float32 {
	switch {
	}
	if x == y {
		return 17
	}
	return 42
}
func nebr32_ssa(x, y float32) float32 {
	switch {
	}
	if x != y {
		return 17
	}
	return 42
}
func gebr32_ssa(x, y float32) float32 {
	switch {
	}
	if x >= y {
		return 17
	}
	return 42
}
func lebr32_ssa(x, y float32) float32 {
	switch {
	}
	if x <= y {
		return 17
	}
	return 42
}
func ltbr32_ssa(x, y float32) float32 {
	switch {
	}
	if x < y {
		return 17
	}
	return 42
}
func gtbr32_ssa(x, y float32) float32 {
	switch {
	}
	if x > y {
		return 17
	}
	return 42
}

func fail64(s string, f func(a, b float64) float64, a, b, e float64) int {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float64) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
		return 1
	}
	return 0
}

func fail64bool(s string, f func(a, b float64) bool, a, b float64, e bool) int {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float64) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
		return 1
	}
	return 0
}

func fail32(s string, f func(a, b float32) float32, a, b, e float32) int {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float32) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
		return 1
	}
	return 0
}

func fail32bool(s string, f func(a, b float32) bool, a, b float32, e bool) int {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float32) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
		return 1
	}
	return 0
}

func expect64(s string, x, expected float64) int {
	if x != expected {
		println("Expected", expected, "for", s, ", got", x)
		return 1
	}
	return 0
}

func expect32(s string, x, expected float32) int {
	if x != expected {
		println("Expected", expected, "for", s, ", got", x)
		return 1
	}
	return 0
}

func expectUint64(s string, x, expected uint64) int {
	if x != expected {
		fmt.Printf("Expected 0x%016x for %s, got 0x%016x\n", expected, s, x)
		return 1
	}
	return 0
}

func expectAll64(s string, expected, a, b, c, d, e, f, g, h, i float64) int {
	fails := 0
	fails += expect64(s+":a", a, expected)
	fails += expect64(s+":b", b, expected)
	fails += expect64(s+":c", c, expected)
	fails += expect64(s+":d", d, expected)
	fails += expect64(s+":e", e, expected)
	fails += expect64(s+":f", f, expected)
	fails += expect64(s+":g", g, expected)
	return fails
}

func expectAll32(s string, expected, a, b, c, d, e, f, g, h, i float32) int {
	fails := 0
	fails += expect32(s+":a", a, expected)
	fails += expect32(s+":b", b, expected)
	fails += expect32(s+":c", c, expected)
	fails += expect32(s+":d", d, expected)
	fails += expect32(s+":e", e, expected)
	fails += expect32(s+":f", f, expected)
	fails += expect32(s+":g", g, expected)
	return fails
}

var ev64 [2]float64 = [2]float64{42.0, 17.0}
var ev32 [2]float32 = [2]float32{42.0, 17.0}

func cmpOpTest(s string,
	f func(a, b float64) bool,
	g func(a, b float64) float64,
	ff func(a, b float32) bool,
	gg func(a, b float32) float32,
	zero, one, inf, nan float64, result uint) int {
	fails := 0
	fails += fail64bool(s, f, zero, zero, result>>16&1 == 1)
	fails += fail64bool(s, f, zero, one, result>>12&1 == 1)
	fails += fail64bool(s, f, zero, inf, result>>8&1 == 1)
	fails += fail64bool(s, f, zero, nan, result>>4&1 == 1)
	fails += fail64bool(s, f, nan, nan, result&1 == 1)

	fails += fail64(s, g, zero, zero, ev64[result>>16&1])
	fails += fail64(s, g, zero, one, ev64[result>>12&1])
	fails += fail64(s, g, zero, inf, ev64[result>>8&1])
	fails += fail64(s, g, zero, nan, ev64[result>>4&1])
	fails += fail64(s, g, nan, nan, ev64[result>>0&1])

	{
		zero := float32(zero)
		one := float32(one)
		inf := float32(inf)
		nan := float32(nan)
		fails += fail32bool(s, ff, zero, zero, (result>>16)&1 == 1)
		fails += fail32bool(s, ff, zero, one, (result>>12)&1 == 1)
		fails += fail32bool(s, ff, zero, inf, (result>>8)&1 == 1)
		fails += fail32bool(s, ff, zero, nan, (result>>4)&1 == 1)
		fails += fail32bool(s, ff, nan, nan, result&1 == 1)

		fails += fail32(s, gg, zero, zero, ev32[(result>>16)&1])
		fails += fail32(s, gg, zero, one, ev32[(result>>12)&1])
		fails += fail32(s, gg, zero, inf, ev32[(result>>8)&1])
		fails += fail32(s, gg, zero, nan, ev32[(result>>4)&1])
		fails += fail32(s, gg, nan, nan, ev32[(result>>0)&1])
	}

	return fails
}

func main() {

	a := 3.0
	b := 4.0

	c := float32(3.0)
	d := float32(4.0)

	tiny := float32(1.5E-45) // smallest f32 denorm = 2**(-149)
	dtiny := float64(tiny)   // well within range of f64

	fails := 0
	fails += fail64("+", add64_ssa, a, b, 7.0)
	fails += fail64("*", mul64_ssa, a, b, 12.0)
	fails += fail64("-", sub64_ssa, a, b, -1.0)
	fails += fail64("/", div64_ssa, a, b, 0.75)

	fails += fail32("+", add32_ssa, c, d, 7.0)
	fails += fail32("*", mul32_ssa, c, d, 12.0)
	fails += fail32("-", sub32_ssa, c, d, -1.0)
	fails += fail32("/", div32_ssa, c, d, 0.75)

	// denorm-squared should underflow to zero.
	fails += fail32("*", mul32_ssa, tiny, tiny, 0)

	// but should not underflow in float and in fact is exactly representable.
	fails += fail64("*", mul64_ssa, dtiny, dtiny, 1.9636373861190906e-90)

	// Intended to create register pressure which forces
	// asymmetric op into different code paths.
	aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd := manysub_ssa(1000.0, 100.0, 10.0, 1.0)

	fails += expect64("aa", aa, 11.0)
	fails += expect64("ab", ab, 900.0)
	fails += expect64("ac", ac, 990.0)
	fails += expect64("ad", ad, 999.0)

	fails += expect64("ba", ba, -900.0)
	fails += expect64("bb", bb, 22.0)
	fails += expect64("bc", bc, 90.0)
	fails += expect64("bd", bd, 99.0)

	fails += expect64("ca", ca, -990.0)
	fails += expect64("cb", cb, -90.0)
	fails += expect64("cc", cc, 33.0)
	fails += expect64("cd", cd, 9.0)

	fails += expect64("da", da, -999.0)
	fails += expect64("db", db, -99.0)
	fails += expect64("dc", dc, -9.0)
	fails += expect64("dd", dd, 44.0)

	fails += integer2floatConversions()

	var zero64 float64 = 0.0
	var one64 float64 = 1.0
	var inf64 float64 = 1.0 / zero64
	var nan64 float64 = sub64_ssa(inf64, inf64)

	fails += cmpOpTest("!=", ne64_ssa, nebr64_ssa, ne32_ssa, nebr32_ssa, zero64, one64, inf64, nan64, 0x01111)
	fails += cmpOpTest("==", eq64_ssa, eqbr64_ssa, eq32_ssa, eqbr32_ssa, zero64, one64, inf64, nan64, 0x10000)
	fails += cmpOpTest("<=", le64_ssa, lebr64_ssa, le32_ssa, lebr32_ssa, zero64, one64, inf64, nan64, 0x11100)
	fails += cmpOpTest("<", lt64_ssa, ltbr64_ssa, lt32_ssa, ltbr32_ssa, zero64, one64, inf64, nan64, 0x01100)
	fails += cmpOpTest(">", gt64_ssa, gtbr64_ssa, gt32_ssa, gtbr32_ssa, zero64, one64, inf64, nan64, 0x00000)
	fails += cmpOpTest(">=", ge64_ssa, gebr64_ssa, ge32_ssa, gebr32_ssa, zero64, one64, inf64, nan64, 0x10000)

	{
		lt, le, eq, ne, ge, gt := compares64_ssa(0.0, 1.0, inf64, nan64)
		fails += expectUint64("lt", lt, 0x0110001000000000)
		fails += expectUint64("le", le, 0x1110011000100000)
		fails += expectUint64("eq", eq, 0x1000010000100000)
		fails += expectUint64("ne", ne, 0x0111101111011111)
		fails += expectUint64("ge", ge, 0x1000110011100000)
		fails += expectUint64("gt", gt, 0x0000100011000000)
		// fmt.Printf("lt=0x%016x, le=0x%016x, eq=0x%016x, ne=0x%016x, ge=0x%016x, gt=0x%016x\n",
		// 	lt, le, eq, ne, ge, gt)
	}
	{
		lt, le, eq, ne, ge, gt := compares32_ssa(0.0, 1.0, float32(inf64), float32(nan64))
		fails += expectUint64("lt", lt, 0x0110001000000000)
		fails += expectUint64("le", le, 0x1110011000100000)
		fails += expectUint64("eq", eq, 0x1000010000100000)
		fails += expectUint64("ne", ne, 0x0111101111011111)
		fails += expectUint64("ge", ge, 0x1000110011100000)
		fails += expectUint64("gt", gt, 0x0000100011000000)
	}

	if fails > 0 {
		fmt.Printf("Saw %v failures\n", fails)
		panic("Failed.")
	}
}
