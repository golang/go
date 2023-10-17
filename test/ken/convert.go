// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, near-exhaustive, of converting numbers between types.
// No complex numbers though.

package main

var i8 int8;
var u8 uint8;
var i16 int16;
var u16 uint16;
var i32 int32;
var u32 uint32;
var i64 int64;
var u64 uint64;
var f32 float32;
var f64 float64;

type	big	float64

type	t	struct {
	from, to	int
	val		big
}

const (
	ti8	= iota+1
	tu8
	ti16
	tu16
	ti32
	tu32
	ti64
	tu64
	tf32
	tf64
)

var	x = []t{

	/* value good in all types (10) */
	{ ti8,  ti8,  10 }, { ti8,  tu8,  10 }, { ti8,  ti16, 10 }, { ti8,  tu16, 10 },
	{ ti8,  ti32, 10 }, { ti8,  tu32, 10 }, { ti8,  ti64, 10 }, { ti8,  tu64, 10 },
	{ ti8,  tf32, 10 }, { ti8,  tf64, 10 },

	{ tu8,  ti8,  10 }, { tu8,  tu8,  10 }, { tu8,  ti16, 10 }, { tu8,  tu16, 10 },
	{ tu8,  ti32, 10 }, { tu8,  tu32, 10 }, { tu8,  ti64, 10 }, { tu8,  tu64, 10 },
	{ tu8,  tf32, 10 }, { tu8,  tf64, 10 },

	{ ti16, ti8,  10 }, { ti16, tu8,  10 }, { ti16, ti16, 10 }, { ti16, tu16, 10 },
	{ ti16, ti32, 10 }, { ti16, tu32, 10 }, { ti16, ti64, 10 }, { ti16, tu64, 10 },
	{ ti16, tf32, 10 }, { ti16, tf64, 10 },

	{ tu16, ti8,  10 }, { tu16, tu8,  10 }, { tu16, ti16, 10 }, { tu16, tu16, 10 },
	{ tu16, ti32, 10 }, { tu16, tu32, 10 }, { tu16, ti64, 10 }, { tu16, tu64, 10 },
	{ tu16, tf32, 10 }, { tu16, tf64, 10 },

	{ ti32, ti8,  10 }, { ti32, tu8,  10 }, { ti32, ti16, 10 }, { ti32, tu16, 10 },
	{ ti32, ti32, 10 }, { ti32, tu32, 10 }, { ti32, ti64, 10 }, { ti32, tu64, 10 },
	{ ti32, tf32, 10 }, { ti32, tf64, 10 },

	{ tu32, ti8,  10 }, { tu32, tu8,  10 }, { tu32, ti16, 10 }, { tu32, tu16, 10 },
	{ tu32, ti32, 10 }, { tu32, tu32, 10 }, { tu32, ti64, 10 }, { tu32, tu64, 10 },
	{ tu32, tf32, 10 }, { tu32, tf64, 10 },

	{ ti64, ti8,  10 }, { ti64, tu8,  10 }, { ti64, ti16, 10 }, { ti64, tu16, 10 },
	{ ti64, ti32, 10 }, { ti64, tu32, 10 }, { ti64, ti64, 10 }, { ti64, tu64, 10 },
	{ ti64, tf32, 10 }, { ti64, tf64, 10 },

	{ tu64, ti8,  10 }, { tu64, tu8,  10 }, { tu64, ti16, 10 }, { tu64, tu16, 10 },
	{ tu64, ti32, 10 }, { tu64, tu32, 10 }, { tu64, ti64, 10 }, { tu64, tu64, 10 },
	{ tu64, tf32, 10 }, { tu64, tf64, 10 },

	{ tf32, ti8,  10 }, { tf32, tu8,  10 }, { tf32, ti16, 10 }, { tf32, tu16, 10 },
	{ tf32, ti32, 10 }, { tf32, tu32, 10 }, { tf32, ti64, 10 }, { tf32, tu64, 10 },
	{ tf32, tf32, 10 }, { tf32, tf64, 10 },

	{ tf64, ti8,  10 }, { tf64, tu8,  10 }, { tf64, ti16, 10 }, { tf64, tu16, 10 },
	{ tf64, ti32, 10 }, { tf64, tu32, 10 }, { tf64, ti64, 10 }, { tf64, tu64, 10 },
	{ tf64, tf32, 10 }, { tf64, tf64, 10 },

	/* value good in all signed types (-4) */
	{ ti8,  ti8,  -4 }, { ti8,  ti16, -4 },
	{ ti8,  ti32, -4 }, { ti8,  ti64, -4 },
	{ ti8,  tf32, -4 }, { ti8,  tf64, -4 },

	{ ti16, ti8,  -4 }, { ti16, ti16, -4 },
	{ ti16, ti32, -4 }, { ti16, ti64, -4 },
	{ ti16, tf32, -4 },

	{ ti32, ti8,  -4 }, { ti32, ti16, -4 },
	{ ti32, ti32, -4 }, { ti32, ti64, -4 },
	{ ti32, tf32, -4 }, { ti32, tf64, -4 },

	{ ti64, ti8,  -4 }, { ti64, ti16, -4 },
	{ ti64, ti32, -4 }, { ti64, ti64, -4 },
	{ ti64, tf32, -4 },

	{ tf32, ti8,  -4 }, { tf32, ti16, -4 },
	{ tf32, ti32, -4 }, { tf32, ti64, -4 },
	{ tf32, tf32, -4 },

	{ tf64, ti8,  -4 }, { tf64, ti16, -4 },
	{ tf64, ti32, -4 }, { tf64, ti64, -4 },
	{ tf64, tf32, -4 }, { tf64, tf64, -4 },

	/* value good in u8 and up (175) */
	{ tu8,  tu8,  175 }, { tu8,  ti16, 175 }, { tu8,  tu16, 175 },
	{ tu8,  ti32, 175 }, { tu8,  tu32, 175 }, { tu8,  ti64, 175 }, { tu8,  tu64, 175 },
	{ tu8,  tf32, 175 }, { tu8,  tf64, 175 },

	{ ti16, tu8,  175 }, { ti16, ti16, 175 }, { ti16, tu16, 175 },
	{ ti16, ti32, 175 }, { ti16, tu32, 175 }, { ti16, ti64, 175 }, { ti16, tu64, 175 },
	{ ti16, tf32, 175 }, { ti16, tf64, 175 },

	{ tu16, tu8,  175 }, { tu16, ti16, 175 }, { tu16, tu16, 175 },
	{ tu16, ti32, 175 }, { tu16, tu32, 175 }, { tu16, ti64, 175 }, { tu16, tu64, 175 },
	{ tu16, tf32, 175 }, { tu16, tf64, 175 },

	{ ti32, tu8,  175 }, { ti32, ti16, 175 }, { ti32, tu16, 175 },
	{ ti32, ti32, 175 }, { ti32, tu32, 175 }, { ti32, ti64, 175 }, { ti32, tu64, 175 },
	{ ti32, tf32, 175 }, { ti32, tf64, 175 },

	{ tu32, tu8,  175 }, { tu32, ti16, 175 }, { tu32, tu16, 175 },
	{ tu32, ti32, 175 }, { tu32, tu32, 175 }, { tu32, ti64, 175 }, { tu32, tu64, 175 },
	{ tu32, tf32, 175 }, { tu32, tf64, 175 },

	{ ti64, tu8,  175 }, { ti64, ti16, 175 }, { ti64, tu16, 175 },
	{ ti64, ti32, 175 }, { ti64, tu32, 175 }, { ti64, ti64, 175 }, { ti64, tu64, 175 },
	{ ti64, tf32, 175 }, { ti64, tf64, 175 },

	{ tu64, tu8,  175 }, { tu64, ti16, 175 }, { tu64, tu16, 175 },
	{ tu64, ti32, 175 }, { tu64, tu32, 175 }, { tu64, ti64, 175 }, { tu64, tu64, 175 },
	{ tu64, tf32, 175 }, { tu64, tf64, 175 },

	{ tf32, tu8,  175 }, { tf32, ti16, 175 }, { tf32, tu16, 175 },
	{ tf32, ti32, 175 }, { tf32, tu32, 175 }, { tf32, ti64, 175 }, { tf32, tu64, 175 },
	{ tf32, tf32, 175 }, { tf32, tf64, 175 },

	{ tf64, tu8,  175 }, { tf64, ti16, 175 }, { tf64, tu16, 175 },
	{ tf64, ti32, 175 }, { tf64, tu32, 175 }, { tf64, ti64, 175 }, { tf64, tu64, 175 },
	{ tf64, tf32, 175 }, { tf64, tf64, 175 },

	/* value good in u16 and up (41259) */
	{ tu16, tu16, 41259 },
	{ tu16, ti32, 41259 }, { tu16, ti64, 41259 }, { tu16, tu64, 41259 },
	{ tu16, tf32, 41259 }, { tu16, tf64, 41259 },

	{ ti32, tu16, 41259 },
	{ ti32, ti32, 41259 }, { ti32, tu32, 41259 }, { ti32, ti64, 41259 }, { ti32, tu64, 41259 },
	{ ti32, tf32, 41259 }, { ti32, tf64, 41259 },

	{ tu32, tu16, 41259 },
	{ tu32, ti32, 41259 }, { tu32, tu32, 41259 }, { tu32, ti64, 41259 }, { tu32, tu64, 41259 },
	{ tu32, tf32, 41259 }, { tu32, tf64, 41259 },

	{ ti64, tu16, 41259 },
	{ ti64, ti32, 41259 }, { ti64, tu32, 41259 }, { ti64, ti64, 41259 }, { ti64, tu64, 41259 },
	{ ti64, tf32, 41259 }, { ti64, tf64, 41259 },

	{ tu64, tu16, 41259 },
	{ tu64, ti32, 41259 }, { tu64, tu32, 41259 }, { tu64, ti64, 41259 }, { tu64, tu64, 41259 },
	{ tu64, tf32, 41259 }, { tu64, tf64, 41259 },

	{ tf32, tu16, 41259 },
	{ tf32, ti32, 41259 }, { tf32, tu32, 41259 }, { tf32, ti64, 41259 }, { tf32, tu64, 41259 },
	{ tf32, tf32, 41259 }, { tf32, tf64, 41259 },

	{ tf64, tu16, 41259 },
	{ tf64, ti32, 41259 }, { tf64, tu32, 41259 }, { tf64, ti64, 41259 }, { tf64, tu64, 41259 },
	{ tf64, tf32, 41259 }, { tf64, tf64, 41259 },

	/* value good in u32 and up (3758096384) */
	{ tu32, tu32, 3758096384 }, { tu32, ti64, 3758096384 }, { tu32, tu64, 3758096384 },
	{ tu32, tf32, 3758096384 }, { tu32, tf64, 3758096384 },

	{ ti64, tu32, 3758096384 }, { ti64, ti64, 3758096384 }, { ti64, tu64, 3758096384 },
	{ ti64, tf32, 3758096384 }, { ti64, tf64, 3758096384 },

	{ tu64, tu32, 3758096384 }, { tu64, ti64, 3758096384 }, { tu64, tu64, 3758096384 },
	{ tu64, tf32, 3758096384 }, { tu64, tf64, 3758096384 },

	{ tf32, tu32, 3758096384 }, { tf32, ti64, 3758096384 }, { tf32, tu64, 3758096384 },
	{ tf32, tf32, 3758096384 }, { tf32, tf64, 3758096384 },

	{ tf64, tu32, 3758096384 }, { tf64, ti64, 3758096384 }, { tf64, tu64, 3758096384 },
	{ tf64, tf32, 3758096384 }, { tf64, tf64, 3758096384 },

	/* value good in u64 and up (16717361816799281152) */
	{ tu64, tu64, 16717361816799281152 },
	{ tu64, tf32, 16717361816799281152 }, { tu64, tf64, 16717361816799281152 },

	{ tf32, tu64, 16717361816799281152 },
	{ tf32, tf32, 16717361816799281152 }, { tf32, tf64, 16717361816799281152 },

	{ tf64, tu64, 16717361816799281152 },
	{ tf64, tf32, 16717361816799281152 }, { tf64, tf64, 16717361816799281152 },
}

func main() {
	for i:=0; i<len(x); i++ {
		v := x[i].val		// input value
		w := big(0)		// output value
		f := x[i].from		// input type
		t := x[i].to		// output type

		i8  = 0; u8  = 0; i16 = 0; u16 = 0
		i32 = 0; u32 = 0; i64 = 0; u64 = 0
		f32 = 0; f64 = 0

		switch f*100 + t {
		default:
			println("missing case", i, v, f, t)
			w = v

		case ti8*100 + ti8:
			i8 = int8(v); i8 = int8(i8); w = big(i8)
		case ti8*100 + tu8:
			i8 = int8(v); u8 = uint8(i8); w = big(u8)
		case ti8*100 + ti16:
			i8 = int8(v); i16 = int16(i8); w = big(i16)
		case ti8*100 + tu16:
			i8 = int8(v); u16 = uint16(i8); w = big(u16)
		case ti8*100 + ti32:
			i8 = int8(v); i32 = int32(i8); w = big(i32)
		case ti8*100 + tu32:
			i8 = int8(v); u32 = uint32(i8); w = big(u32)
		case ti8*100 + ti64:
			i8 = int8(v); i64 = int64(i8); w = big(i64)
		case ti8*100 + tu64:
			i8 = int8(v); u64 = uint64(i8); w = big(u64)
		case ti8*100 + tf32:
			i8 = int8(v); f32 = float32(i8); w = big(f32)
		case ti8*100 + tf64:
			i8 = int8(v); f64 = float64(i8); w = big(f64)

		case tu8*100 + ti8:
			u8 = uint8(v); i8 = int8(u8); w = big(i8)
		case tu8*100 + tu8:
			u8 = uint8(v); u8 = uint8(u8); w = big(u8)
		case tu8*100 + ti16:
			u8 = uint8(v); i16 = int16(u8); w = big(i16)
		case tu8*100 + tu16:
			u8 = uint8(v); u16 = uint16(u8); w = big(u16)
		case tu8*100 + ti32:
			u8 = uint8(v); i32 = int32(u8); w = big(i32)
		case tu8*100 + tu32:
			u8 = uint8(v); u32 = uint32(u8); w = big(u32)
		case tu8*100 + ti64:
			u8 = uint8(v); i64 = int64(u8); w = big(i64)
		case tu8*100 + tu64:
			u8 = uint8(v); u64 = uint64(u8); w = big(u64)
		case tu8*100 + tf32:
			u8 = uint8(v); f32 = float32(u8); w = big(f32)
		case tu8*100 + tf64:
			u8 = uint8(v); f64 = float64(u8); w = big(f64)

		case ti16*100 + ti8:
			i16 = int16(v); i8 = int8(i16); w = big(i8)
		case ti16*100 + tu8:
			i16 = int16(v); u8 = uint8(i16); w = big(u8)
		case ti16*100 + ti16:
			i16 = int16(v); i16 = int16(i16); w = big(i16)
		case ti16*100 + tu16:
			i16 = int16(v); u16 = uint16(i16); w = big(u16)
		case ti16*100 + ti32:
			i16 = int16(v); i32 = int32(i16); w = big(i32)
		case ti16*100 + tu32:
			i16 = int16(v); u32 = uint32(i16); w = big(u32)
		case ti16*100 + ti64:
			i16 = int16(v); i64 = int64(i16); w = big(i64)
		case ti16*100 + tu64:
			i16 = int16(v); u64 = uint64(i16); w = big(u64)
		case ti16*100 + tf32:
			i16 = int16(v); f32 = float32(i16); w = big(f32)
		case ti16*100 + tf64:
			i16 = int16(v); f64 = float64(i16); w = big(f64)

		case tu16*100 + ti8:
			u16 = uint16(v); i8 = int8(u16); w = big(i8)
		case tu16*100 + tu8:
			u16 = uint16(v); u8 = uint8(u16); w = big(u8)
		case tu16*100 + ti16:
			u16 = uint16(v); i16 = int16(u16); w = big(i16)
		case tu16*100 + tu16:
			u16 = uint16(v); u16 = uint16(u16); w = big(u16)
		case tu16*100 + ti32:
			u16 = uint16(v); i32 = int32(u16); w = big(i32)
		case tu16*100 + tu32:
			u16 = uint16(v); u32 = uint32(u16); w = big(u32)
		case tu16*100 + ti64:
			u16 = uint16(v); i64 = int64(u16); w = big(i64)
		case tu16*100 + tu64:
			u16 = uint16(v); u64 = uint64(u16); w = big(u64)
		case tu16*100 + tf32:
			u16 = uint16(v); f32 = float32(u16); w = big(f32)
		case tu16*100 + tf64:
			u16 = uint16(v); f64 = float64(u16); w = big(f64)

		case ti32*100 + ti8:
			i32 = int32(v); i8 = int8(i32); w = big(i8)
		case ti32*100 + tu8:
			i32 = int32(v); u8 = uint8(i32); w = big(u8)
		case ti32*100 + ti16:
			i32 = int32(v); i16 = int16(i32); w = big(i16)
		case ti32*100 + tu16:
			i32 = int32(v); u16 = uint16(i32); w = big(u16)
		case ti32*100 + ti32:
			i32 = int32(v); i32 = int32(i32); w = big(i32)
		case ti32*100 + tu32:
			i32 = int32(v); u32 = uint32(i32); w = big(u32)
		case ti32*100 + ti64:
			i32 = int32(v); i64 = int64(i32); w = big(i64)
		case ti32*100 + tu64:
			i32 = int32(v); u64 = uint64(i32); w = big(u64)
		case ti32*100 + tf32:
			i32 = int32(v); f32 = float32(i32); w = big(f32)
		case ti32*100 + tf64:
			i32 = int32(v); f64 = float64(i32); w = big(f64)

		case tu32*100 + ti8:
			u32 = uint32(v); i8 = int8(u32); w = big(i8)
		case tu32*100 + tu8:
			u32 = uint32(v); u8 = uint8(u32); w = big(u8)
		case tu32*100 + ti16:
			u32 = uint32(v); i16 = int16(u32); w = big(i16)
		case tu32*100 + tu16:
			u32 = uint32(v); u16 = uint16(u32); w = big(u16)
		case tu32*100 + ti32:
			u32 = uint32(v); i32 = int32(u32); w = big(i32)
		case tu32*100 + tu32:
			u32 = uint32(v); u32 = uint32(u32); w = big(u32)
		case tu32*100 + ti64:
			u32 = uint32(v); i64 = int64(u32); w = big(i64)
		case tu32*100 + tu64:
			u32 = uint32(v); u64 = uint64(u32); w = big(u64)
		case tu32*100 + tf32:
			u32 = uint32(v); f32 = float32(u32); w = big(f32)
		case tu32*100 + tf64:
			u32 = uint32(v); f64 = float64(u32); w = big(f64)

		case ti64*100 + ti8:
			i64 = int64(v); i8 = int8(i64); w = big(i8)
		case ti64*100 + tu8:
			i64 = int64(v); u8 = uint8(i64); w = big(u8)
		case ti64*100 + ti16:
			i64 = int64(v); i16 = int16(i64); w = big(i16)
		case ti64*100 + tu16:
			i64 = int64(v); u16 = uint16(i64); w = big(u16)
		case ti64*100 + ti32:
			i64 = int64(v); i32 = int32(i64); w = big(i32)
		case ti64*100 + tu32:
			i64 = int64(v); u32 = uint32(i64); w = big(u32)
		case ti64*100 + ti64:
			i64 = int64(v); i64 = int64(i64); w = big(i64)
		case ti64*100 + tu64:
			i64 = int64(v); u64 = uint64(i64); w = big(u64)
		case ti64*100 + tf32:
			i64 = int64(v); f32 = float32(i64); w = big(f32)
		case ti64*100 + tf64:
			i64 = int64(v); f64 = float64(i64); w = big(f64)

		case tu64*100 + ti8:
			u64 = uint64(v); i8 = int8(u64); w = big(i8)
		case tu64*100 + tu8:
			u64 = uint64(v); u8 = uint8(u64); w = big(u8)
		case tu64*100 + ti16:
			u64 = uint64(v); i16 = int16(u64); w = big(i16)
		case tu64*100 + tu16:
			u64 = uint64(v); u16 = uint16(u64); w = big(u16)
		case tu64*100 + ti32:
			u64 = uint64(v); i32 = int32(u64); w = big(i32)
		case tu64*100 + tu32:
			u64 = uint64(v); u32 = uint32(u64); w = big(u32)
		case tu64*100 + ti64:
			u64 = uint64(v); i64 = int64(u64); w = big(i64)
		case tu64*100 + tu64:
			u64 = uint64(v); u64 = uint64(u64); w = big(u64)
		case tu64*100 + tf32:
			u64 = uint64(v); f32 = float32(u64); w = big(f32)
		case tu64*100 + tf64:
			u64 = uint64(v); f64 = float64(u64); w = big(f64)

		case tf32*100 + ti8:
			f32 = float32(v); i8 = int8(f32); w = big(i8)
		case tf32*100 + tu8:
			f32 = float32(v); u8 = uint8(f32); w = big(u8)
		case tf32*100 + ti16:
			f32 = float32(v); i16 = int16(f32); w = big(i16)
		case tf32*100 + tu16:
			f32 = float32(v); u16 = uint16(f32); w = big(u16)
		case tf32*100 + ti32:
			f32 = float32(v); i32 = int32(f32); w = big(i32)
		case tf32*100 + tu32:
			f32 = float32(v); u32 = uint32(f32); w = big(u32)
		case tf32*100 + ti64:
			f32 = float32(v); i64 = int64(f32); w = big(i64)
		case tf32*100 + tu64:
			f32 = float32(v); u64 = uint64(f32); w = big(u64)
		case tf32*100 + tf32:
			f32 = float32(v); f32 = float32(f32); w = big(f32)
		case tf32*100 + tf64:
			f32 = float32(v); f64 = float64(f32); w = big(f64)

		case tf64*100 + ti8:
			f64 = float64(v); i8 = int8(f64); w = big(i8)
		case tf64*100 + tu8:
			f64 = float64(v); u8 = uint8(f64); w = big(u8)
		case tf64*100 + ti16:
			f64 = float64(v); i16 = int16(f64); w = big(i16)
		case tf64*100 + tu16:
			f64 = float64(v); u16 = uint16(f64); w = big(u16)
		case tf64*100 + ti32:
			f64 = float64(v); i32 = int32(f64); w = big(i32)
		case tf64*100 + tu32:
			f64 = float64(v); u32 = uint32(f64); w = big(u32)
		case tf64*100 + ti64:
			f64 = float64(v); i64 = int64(f64); w = big(i64)
		case tf64*100 + tu64:
			f64 = float64(v); u64 = uint64(f64); w = big(u64)
		case tf64*100 + tf32:
			f64 = float64(v); f32 = float32(f64); w = big(f32)
		case tf64*100 + tf64:
			f64 = float64(v); f64 = float64(f64); w = big(f64)
		}
		if v != w { println(i, v, w, f, t) }
	}
}
