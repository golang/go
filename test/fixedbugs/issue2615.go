// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2615: a long chain of else if's causes an overflow
// in the parser stack.

package main

// test returns the index of the lowest set bit in a 256-bit vector.
func test(x [4]uint64) int {
	if x[0]&(1<<0) != 0 {
		return 0
	} else if x[0]&(1<<1) != 0 {
		return 1
	} else if x[0]&(1<<2) != 0 {
		return 2
	} else if x[0]&(1<<3) != 0 {
		return 3
	} else if x[0]&(1<<4) != 0 {
		return 4
	} else if x[0]&(1<<5) != 0 {
		return 5
	} else if x[0]&(1<<6) != 0 {
		return 6
	} else if x[0]&(1<<7) != 0 {
		return 7
	} else if x[0]&(1<<8) != 0 {
		return 8
	} else if x[0]&(1<<9) != 0 {
		return 9
	} else if x[0]&(1<<10) != 0 {
		return 10
	} else if x[0]&(1<<11) != 0 {
		return 11
	} else if x[0]&(1<<12) != 0 {
		return 12
	} else if x[0]&(1<<13) != 0 {
		return 13
	} else if x[0]&(1<<14) != 0 {
		return 14
	} else if x[0]&(1<<15) != 0 {
		return 15
	} else if x[0]&(1<<16) != 0 {
		return 16
	} else if x[0]&(1<<17) != 0 {
		return 17
	} else if x[0]&(1<<18) != 0 {
		return 18
	} else if x[0]&(1<<19) != 0 {
		return 19
	} else if x[0]&(1<<20) != 0 {
		return 20
	} else if x[0]&(1<<21) != 0 {
		return 21
	} else if x[0]&(1<<22) != 0 {
		return 22
	} else if x[0]&(1<<23) != 0 {
		return 23
	} else if x[0]&(1<<24) != 0 {
		return 24
	} else if x[0]&(1<<25) != 0 {
		return 25
	} else if x[0]&(1<<26) != 0 {
		return 26
	} else if x[0]&(1<<27) != 0 {
		return 27
	} else if x[0]&(1<<28) != 0 {
		return 28
	} else if x[0]&(1<<29) != 0 {
		return 29
	} else if x[0]&(1<<30) != 0 {
		return 30
	} else if x[0]&(1<<31) != 0 {
		return 31
	} else if x[0]&(1<<32) != 0 {
		return 32
	} else if x[0]&(1<<33) != 0 {
		return 33
	} else if x[0]&(1<<34) != 0 {
		return 34
	} else if x[0]&(1<<35) != 0 {
		return 35
	} else if x[0]&(1<<36) != 0 {
		return 36
	} else if x[0]&(1<<37) != 0 {
		return 37
	} else if x[0]&(1<<38) != 0 {
		return 38
	} else if x[0]&(1<<39) != 0 {
		return 39
	} else if x[0]&(1<<40) != 0 {
		return 40
	} else if x[0]&(1<<41) != 0 {
		return 41
	} else if x[0]&(1<<42) != 0 {
		return 42
	} else if x[0]&(1<<43) != 0 {
		return 43
	} else if x[0]&(1<<44) != 0 {
		return 44
	} else if x[0]&(1<<45) != 0 {
		return 45
	} else if x[0]&(1<<46) != 0 {
		return 46
	} else if x[0]&(1<<47) != 0 {
		return 47
	} else if x[0]&(1<<48) != 0 {
		return 48
	} else if x[0]&(1<<49) != 0 {
		return 49
	} else if x[0]&(1<<50) != 0 {
		return 50
	} else if x[0]&(1<<51) != 0 {
		return 51
	} else if x[0]&(1<<52) != 0 {
		return 52
	} else if x[0]&(1<<53) != 0 {
		return 53
	} else if x[0]&(1<<54) != 0 {
		return 54
	} else if x[0]&(1<<55) != 0 {
		return 55
	} else if x[0]&(1<<56) != 0 {
		return 56
	} else if x[0]&(1<<57) != 0 {
		return 57
	} else if x[0]&(1<<58) != 0 {
		return 58
	} else if x[0]&(1<<59) != 0 {
		return 59
	} else if x[0]&(1<<60) != 0 {
		return 60
	} else if x[0]&(1<<61) != 0 {
		return 61
	} else if x[0]&(1<<62) != 0 {
		return 62
	} else if x[0]&(1<<63) != 0 {
		return 63
	} else if x[1]&(1<<0) != 0 {
		return 64
	} else if x[1]&(1<<1) != 0 {
		return 65
	} else if x[1]&(1<<2) != 0 {
		return 66
	} else if x[1]&(1<<3) != 0 {
		return 67
	} else if x[1]&(1<<4) != 0 {
		return 68
	} else if x[1]&(1<<5) != 0 {
		return 69
	} else if x[1]&(1<<6) != 0 {
		return 70
	} else if x[1]&(1<<7) != 0 {
		return 71
	} else if x[1]&(1<<8) != 0 {
		return 72
	} else if x[1]&(1<<9) != 0 {
		return 73
	} else if x[1]&(1<<10) != 0 {
		return 74
	} else if x[1]&(1<<11) != 0 {
		return 75
	} else if x[1]&(1<<12) != 0 {
		return 76
	} else if x[1]&(1<<13) != 0 {
		return 77
	} else if x[1]&(1<<14) != 0 {
		return 78
	} else if x[1]&(1<<15) != 0 {
		return 79
	} else if x[1]&(1<<16) != 0 {
		return 80
	} else if x[1]&(1<<17) != 0 {
		return 81
	} else if x[1]&(1<<18) != 0 {
		return 82
	} else if x[1]&(1<<19) != 0 {
		return 83
	} else if x[1]&(1<<20) != 0 {
		return 84
	} else if x[1]&(1<<21) != 0 {
		return 85
	} else if x[1]&(1<<22) != 0 {
		return 86
	} else if x[1]&(1<<23) != 0 {
		return 87
	} else if x[1]&(1<<24) != 0 {
		return 88
	} else if x[1]&(1<<25) != 0 {
		return 89
	} else if x[1]&(1<<26) != 0 {
		return 90
	} else if x[1]&(1<<27) != 0 {
		return 91
	} else if x[1]&(1<<28) != 0 {
		return 92
	} else if x[1]&(1<<29) != 0 {
		return 93
	} else if x[1]&(1<<30) != 0 {
		return 94
	} else if x[1]&(1<<31) != 0 {
		return 95
	} else if x[1]&(1<<32) != 0 {
		return 96
	} else if x[1]&(1<<33) != 0 {
		return 97
	} else if x[1]&(1<<34) != 0 {
		return 98
	} else if x[1]&(1<<35) != 0 {
		return 99
	} else if x[1]&(1<<36) != 0 {
		return 100
	} else if x[1]&(1<<37) != 0 {
		return 101
	} else if x[1]&(1<<38) != 0 {
		return 102
	} else if x[1]&(1<<39) != 0 {
		return 103
	} else if x[1]&(1<<40) != 0 {
		return 104
	} else if x[1]&(1<<41) != 0 {
		return 105
	} else if x[1]&(1<<42) != 0 {
		return 106
	} else if x[1]&(1<<43) != 0 {
		return 107
	} else if x[1]&(1<<44) != 0 {
		return 108
	} else if x[1]&(1<<45) != 0 {
		return 109
	} else if x[1]&(1<<46) != 0 {
		return 110
	} else if x[1]&(1<<47) != 0 {
		return 111
	} else if x[1]&(1<<48) != 0 {
		return 112
	} else if x[1]&(1<<49) != 0 {
		return 113
	} else if x[1]&(1<<50) != 0 {
		return 114
	} else if x[1]&(1<<51) != 0 {
		return 115
	} else if x[1]&(1<<52) != 0 {
		return 116
	} else if x[1]&(1<<53) != 0 {
		return 117
	} else if x[1]&(1<<54) != 0 {
		return 118
	} else if x[1]&(1<<55) != 0 {
		return 119
	} else if x[1]&(1<<56) != 0 {
		return 120
	} else if x[1]&(1<<57) != 0 {
		return 121
	} else if x[1]&(1<<58) != 0 {
		return 122
	} else if x[1]&(1<<59) != 0 {
		return 123
	} else if x[1]&(1<<60) != 0 {
		return 124
	} else if x[1]&(1<<61) != 0 {
		return 125
	} else if x[1]&(1<<62) != 0 {
		return 126
	} else if x[1]&(1<<63) != 0 {
		return 127
	} else if x[2]&(1<<0) != 0 {
		return 128
	} else if x[2]&(1<<1) != 0 {
		return 129
	} else if x[2]&(1<<2) != 0 {
		return 130
	} else if x[2]&(1<<3) != 0 {
		return 131
	} else if x[2]&(1<<4) != 0 {
		return 132
	} else if x[2]&(1<<5) != 0 {
		return 133
	} else if x[2]&(1<<6) != 0 {
		return 134
	} else if x[2]&(1<<7) != 0 {
		return 135
	} else if x[2]&(1<<8) != 0 {
		return 136
	} else if x[2]&(1<<9) != 0 {
		return 137
	} else if x[2]&(1<<10) != 0 {
		return 138
	} else if x[2]&(1<<11) != 0 {
		return 139
	} else if x[2]&(1<<12) != 0 {
		return 140
	} else if x[2]&(1<<13) != 0 {
		return 141
	} else if x[2]&(1<<14) != 0 {
		return 142
	} else if x[2]&(1<<15) != 0 {
		return 143
	} else if x[2]&(1<<16) != 0 {
		return 144
	} else if x[2]&(1<<17) != 0 {
		return 145
	} else if x[2]&(1<<18) != 0 {
		return 146
	} else if x[2]&(1<<19) != 0 {
		return 147
	} else if x[2]&(1<<20) != 0 {
		return 148
	} else if x[2]&(1<<21) != 0 {
		return 149
	} else if x[2]&(1<<22) != 0 {
		return 150
	} else if x[2]&(1<<23) != 0 {
		return 151
	} else if x[2]&(1<<24) != 0 {
		return 152
	} else if x[2]&(1<<25) != 0 {
		return 153
	} else if x[2]&(1<<26) != 0 {
		return 154
	} else if x[2]&(1<<27) != 0 {
		return 155
	} else if x[2]&(1<<28) != 0 {
		return 156
	} else if x[2]&(1<<29) != 0 {
		return 157
	} else if x[2]&(1<<30) != 0 {
		return 158
	} else if x[2]&(1<<31) != 0 {
		return 159
	} else if x[2]&(1<<32) != 0 {
		return 160
	} else if x[2]&(1<<33) != 0 {
		return 161
	} else if x[2]&(1<<34) != 0 {
		return 162
	} else if x[2]&(1<<35) != 0 {
		return 163
	} else if x[2]&(1<<36) != 0 {
		return 164
	} else if x[2]&(1<<37) != 0 {
		return 165
	} else if x[2]&(1<<38) != 0 {
		return 166
	} else if x[2]&(1<<39) != 0 {
		return 167
	} else if x[2]&(1<<40) != 0 {
		return 168
	} else if x[2]&(1<<41) != 0 {
		return 169
	} else if x[2]&(1<<42) != 0 {
		return 170
	} else if x[2]&(1<<43) != 0 {
		return 171
	} else if x[2]&(1<<44) != 0 {
		return 172
	} else if x[2]&(1<<45) != 0 {
		return 173
	} else if x[2]&(1<<46) != 0 {
		return 174
	} else if x[2]&(1<<47) != 0 {
		return 175
	} else if x[2]&(1<<48) != 0 {
		return 176
	} else if x[2]&(1<<49) != 0 {
		return 177
	} else if x[2]&(1<<50) != 0 {
		return 178
	} else if x[2]&(1<<51) != 0 {
		return 179
	} else if x[2]&(1<<52) != 0 {
		return 180
	} else if x[2]&(1<<53) != 0 {
		return 181
	} else if x[2]&(1<<54) != 0 {
		return 182
	} else if x[2]&(1<<55) != 0 {
		return 183
	} else if x[2]&(1<<56) != 0 {
		return 184
	} else if x[2]&(1<<57) != 0 {
		return 185
	} else if x[2]&(1<<58) != 0 {
		return 186
	} else if x[2]&(1<<59) != 0 {
		return 187
	} else if x[2]&(1<<60) != 0 {
		return 188
	} else if x[2]&(1<<61) != 0 {
		return 189
	} else if x[2]&(1<<62) != 0 {
		return 190
	} else if x[2]&(1<<63) != 0 {
		return 191
	} else if x[3]&(1<<0) != 0 {
		return 192
	} else if x[3]&(1<<1) != 0 {
		return 193
	} else if x[3]&(1<<2) != 0 {
		return 194
	} else if x[3]&(1<<3) != 0 {
		return 195
	} else if x[3]&(1<<4) != 0 {
		return 196
	} else if x[3]&(1<<5) != 0 {
		return 197
	} else if x[3]&(1<<6) != 0 {
		return 198
	} else if x[3]&(1<<7) != 0 {
		return 199
	} else if x[3]&(1<<8) != 0 {
		return 200
	} else if x[3]&(1<<9) != 0 {
		return 201
	} else if x[3]&(1<<10) != 0 {
		return 202
	} else if x[3]&(1<<11) != 0 {
		return 203
	} else if x[3]&(1<<12) != 0 {
		return 204
	} else if x[3]&(1<<13) != 0 {
		return 205
	} else if x[3]&(1<<14) != 0 {
		return 206
	} else if x[3]&(1<<15) != 0 {
		return 207
	} else if x[3]&(1<<16) != 0 {
		return 208
	} else if x[3]&(1<<17) != 0 {
		return 209
	} else if x[3]&(1<<18) != 0 {
		return 210
	} else if x[3]&(1<<19) != 0 {
		return 211
	} else if x[3]&(1<<20) != 0 {
		return 212
	} else if x[3]&(1<<21) != 0 {
		return 213
	} else if x[3]&(1<<22) != 0 {
		return 214
	} else if x[3]&(1<<23) != 0 {
		return 215
	} else if x[3]&(1<<24) != 0 {
		return 216
	} else if x[3]&(1<<25) != 0 {
		return 217
	} else if x[3]&(1<<26) != 0 {
		return 218
	} else if x[3]&(1<<27) != 0 {
		return 219
	} else if x[3]&(1<<28) != 0 {
		return 220
	} else if x[3]&(1<<29) != 0 {
		return 221
	} else if x[3]&(1<<30) != 0 {
		return 222
	} else if x[3]&(1<<31) != 0 {
		return 223
	} else if x[3]&(1<<32) != 0 {
		return 224
	} else if x[3]&(1<<33) != 0 {
		return 225
	} else if x[3]&(1<<34) != 0 {
		return 226
	} else if x[3]&(1<<35) != 0 {
		return 227
	} else if x[3]&(1<<36) != 0 {
		return 228
	} else if x[3]&(1<<37) != 0 {
		return 229
	} else if x[3]&(1<<38) != 0 {
		return 230
	} else if x[3]&(1<<39) != 0 {
		return 231
	} else if x[3]&(1<<40) != 0 {
		return 232
	} else if x[3]&(1<<41) != 0 {
		return 233
	} else if x[3]&(1<<42) != 0 {
		return 234
	} else if x[3]&(1<<43) != 0 {
		return 235
	} else if x[3]&(1<<44) != 0 {
		return 236
	} else if x[3]&(1<<45) != 0 {
		return 237
	} else if x[3]&(1<<46) != 0 {
		return 238
	} else if x[3]&(1<<47) != 0 {
		return 239
	} else if x[3]&(1<<48) != 0 {
		return 240
	} else if x[3]&(1<<49) != 0 {
		return 241
	} else if x[3]&(1<<50) != 0 {
		return 242
	} else if x[3]&(1<<51) != 0 {
		return 243
	} else if x[3]&(1<<52) != 0 {
		return 244
	} else if x[3]&(1<<53) != 0 {
		return 245
	} else if x[3]&(1<<54) != 0 {
		return 246
	} else if x[3]&(1<<55) != 0 {
		return 247
	} else if x[3]&(1<<56) != 0 {
		return 248
	} else if x[3]&(1<<57) != 0 {
		return 249
	} else if x[3]&(1<<58) != 0 {
		return 250
	} else if x[3]&(1<<59) != 0 {
		return 251
	} else if x[3]&(1<<60) != 0 {
		return 252
	} else if x[3]&(1<<61) != 0 {
		return 253
	} else if x[3]&(1<<62) != 0 {
		return 254
	} else if x[3]&(1<<63) != 0 {
		return 255
	}
	return -1
}

func main() {
	const ones = ^uint64(0)
	for i := 0; i < 256; i++ {
		bits := [4]uint64{ones, ones, ones, ones}

		// clear bottom i bits
		bits[i/64] ^= 1<<(uint(i)&63) - 1
		for j := i/64 - 1; j >= 0; j-- {
			bits[j] = 0
		}

		k := test(bits)
		if k != i {
			print("test(bits)=", k, " want ", i, "\n")
			panic("failed")
		}
	}
}
