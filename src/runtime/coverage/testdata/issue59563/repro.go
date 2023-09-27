// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package repro

import (
	"fmt"
	"net/http"
)

func small() {
	go func() {
		fmt.Println(http.ListenAndServe("localhost:7070", nil))
	}()
}

func large(x int) int {
	if x == 0 {
		x += 0
	} else if x == 1 {
		x += 1
	} else if x == 2 {
		x += 2
	} else if x == 3 {
		x += 3
	} else if x == 4 {
		x += 4
	} else if x == 5 {
		x += 5
	} else if x == 6 {
		x += 6
	} else if x == 7 {
		x += 7
	} else if x == 8 {
		x += 8
	} else if x == 9 {
		x += 9
	} else if x == 10 {
		x += 10
	} else if x == 11 {
		x += 11
	} else if x == 12 {
		x += 12
	} else if x == 13 {
		x += 13
	} else if x == 14 {
		x += 14
	} else if x == 15 {
		x += 15
	} else if x == 16 {
		x += 16
	} else if x == 17 {
		x += 17
	} else if x == 18 {
		x += 18
	} else if x == 19 {
		x += 19
	} else if x == 20 {
		x += 20
	} else if x == 21 {
		x += 21
	} else if x == 22 {
		x += 22
	} else if x == 23 {
		x += 23
	} else if x == 24 {
		x += 24
	} else if x == 25 {
		x += 25
	} else if x == 26 {
		x += 26
	} else if x == 27 {
		x += 27
	} else if x == 28 {
		x += 28
	} else if x == 29 {
		x += 29
	} else if x == 30 {
		x += 30
	} else if x == 31 {
		x += 31
	} else if x == 32 {
		x += 32
	} else if x == 33 {
		x += 33
	} else if x == 34 {
		x += 34
	} else if x == 35 {
		x += 35
	} else if x == 36 {
		x += 36
	} else if x == 37 {
		x += 37
	} else if x == 38 {
		x += 38
	} else if x == 39 {
		x += 39
	} else if x == 40 {
		x += 40
	} else if x == 41 {
		x += 41
	} else if x == 42 {
		x += 42
	} else if x == 43 {
		x += 43
	} else if x == 44 {
		x += 44
	} else if x == 45 {
		x += 45
	} else if x == 46 {
		x += 46
	} else if x == 47 {
		x += 47
	} else if x == 48 {
		x += 48
	} else if x == 49 {
		x += 49
	} else if x == 50 {
		x += 50
	} else if x == 51 {
		x += 51
	} else if x == 52 {
		x += 52
	} else if x == 53 {
		x += 53
	} else if x == 54 {
		x += 54
	} else if x == 55 {
		x += 55
	} else if x == 56 {
		x += 56
	} else if x == 57 {
		x += 57
	} else if x == 58 {
		x += 58
	} else if x == 59 {
		x += 59
	} else if x == 60 {
		x += 60
	} else if x == 61 {
		x += 61
	} else if x == 62 {
		x += 62
	} else if x == 63 {
		x += 63
	} else if x == 64 {
		x += 64
	} else if x == 65 {
		x += 65
	} else if x == 66 {
		x += 66
	} else if x == 67 {
		x += 67
	} else if x == 68 {
		x += 68
	} else if x == 69 {
		x += 69
	} else if x == 70 {
		x += 70
	} else if x == 71 {
		x += 71
	} else if x == 72 {
		x += 72
	} else if x == 73 {
		x += 73
	} else if x == 74 {
		x += 74
	} else if x == 75 {
		x += 75
	} else if x == 76 {
		x += 76
	} else if x == 77 {
		x += 77
	} else if x == 78 {
		x += 78
	} else if x == 79 {
		x += 79
	} else if x == 80 {
		x += 80
	} else if x == 81 {
		x += 81
	} else if x == 82 {
		x += 82
	} else if x == 83 {
		x += 83
	} else if x == 84 {
		x += 84
	} else if x == 85 {
		x += 85
	} else if x == 86 {
		x += 86
	} else if x == 87 {
		x += 87
	} else if x == 88 {
		x += 88
	} else if x == 89 {
		x += 89
	} else if x == 90 {
		x += 90
	} else if x == 91 {
		x += 91
	} else if x == 92 {
		x += 92
	} else if x == 93 {
		x += 93
	} else if x == 94 {
		x += 94
	} else if x == 95 {
		x += 95
	} else if x == 96 {
		x += 96
	} else if x == 97 {
		x += 97
	} else if x == 98 {
		x += 98
	} else if x == 99 {
		x += 99
	} else if x == 100 {
		x += 100
	} else if x == 101 {
		x += 101
	} else if x == 102 {
		x += 102
	} else if x == 103 {
		x += 103
	} else if x == 104 {
		x += 104
	} else if x == 105 {
		x += 105
	} else if x == 106 {
		x += 106
	} else if x == 107 {
		x += 107
	} else if x == 108 {
		x += 108
	} else if x == 109 {
		x += 109
	} else if x == 110 {
		x += 110
	} else if x == 111 {
		x += 111
	} else if x == 112 {
		x += 112
	} else if x == 113 {
		x += 113
	} else if x == 114 {
		x += 114
	} else if x == 115 {
		x += 115
	} else if x == 116 {
		x += 116
	} else if x == 117 {
		x += 117
	} else if x == 118 {
		x += 118
	} else if x == 119 {
		x += 119
	} else if x == 120 {
		x += 120
	} else if x == 121 {
		x += 121
	} else if x == 122 {
		x += 122
	} else if x == 123 {
		x += 123
	} else if x == 124 {
		x += 124
	} else if x == 125 {
		x += 125
	} else if x == 126 {
		x += 126
	} else if x == 127 {
		x += 127
	} else if x == 128 {
		x += 128
	} else if x == 129 {
		x += 129
	} else if x == 130 {
		x += 130
	} else if x == 131 {
		x += 131
	} else if x == 132 {
		x += 132
	} else if x == 133 {
		x += 133
	} else if x == 134 {
		x += 134
	} else if x == 135 {
		x += 135
	} else if x == 136 {
		x += 136
	} else if x == 137 {
		x += 137
	} else if x == 138 {
		x += 138
	} else if x == 139 {
		x += 139
	} else if x == 140 {
		x += 140
	} else if x == 141 {
		x += 141
	} else if x == 142 {
		x += 142
	} else if x == 143 {
		x += 143
	} else if x == 144 {
		x += 144
	} else if x == 145 {
		x += 145
	} else if x == 146 {
		x += 146
	} else if x == 147 {
		x += 147
	} else if x == 148 {
		x += 148
	} else if x == 149 {
		x += 149
	} else if x == 150 {
		x += 150
	} else if x == 151 {
		x += 151
	} else if x == 152 {
		x += 152
	} else if x == 153 {
		x += 153
	} else if x == 154 {
		x += 154
	} else if x == 155 {
		x += 155
	} else if x == 156 {
		x += 156
	} else if x == 157 {
		x += 157
	} else if x == 158 {
		x += 158
	} else if x == 159 {
		x += 159
	} else if x == 160 {
		x += 160
	} else if x == 161 {
		x += 161
	} else if x == 162 {
		x += 162
	} else if x == 163 {
		x += 163
	} else if x == 164 {
		x += 164
	} else if x == 165 {
		x += 165
	} else if x == 166 {
		x += 166
	} else if x == 167 {
		x += 167
	} else if x == 168 {
		x += 168
	} else if x == 169 {
		x += 169
	} else if x == 170 {
		x += 170
	} else if x == 171 {
		x += 171
	} else if x == 172 {
		x += 172
	} else if x == 173 {
		x += 173
	} else if x == 174 {
		x += 174
	} else if x == 175 {
		x += 175
	} else if x == 176 {
		x += 176
	} else if x == 177 {
		x += 177
	} else if x == 178 {
		x += 178
	} else if x == 179 {
		x += 179
	} else if x == 180 {
		x += 180
	} else if x == 181 {
		x += 181
	} else if x == 182 {
		x += 182
	} else if x == 183 {
		x += 183
	} else if x == 184 {
		x += 184
	} else if x == 185 {
		x += 185
	} else if x == 186 {
		x += 186
	} else if x == 187 {
		x += 187
	} else if x == 188 {
		x += 188
	} else if x == 189 {
		x += 189
	} else if x == 190 {
		x += 190
	} else if x == 191 {
		x += 191
	} else if x == 192 {
		x += 192
	} else if x == 193 {
		x += 193
	} else if x == 194 {
		x += 194
	} else if x == 195 {
		x += 195
	} else if x == 196 {
		x += 196
	} else if x == 197 {
		x += 197
	} else if x == 198 {
		x += 198
	} else if x == 199 {
		x += 199
	} else if x == 200 {
		x += 200
	} else if x == 201 {
		x += 201
	} else if x == 202 {
		x += 202
	} else if x == 203 {
		x += 203
	} else if x == 204 {
		x += 204
	} else if x == 205 {
		x += 205
	} else if x == 206 {
		x += 206
	} else if x == 207 {
		x += 207
	} else if x == 208 {
		x += 208
	} else if x == 209 {
		x += 209
	} else if x == 210 {
		x += 210
	} else if x == 211 {
		x += 211
	} else if x == 212 {
		x += 212
	} else if x == 213 {
		x += 213
	} else if x == 214 {
		x += 214
	} else if x == 215 {
		x += 215
	} else if x == 216 {
		x += 216
	} else if x == 217 {
		x += 217
	} else if x == 218 {
		x += 218
	} else if x == 219 {
		x += 219
	} else if x == 220 {
		x += 220
	} else if x == 221 {
		x += 221
	} else if x == 222 {
		x += 222
	} else if x == 223 {
		x += 223
	} else if x == 224 {
		x += 224
	} else if x == 225 {
		x += 225
	} else if x == 226 {
		x += 226
	} else if x == 227 {
		x += 227
	} else if x == 228 {
		x += 228
	} else if x == 229 {
		x += 229
	} else if x == 230 {
		x += 230
	} else if x == 231 {
		x += 231
	} else if x == 232 {
		x += 232
	} else if x == 233 {
		x += 233
	} else if x == 234 {
		x += 234
	} else if x == 235 {
		x += 235
	} else if x == 236 {
		x += 236
	} else if x == 237 {
		x += 237
	} else if x == 238 {
		x += 238
	} else if x == 239 {
		x += 239
	} else if x == 240 {
		x += 240
	} else if x == 241 {
		x += 241
	} else if x == 242 {
		x += 242
	} else if x == 243 {
		x += 243
	} else if x == 244 {
		x += 244
	} else if x == 245 {
		x += 245
	} else if x == 246 {
		x += 246
	} else if x == 247 {
		x += 247
	} else if x == 248 {
		x += 248
	} else if x == 249 {
		x += 249
	} else if x == 250 {
		x += 250
	} else if x == 251 {
		x += 251
	} else if x == 252 {
		x += 252
	} else if x == 253 {
		x += 253
	} else if x == 254 {
		x += 254
	} else if x == 255 {
		x += 255
	} else if x == 256 {
		x += 256
	} else if x == 257 {
		x += 257
	} else if x == 258 {
		x += 258
	} else if x == 259 {
		x += 259
	} else if x == 260 {
		x += 260
	} else if x == 261 {
		x += 261
	} else if x == 262 {
		x += 262
	} else if x == 263 {
		x += 263
	} else if x == 264 {
		x += 264
	} else if x == 265 {
		x += 265
	} else if x == 266 {
		x += 266
	} else if x == 267 {
		x += 267
	} else if x == 268 {
		x += 268
	} else if x == 269 {
		x += 269
	} else if x == 270 {
		x += 270
	} else if x == 271 {
		x += 271
	} else if x == 272 {
		x += 272
	} else if x == 273 {
		x += 273
	} else if x == 274 {
		x += 274
	} else if x == 275 {
		x += 275
	} else if x == 276 {
		x += 276
	} else if x == 277 {
		x += 277
	} else if x == 278 {
		x += 278
	} else if x == 279 {
		x += 279
	} else if x == 280 {
		x += 280
	} else if x == 281 {
		x += 281
	} else if x == 282 {
		x += 282
	} else if x == 283 {
		x += 283
	} else if x == 284 {
		x += 284
	} else if x == 285 {
		x += 285
	} else if x == 286 {
		x += 286
	} else if x == 287 {
		x += 287
	} else if x == 288 {
		x += 288
	} else if x == 289 {
		x += 289
	} else if x == 290 {
		x += 290
	} else if x == 291 {
		x += 291
	} else if x == 292 {
		x += 292
	} else if x == 293 {
		x += 293
	} else if x == 294 {
		x += 294
	} else if x == 295 {
		x += 295
	} else if x == 296 {
		x += 296
	} else if x == 297 {
		x += 297
	} else if x == 298 {
		x += 298
	} else if x == 299 {
		x += 299
	} else if x == 300 {
		x += 300
	} else if x == 301 {
		x += 301
	} else if x == 302 {
		x += 302
	} else if x == 303 {
		x += 303
	} else if x == 304 {
		x += 304
	} else if x == 305 {
		x += 305
	} else if x == 306 {
		x += 306
	} else if x == 307 {
		x += 307
	} else if x == 308 {
		x += 308
	} else if x == 309 {
		x += 309
	} else if x == 310 {
		x += 310
	} else if x == 311 {
		x += 311
	} else if x == 312 {
		x += 312
	} else if x == 313 {
		x += 313
	} else if x == 314 {
		x += 314
	} else if x == 315 {
		x += 315
	} else if x == 316 {
		x += 316
	} else if x == 317 {
		x += 317
	} else if x == 318 {
		x += 318
	} else if x == 319 {
		x += 319
	} else if x == 320 {
		x += 320
	} else if x == 321 {
		x += 321
	} else if x == 322 {
		x += 322
	} else if x == 323 {
		x += 323
	} else if x == 324 {
		x += 324
	} else if x == 325 {
		x += 325
	} else if x == 326 {
		x += 326
	} else if x == 327 {
		x += 327
	} else if x == 328 {
		x += 328
	} else if x == 329 {
		x += 329
	} else if x == 330 {
		x += 330
	} else if x == 331 {
		x += 331
	} else if x == 332 {
		x += 332
	} else if x == 333 {
		x += 333
	} else if x == 334 {
		x += 334
	} else if x == 335 {
		x += 335
	} else if x == 336 {
		x += 336
	} else if x == 337 {
		x += 337
	} else if x == 338 {
		x += 338
	} else if x == 339 {
		x += 339
	} else if x == 340 {
		x += 340
	} else if x == 341 {
		x += 341
	} else if x == 342 {
		x += 342
	} else if x == 343 {
		x += 343
	} else if x == 344 {
		x += 344
	} else if x == 345 {
		x += 345
	} else if x == 346 {
		x += 346
	} else if x == 347 {
		x += 347
	} else if x == 348 {
		x += 348
	} else if x == 349 {
		x += 349
	} else if x == 350 {
		x += 350
	} else if x == 351 {
		x += 351
	} else if x == 352 {
		x += 352
	} else if x == 353 {
		x += 353
	} else if x == 354 {
		x += 354
	} else if x == 355 {
		x += 355
	} else if x == 356 {
		x += 356
	} else if x == 357 {
		x += 357
	} else if x == 358 {
		x += 358
	} else if x == 359 {
		x += 359
	} else if x == 360 {
		x += 360
	} else if x == 361 {
		x += 361
	} else if x == 362 {
		x += 362
	} else if x == 363 {
		x += 363
	} else if x == 364 {
		x += 364
	} else if x == 365 {
		x += 365
	} else if x == 366 {
		x += 366
	} else if x == 367 {
		x += 367
	} else if x == 368 {
		x += 368
	} else if x == 369 {
		x += 369
	} else if x == 370 {
		x += 370
	} else if x == 371 {
		x += 371
	} else if x == 372 {
		x += 372
	} else if x == 373 {
		x += 373
	} else if x == 374 {
		x += 374
	} else if x == 375 {
		x += 375
	} else if x == 376 {
		x += 376
	} else if x == 377 {
		x += 377
	} else if x == 378 {
		x += 378
	} else if x == 379 {
		x += 379
	} else if x == 380 {
		x += 380
	} else if x == 381 {
		x += 381
	} else if x == 382 {
		x += 382
	} else if x == 383 {
		x += 383
	} else if x == 384 {
		x += 384
	} else if x == 385 {
		x += 385
	} else if x == 386 {
		x += 386
	} else if x == 387 {
		x += 387
	} else if x == 388 {
		x += 388
	} else if x == 389 {
		x += 389
	} else if x == 390 {
		x += 390
	} else if x == 391 {
		x += 391
	} else if x == 392 {
		x += 392
	} else if x == 393 {
		x += 393
	} else if x == 394 {
		x += 394
	} else if x == 395 {
		x += 395
	} else if x == 396 {
		x += 396
	} else if x == 397 {
		x += 397
	} else if x == 398 {
		x += 398
	} else if x == 399 {
		x += 399
	} else if x == 400 {
		x += 400
	}
	return x * x
}
