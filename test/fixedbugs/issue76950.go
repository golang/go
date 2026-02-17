// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func MatchLog(input string) bool {
	pos := 0
	n := len(input)
	matchState := -1
	var c byte

	goto State12

State8:
	goto State65

State12:
	if pos >= n {
		goto End
	}
	c = input[pos]
	switch {
	case c >= 0x09 && c <= 0x0A || c >= 0x0C && c <= 0x0D || c == ' ':
	case c >= '0' && c <= '9':
	case c >= 'A' && c <= 'Z' || c == '_' || c >= 'b' && c <= 'z':
	case c == '[':
		goto State8
	case c == 'a':
	default:
		goto End
	}

State64:
	matchState = 179
	if pos >= n {
		goto End
	}
	pos = n
	goto State64

State65:

State66:
	matchState = 181
	if pos >= n {
		goto End
	}
	pos = n
	goto State66

End:
	if matchState != -1 {
		switch matchState {
		case 178:
		case 156:
		case 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175:
		case 176, 177, 181, 182, 183:
		case 179, 184:
		case 180:
		}
		return true
	}
	return false
}
