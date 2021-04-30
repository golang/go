package basiclit

func _() {
	var a int // something for lexical completions

	_ = "hello." //@complete(".")

	_ = 1 //@complete(" //")

	_ = 1. //@complete(".")

	_ = 'a' //@complete("' ")

	_ = 'a' //@hover("'a'", "'a', U+0061, LATIN SMALL LETTER A")
	_ = 0x61 //@hover("0x61", "'a', U+0061, LATIN SMALL LETTER A")

	_ = '\u2211' //@hover("'\\u2211'", "'∑', U+2211, N-ARY SUMMATION")
	_ = 0x2211 //@hover("0x2211", "'∑', U+2211, N-ARY SUMMATION")
	_ = "foo \u2211 bar" //@hover("\\u2211", "'∑', U+2211, N-ARY SUMMATION")

	_ = '\a' //@hover("'\\a'", "U+0007, control")
	_ = "foo \a bar" //@hover("\\a", "U+0007, control")

	_ = '\U0001F30A' //@hover("'\\U0001F30A'", "'🌊', U+1F30A, WATER WAVE")
	_ = 0x0001F30A //@hover("0x0001F30A", "'🌊', U+1F30A, WATER WAVE")
	_ = "foo \U0001F30A bar" //@hover("\\U0001F30A", "'🌊', U+1F30A, WATER WAVE")

	_ = '\x7E' //@hover("'\\x7E'", "'~', U+007E, TILDE")
	_ = "foo \x7E bar" //@hover("\\x7E", "'~', U+007E, TILDE")
	_ = "foo \a bar" //@hover("\\a", "U+0007, control")

	_ = '\173' //@hover("'\\173'", "'{', U+007B, LEFT CURLY BRACKET")
	_ = "foo \173 bar" //@hover("\\173", "'{', U+007B, LEFT CURLY BRACKET")
	_ = "foo \173 bar \u2211 baz" //@hover("\\173", "'{', U+007B, LEFT CURLY BRACKET")
	_ = "foo \173 bar \u2211 baz" //@hover("\\u2211", "'∑', U+2211, N-ARY SUMMATION")
	_ = "foo\173bar\u2211baz" //@hover("\\173", "'{', U+007B, LEFT CURLY BRACKET")
	_ = "foo\173bar\u2211baz" //@hover("\\u2211", "'∑', U+2211, N-ARY SUMMATION")

	// search for runes in string only if there is an escaped sequence
	_ = "hello" //@hover("\"hello\"", "")

	// incorrect escaped rune sequences
	_ = '\0' //@hover("'\\0'", "")
	_ = '\u22111' //@hover("'\\u22111'", "")
	_ = '\U00110000' //@hover("'\\U00110000'", "")
	_ = '\u12e45'//@hover("'\\u12e45'", "")
	_ = '\xa' //@hover("'\\xa'", "")
	_ = 'aa' //@hover("'aa'", "")

	// other basic lits
	_ = 1 //@hover("1", "")
	_ = 1.2 //@hover("1.2", "")
	_ = 1.2i //@hover("1.2i", "")
	_ = 0123 //@hover("0123", "")
	_ = 0x1234567890 //@hover("0x1234567890", "")
}
