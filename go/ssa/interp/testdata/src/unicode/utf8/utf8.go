package utf8

func DecodeRuneInString(string) (rune, int)

func DecodeRune(b []byte) (rune, int) {
	return DecodeRuneInString(string(b))
}

const RuneError = '\uFFFD'
