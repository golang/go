package stub

import "io"

func main() {
	var br io.ByteWriter
	br = &byteWriter{} //@suggestedfix("&", "quickfix", "")
}

type byteWriter struct{}
