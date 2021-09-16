package stub

import "io"

func main() {
	var br io.ByteWriter
	br = &byteWriter{} //@suggestedfix("&", "refactor.rewrite")
}

type byteWriter struct{}
