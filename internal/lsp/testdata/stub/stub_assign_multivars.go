package stub

import "io"

func main() {
	var br io.ByteWriter
	var i int
	i, br = 1, &multiByteWriter{} //@suggestedfix("&", "refactor.rewrite")
}

type multiByteWriter struct{}
