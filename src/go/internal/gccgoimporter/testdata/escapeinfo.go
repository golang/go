// Test case for escape info in export data. To compile and extract .gox file:
// gccgo -fgo-optimize-allocs -c escapeinfo.go
// objcopy -j .go_export escapeinfo.o escapeinfo.gox

package escapeinfo

type T struct{ data []byte }

func NewT(data []byte) *T {
	return &T{data}
}

func (*T) Read(p []byte) {}
