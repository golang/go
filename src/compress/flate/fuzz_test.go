package flate

import (
	"bytes"
	"flag"
	"io"
	"os"
	"strconv"
	"testing"
)

// Fuzzing tweaks:
var fuzzStartF = flag.Int("start", HuffmanOnly, "Start fuzzing at this level")
var fuzzEndF = flag.Int("end", BestCompression, "End fuzzing at this level (inclusive)")
var fuzzMaxF = flag.Int("max", 1<<20, "Maximum input size")

func TestMain(m *testing.M) {
	flag.Parse()
	os.Exit(m.Run())
}

// FuzzEncoding tests the fuzzer by doing roundtrips.
// Every input is run through the fuzzer at every level.
// Note: When running the fuzzer, it may hit the 10-second timeout on slower CPUs.
func FuzzEncoding(f *testing.F) {
	startFuzz := *fuzzStartF
	endFuzz := *fuzzEndF
	maxSize := *fuzzMaxF

	decoder := NewReader(nil)
	buf, buf2 := new(bytes.Buffer), new(bytes.Buffer)
	encs := make([]*Writer, endFuzz-startFuzz+1)
	for i := range encs {
		var err error
		encs[i], err = NewWriter(nil, i+startFuzz)
		if err != nil {
			f.Fatal(err.Error())
		}
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > maxSize {
			return
		}
		for level := startFuzz; level <= endFuzz; level++ {
			if level == DefaultCompression {
				continue // Already covered.
			}
			msg := "level " + strconv.Itoa(level) + ":"
			buf.Reset()
			fw := encs[level-startFuzz]
			fw.Reset(buf)
			n, err := fw.Write(data)
			if n != len(data) {
				t.Fatal(msg + "short write")
			}
			if err != nil {
				t.Fatal(msg + err.Error())
			}
			err = fw.Close()
			if err != nil {
				t.Fatal(msg + err.Error())
			}
			compressed := buf.Bytes()
			err = decoder.(Resetter).Reset(buf, nil)
			if err != nil {
				t.Fatal(msg + err.Error())
			}
			data2, err := io.ReadAll(decoder)
			if err != nil {
				t.Fatal(msg + err.Error())
			}
			if !bytes.Equal(data, data2) {
				t.Fatal(msg + "decompressed not equal")
			}

			// Do it again...
			msg = "level " + strconv.Itoa(level) + " (reset):"
			buf2.Reset()
			fw.Reset(buf2)
			n, err = fw.Write(data)
			if n != len(data) {
				t.Fatal(msg + "short write")
			}
			if err != nil {
				t.Fatal(msg + err.Error())
			}
			err = fw.Close()
			if err != nil {
				t.Fatal(msg + err.Error())
			}
			compressed2 := buf2.Bytes()
			err = decoder.(Resetter).Reset(buf2, nil)
			if err != nil {
				t.Fatal(msg + err.Error())
			}
			data2, err = io.ReadAll(decoder)
			if err != nil {
				t.Fatal(msg + err.Error())
			}
			if !bytes.Equal(data, data2) {
				t.Fatal(msg + "decompressed not equal")
			}
			// Determinism checks will usually not be reproducible,
			// since it often relies on the internal state of the compressor.
			if !bytes.Equal(compressed, compressed2) {
				t.Fatal(msg + "non-deterministic output")
			}
		}
	})
}
