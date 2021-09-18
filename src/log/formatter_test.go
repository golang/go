package log

import (
	"bytes"
	"fmt"
	"testing"
)

type testFormatter struct {
}

func (f *testFormatter) Format(entry *Entry) ([]byte, error) {
	testString := "formatter message: " + entry.Message()
	return []byte(testString), nil
}

func ExampleLoggerWithFormatter() {
	var (
		buf    bytes.Buffer
		logger = New(&buf, "logger: ", Lshortfile)
	)
	logger.SetFormatter(&testFormatter{})
	logger.Info("Hello, log file!")

	fmt.Print(&buf)
	// Output:
	// formatter message: Hello, log file!
}

func BenchmarkPrintlnWithFormatter(b *testing.B) {
	const testString = "test"
	var buf bytes.Buffer
	l := New(&buf, "", 0)
	l.SetFormatter(&testFormatter{})

	for i := 0; i < b.N; i++ {
		buf.Reset()
		l.Println(testString)
	}
}
