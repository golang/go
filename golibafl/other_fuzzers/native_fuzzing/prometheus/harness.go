package harness

import (
	"errors"
	"io"
	"github.com/prometheus/prometheus/model/labels"
	"github.com/prometheus/prometheus/model/textparse"
	"github.com/prometheus/prometheus/promql/parser"
)

var symbolTable = labels.NewSymbolTable()

func fuzzParseMetricWithContentType(in []byte, contentType string) int {
	p, warning := textparse.New(in, contentType, "", false, false, symbolTable)
	if p == nil || warning != nil {
		// An invalid content type is being passed, which should not happen
		// in this context.
		panic(warning)
	}

	var err error
	for {
		_, err = p.Next()
		if err != nil {
			break
		}
	}
	if errors.Is(err, io.EOF) {
		err = nil
	}

	return 0
}

func harness(data []byte) int {
	if len(data) < 2 {
		return 0
	}
	switch data[0] {
	case 0x00:
		parser.ParseExpr(string(data[1:]))
	case 0x01:
		parser.ParseMetricSelector(string(data[1:]))
	case 0x02:
		fuzzParseMetricWithContentType(data[1:], "text/plain")
	case 0x03:
		fuzzParseMetricWithContentType(data[1:], "application/openmetrics-text")
	}
	return 0
}

