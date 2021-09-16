package other

import (
	"bytes"
	renamed_context "context"
)

type Interface interface {
	Get(renamed_context.Context) *bytes.Buffer
}
