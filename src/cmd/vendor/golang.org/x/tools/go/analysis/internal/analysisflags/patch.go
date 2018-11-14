package analysisflags

import "cmd/internal/objabi"

// This additional file changes the behavior of the vendored code.

func init() { addVersionFlag = objabi.AddVersionFlag }
