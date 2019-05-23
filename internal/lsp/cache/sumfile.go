package cache

import (
	"context"
	"go/token"
)

// sumFile holds all of the information we know about a sum file.
type sumFile struct {
	fileBase
}

func (*sumFile) GetToken(context.Context) *token.File { return nil }
func (*sumFile) setContent(content []byte)            {}
func (*sumFile) filename() string                     { return "" }
func (*sumFile) isActive() bool                       { return false }
